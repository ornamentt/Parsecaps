
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
import time
from tqdm import tqdm
from thop import profile
import numpy as np
from icecream import ic
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,MultiStepLR
from torch.optim import AdamW,Adam,SGD
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

## DDP packages
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.cuda.amp import GradScaler
# from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

## Local packages

# from models.deepcaps_crelu import DeepCapsModel_crelu
# from models.deepcaps import DeepCapsModel
# from models.capsnet import CapsNet
# from models.convcaps import *
from models.parsecaps import *
from Utils.dataloader import *
from Utils.helpers import *
from Utils.plot import plot_loss_acc2, plot_reconstruction
import cfg
from layers import Loss


# local_rank: 当前进程序号
# world_size： 总进程数
def init_ddp(local_rank):
    '''
    进程初始化函数
    '''
    # 有了这一句话后，在转换device的时候直接使用 a=a.cuda() 即可，否则要用a=a.cuda(local_rank)
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

# 对多个进程的计算结果进行汇总，如loss，评价指标
def reduce_tensor(tensor: torch.Tensor):
    '''
    对多个进程计算的多个Tensor 类型的输出值取平均操作
    '''
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

# def get_ddp_generator(seed=3407):
#     '''
#     对每个进程使用不同的随机种子，增强训练的随机性
#     '''
#     local_rank = dist.get_rank()
#     g = torch.Generator()
#     g.manual_seed(seed + local_rank)
#     return g


def train(Model, train_loader, test_loader, num_classes, args):
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=100, verbose=True)
    
    # OPTIMIZER
    optimizer = Adam(Model.parameters(), lr=args['learning_rate'], weight_decay=cfg.WEIGHT_DECAY)
    # optimizer = Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr=learning_rate, weight_decay=cfg.WEIGHT_DECAY)
    # optimizer = SGD(Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY, nesterov=True)
    # optimizer = SGD(filter(lambda p: p.requires_grad, Model.parameters()), lr=learning_rate, momentum=0.9)
    
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-5)
    # scheduler=CosineAnnealingWarmRestarts(optimizer,T_0=cfg.NUM_EPOCHS,eta_min=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode= 'min', factor=cfg.FACTOR, patience=cfg.PATIENCE, verbose=True)
    # scheduler = MultiStepLR(optimizer, milestones=[200, 300, 400], gamma=0.1)
    
    cosine_scheduler = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=5, after_scheduler=scheduler)

    cross_entropy_loss = Loss.CapsuleLossCrossEntropy(reconstruction=cfg.IFR, rl=cfg.RECONSTRUCTION)
    
    # INITIALIZE
    training_loss_list = []
    training_acc_list = []
    testing_loss_list = []
    testing_acc_list = []
    
    # beta = cfg.BETA
    # cutmix_prob = cfg.CUTMIX_PROB
    
    checkpoint_path =  args['checkpoint_folder'] + args['checkpoint_name']
    
    for epoch_idx in range(args['num_epochs']):
        
        train_loader.sampler.set_epoch(epoch_idx) # 训练时每次的sampling顺序不同
        
        if cfg.MODE == 'train':
            
            if dist.get_rank() == 0:
                print('--------------Beginning training and validating--------------')
                print(f"Training and testing for epoch {epoch_idx} began with LR : {get_learning_rate(optimizer)} BS : {cfg.BATCH_SIZE}")
    
            batch_loss = 0
            batch_accuracy = 0
            batch_idx = 0

            Model.train() #train mode
            for batch_idx, (train_data, labels) in tqdm(enumerate(train_loader)): 
                
                labels = labels.type(torch.int64)
                onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
                data, onehot_labels, labels= train_data.cuda(), onehot_labels.cuda(), labels.cuda()
                
                optimizer.zero_grad()

                indices, x_recon, y_pred, concept = Model(data, onehot_labels)
                
                
                loss = cross_entropy_loss(labels, y_pred, data, x_recon)
                # loss = Loss.caps_loss(onehot_labels, y_pred, data, x_recon, cfg.RECON)
                
                reduced_loss = reduce_tensor(loss.data) # 对并行进程计算的多个loss 取平均
                
                loss.backward()
                
                optimizer.step()

                # batch_loss = batch_loss + loss.item()
                batch_loss += reduced_loss.item()
                
                # batch_accuracy = batch_accuracy + accuracy_calc(predictions=indices, labels=labels)
                reduced_accuracy = reduce_tensor(torch.tensor(accuracy_calc(predictions=indices, labels=labels)).cuda())
                batch_accuracy += reduced_accuracy.item()
                
                if (batch_idx + 1) % 100 == 0 and dist.get_rank() == 0:
                    # print(f"Batch : {batch_idx}, Testing Accuracy : {batch_accuracy/(batch_idx+1)}, Testing Loss : {batch_loss/(batch_idx+1)}")
                    torch.save(Model.state_dict(), checkpoint_path + str(epoch_idx) + '_' + str(batch_idx) + '.pth')
                
                # torch.cuda.empty_cache()


            train_epoch_accuracy = batch_accuracy/(batch_idx+1)
            train_avg_batch_loss = batch_loss/(batch_idx+1)
            
            if dist.get_rank() == 0:
                print(f"Epoch : {epoch_idx}, Training Accuracy : {train_epoch_accuracy}, Training Loss : {train_avg_batch_loss}")

            training_loss_list.append(train_avg_batch_loss)
            training_acc_list.append(train_epoch_accuracy)
            
            torch.cuda.empty_cache()

            #Testing
            batch_loss = 0
            batch_accuracy = 0
            batch_idx = 0

            Model.eval() #eval mode
            for batch_idx, (test_data, labels) in tqdm(enumerate(test_loader)): #from testing dataset

                labels = labels.type(torch.int64)
                # ic(num_classes)
                onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
                data, onehot_labels,labels = test_data.cuda(), onehot_labels.cuda(), labels.cuda()

                indices, x_recon, y_pred, concept = Model(data, onehot_labels)
                
                loss = cross_entropy_loss(labels, y_pred, data, x_recon)
                
                reduced_loss = reduce_tensor(loss.data)
                batch_loss += reduced_loss.item()
                
                reduced_accuracy = reduce_tensor(torch.tensor(accuracy_calc(predictions=indices, labels=labels)).cuda())
                batch_accuracy += reduced_accuracy.item()
                
                # loss = Loss.caps_loss(onehot_labels, y_pred, data, x_recon, 0.0005)
                # batch_loss += loss.item()
                # batch_accuracy += accuracy_calc(predictions=indices, labels=labels)
                
      
            test_epoch_accuracy = batch_accuracy/(batch_idx+1)
            test_avg_batch_loss = batch_loss/(batch_idx+1)
            
            if dist.get_rank() == 0:
                print(f"Epoch : {epoch_idx}, Validating Accuracy : {test_epoch_accuracy}, Validating Loss : {test_avg_batch_loss}")

            testing_loss_list.append(test_avg_batch_loss)
            testing_acc_list.append(test_epoch_accuracy)

            cosine_scheduler.step(metrics=test_avg_batch_loss)
            scheduler.step(metrics=test_avg_batch_loss)
            # scheduler.step()

            # if not graphs_folder is None and epoch_idx%cfg.GRAPH==0 and epoch_idx>0:
            #     with torch.no_grad():
            #         plot_loss_acc2(path=graphs_folder, num_epoch=epoch_idx, train_accuracies=training_acc_list, train_losses=training_loss_list,
            #                     test_accuracies=testing_acc_list, test_losses=testing_loss_list,plot_name=cfg.PLOT_NAME)
                    
            #         #   plot_reconstruction(path=graphs_folder, num_epoch=epoch_idx, original_images=data.detach(), reconstructed_images=reconstructed.detach())
            #                             # ,predicted_classes=indices.detach(), true_classes=labels.detach())
            #         print("Saved graph at epoch %d"%(epoch_idx))
            #         torch.cuda.empty_cache()
            
                
            # if epoch_idx > 10 and test_epoch_accuracy == max(testing_acc_list):
            
            # if test_epoch_accuracy == max(testing_acc_list):
            # torch.save(Model.state_dict(), checkpoint_path+str(test_epoch_accuracy)+'.pth')
            # print("Saved best model at epoch %d"%(epoch_idx))
             
            # early_stopping(test_avg_batch_loss)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            
            total_params = count_parameters(Model)
            if dist.get_rank() == 0:
                print(f'Total Parameters: {total_params}')
            
            torch.cuda.empty_cache()
            

            

def main(local_rank, args): # 参数列表更新
    
    init_ddp(local_rank) # 进程初始化
    
    # SEED        
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataset
    if cfg.DATASET == 'Cifar10':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = Cifar10(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()
    elif cfg.DATASET == 'MNIST':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = MNIST(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()
    # elif cfg.DATASET == 'Skin':
        
    #     transform = A.Compose([
    #         A.PadIfNeeded(512, 512),
    #         A.CenterCrop(width=512, height=512),
    #         A.Resize(width=224, height=224),
    #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         ToTensorV2(),
    #     ])
    #     train_dataloader_ph2derm7pt_dlv3_ft, val_dataloader_ph2derm7pt_dlv3_ft, _ = data_setup.create_dataloaders(
    #         params=PH2Derm7pt_DLV3_FT_params,
    #         train_transform=train_transform,
    #         val_transform=val_transform)
    #     test_dataloader_ph2_dlv3_ft, _ = data_setup.create_dataloader_for_evaluation(params=PH2_DLV3_FT_params,
    #                                                                          transform=transform)
    #     img_size, in_channel, image_shape = 224, 3, (cfg.BATCH_SIZE, 3, 224, 224)
    elif cfg.DATASET == 'BrainTumor':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = BrainTumor(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()  
    elif cfg.DATASET == 'mini_imagenet':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = mini_imagenet_ddp(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()
    else:
        assert False, "Dataset not found!"
    
    
    # MODELS
    if cfg.MODEL == 'CapsNet':
        Model = CapsNet(num_class=num_classes,img_height=img_size, img_width=img_size, in_channel=in_channel, routing=cfg.ROUTING)
        if dist.get_rank() == 0:
            print("Successfully loaded CapsNet model")
    elif cfg.MODEL == 'OrthCaps_shallow':
        Model = ShallowNet(num_class=num_classes, input_shape=image_shape, layernum=cfg.LAYERNUM, i_c=in_channel, similarity=cfg.SIMILARITY, activation = cfg.ACTIVATION)
        if dist.get_rank() == 0:
            print("Successfully loaded OrthCaps_shallow model")
    elif cfg.MODEL == 'ParseCaps':
        Model = ParseCaps(num_class=num_classes, input_shape=image_shape, dim=cfg.DIM, similarity_threshold=cfg.SIMILARITY, kwargs_dict=cfg.KWARGS_DICT)
        if dist.get_rank() == 0:
            print("Successfully loaded ParseCaps model")
    elif cfg.MODEL == 'DeepUCaps':
        Model = DeepUCaps(num_class=num_classes, input_shape=image_shape, dim=cfg.DIM, similarity_threshold=cfg.SIMILARITY, kwargs_dict=cfg.KWARGS_DICT)
        if dist.get_rank() == 0:
            print("Successfully loaded DeepUCaps model")
        # print(Model)
    else:
        assert False, "Model not found!"

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    # 设置所有的GPU的随机种子
    torch.cuda.manual_seed_all(42)
    Model.apply(weights_init) # Model weight initialize

    # checkpoint_path =  args['checkpoint_folder'] + args['checkpoint_name']
    
    #load the current checkpoint
    if args['load_checkpoint']:
        Model.load_state_dict(torch.load('./saves/ImageNet/ParseCaps_0_6000.pth'))
        if dist.get_rank() == 0:
            print("Checkpoint loaded!")
        
        # load partial weights if mismatch
        # pretrained_dict = torch.load(checkpoint_path+'_36.58.pth')
        # model_dict = Model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == pretrained_dict[k].size()}
        # model_dict.update(pretrained_dict)
        # Model.load_state_dict(model_dict)
    else:
        if dist.get_rank() == 0:
            print("Checkpoint not loaded!")
    
    if args['finetune']:
        if args['finetune_layers']:
            for name, param in Model.named_parameters():
                if name in args['finetune_layers']:
                    param.requires_grad = False
                    if dist.get_rank() == 0:
                        print(f'Layer {name} is frozen')    
        
    Model.cuda()
    Model = nn.SyncBatchNorm.convert_sync_batchnorm(Model) # BN层同步，如果单张卡上的batchSize很大可能会导致训练速度下降。
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(Model, device_ids=[local_rank], output_device=local_rank)
    
    # scaler = GradScaler() # 用于混合精度训练
    
    train(Model=Model, train_loader=train_loader, test_loader=test_loader, num_classes=num_classes, args=args)
    
    
    dist.destroy_process_group() # 消除进程组
    
        

if __name__ == '__main__':
    
    # world_size = 8  # 假设我们有4个GPU
    # rank = 0  # 此处应由启动脚本动态提供正确的值

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', default='0,1,2,3,4,5,6,7', type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument('-learning_rate', default=cfg.LEARNING_RATE, type=float, help='learning rate for training')
    parser.add_argument('-num_epochs', default=cfg.NUM_EPOCHS, type=int, help='number of epochs for training')
    parser.add_argument('-dacay_step', default=cfg.DECAY_STEP, type=int, help='decay step for learning rate')
    parser.add_argument('-gamma', default=cfg.DECAY_GAMMA, type=float, help='decay gamma for learning rate')
    parser.add_argument('-checkpoint_folder', default=cfg.CHECKPOINT_FOLDER, type=str, help='folder to save checkpoints')
    parser.add_argument('-checkpoint_name', default=cfg.CHECKPOINT_NAME, type=str, help='name of checkpoints')
    parser.add_argument('-load_checkpoint', default=cfg.LOAD_TRAIN, type=bool, help='load checkpoint or not')
    parser.add_argument('-graphs_folder', default=cfg.GRAPHS_FOLDER, type=str, help='folder to save graphs')
    parser.add_argument('-finetune', default=cfg.FINETUNE, type=bool, help='finetune or not')
    parser.add_argument('-finetune_layers', default=cfg.FINETUNE_LAYER, type=list, help='layers to finetune')
    
    args = parser.parse_args()
    args = {**vars(args)}

    os.environ['MASTER_ADDR'] = 'localhost' # 0号机器的IP
    os.environ['MASTER_PORT'] = '19198' # 0号机器的可用端口
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    
    time_start = time.time()    
    mp.spawn(fn=main, args=(args, ), nprocs=world_size)    
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds.')
    
    

