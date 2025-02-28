
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
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


## Local packages

from models.deepcaps import DeepCapsModel
from models.capsnet import CapsNet
from models.convcaps import *
from models.parsecaps import *
from Utils.dataloader import *
from Utils.helpers import *
from Utils.plot import plot_loss_acc2, plot_reconstruction
import cfg
from layers import Loss



def train(device=torch.device('cpu'), learning_rate=1e-3, num_epochs=500, decay_step=20, gamma=0.98,
          checkpoint_folder=None, checkpoint_name=None, load_checkpoint=False, graphs_folder=None, 
          finetune=False, finetune_layers=None):
    
    # SEED        
    torch.manual_seed(42)
    np.random.seed(42)
    
    if cfg.DATASET == 'Cifar10':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = Cifar10(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()
    elif cfg.DATASET == 'MNIST':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = MNIST(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()
    elif cfg.DATASET == 'Skin':
        
        transform = A.Compose([
            A.PadIfNeeded(512, 512),
            A.CenterCrop(width=512, height=512),
            A.Resize(width=224, height=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        train_dataloader_ph2derm7pt_dlv3_ft, val_dataloader_ph2derm7pt_dlv3_ft, _ = create_dataloaders(
            path = cfg.DATASET_FOLDER, batch_size = cfg.BATCH_SIZE,
            train_transform=train_transform,
            val_transform=val_transform)
        test_dataloader_ph2_dlv3_ft, _ = create_dataloader_for_evaluation(
            path = cfg.DATASET_FOLDER,
            transform=transform)
        img_size, in_channel, image_shape = 224, 3, (cfg.BATCH_SIZE, 3, 224, 224)
        
    elif cfg.DATASET == 'BrainTumor':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = BrainTumor(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()  
    elif cfg.DATASET == 'mini_imagenet':
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = mini_imagenet(data_path=cfg.DATASET_FOLDER,
                                                                batch_size=cfg.BATCH_SIZE,
                                                                shuffle=True)()
    elif cfg.DATASET == "tissuemnist":
        train_loader, test_loader, img_size, num_classes, in_channel, image_shape = medmnist_dataset(data_flag=cfg.DATASET,
                                                                                                     batch_size=cfg.BATCH_SIZE,
                                                                                                     downdload=True)()
    else:
        assert False, "Dataset not found!"
    
    # print("num_class: ", num_classes)
    
    # MODELS
    if cfg.MODEL == 'CapsNet':
        Model = nn.DataParallel(CapsNet(num_class=num_classes,img_height=img_size, img_width=img_size, in_channel=in_channel, device=device,routing=cfg.ROUTING)).to(device)
        print("Successfully loaded CapsNet model")
    elif cfg.MODEL == 'OrthCaps_shallow':
        Model = nn.DataParallel(ShallowNet(num_class=num_classes, input_shape=image_shape, layernum=cfg.LAYERNUM, i_c=in_channel, similarity=cfg.SIMILARITY,activation = cfg.ACTIVATION,device=device)).to(device)
        print("Successfully loaded OrthCaps_shallow model")
    elif cfg.MODEL == 'ParseCaps':
        Model = nn.DataParallel(ParseCaps(num_class=num_classes, input_shape=image_shape, dim=cfg.DIM, conceptnum=8, kwargs_dict=cfg.KWARGS_DICT, device=device)).to(device)
        print("Successfully loaded ParseCaps model")
    elif cfg.MODEL == 'DeepCaps':
        Model = nn.DataParallel(DeepCapsModel(num_class=num_classes, img_height=image_shape[-1], img_width=image_shape[-1], in_channel=in_channel, device=device)).to(device)
        print("Successfully loaded DeepCaps model")
        # print(Model)
    else:
        assert False, "Model not found!"
    
    checkpoint_path =  checkpoint_folder + checkpoint_name 
    #load the current checkpoint
    if load_checkpoint:
        Model.load_state_dict(torch.load('saves/saved_models/mini_imagenet/240512/ParseCaps_284_45.03.pth'))
        print("Checkpoint loaded!")
        
        # load partial weights if mismatch
        # pretrained_dict = torch.load(checkpoint_path+'_36.58.pth')
        # model_dict = Model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == pretrained_dict[k].size()}
        # model_dict.update(pretrained_dict)
        # Model.load_state_dict(model_dict)
    else:
        print("Checkpoint not loaded!")
    
    
    if finetune:
        if finetune_layers:
            for name, param in Model.named_parameters():
                if name in finetune_layers:
                    param.requires_grad = False
                    print(f'Layer {name} is frozen')
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=100, verbose=True)
    
    # OPTIMIZER
    optimizer = AdamW(Model.parameters(), lr=learning_rate, weight_decay=cfg.WEIGHT_DECAY)
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

    
    for epoch_idx in range(num_epochs):
        
        if cfg.MODE == 'train':
            print('--------------Beginning training and validating--------------')
            print(f"Training and testing for epoch {epoch_idx} began with LR : {get_learning_rate(optimizer)} BS : {cfg.BATCH_SIZE}")
    
            batch_loss = 0
            batch_accuracy = 0
            batch_idx = 0

            Model.train() #train mode
            for batch_idx, (train_data, labels) in tqdm(enumerate(train_loader)): 
                
                labels = labels.type(torch.int64)
                onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
                data, onehot_labels, labels= train_data.to(device), onehot_labels.to(device), labels.to(device)
                
                optimizer.zero_grad()

                indices, x_recon, y_pred, concept = Model(data, onehot_labels)
                
                labels = labels.squeeze()
                # loss = cross_entropy_loss(labels, y_pred, data, x_recon)             # ParseCaps
                loss = Loss.caps_loss(onehot_labels, y_pred, data, x_recon, False)     # OrthCaps
                
                loss.backward()
                
                optimizer.step()

                batch_loss = batch_loss + loss.item()
                batch_accuracy = batch_accuracy + accuracy_calc(predictions=indices, labels=labels)
                
                # if batch_idx % 500 == 0:
                #     # print(f"Batch : {batch_idx}, Testing Accuracy : {batch_accuracy/(batch_idx+1)}, Testing Loss : {batch_loss/(batch_idx+1)}")
                #     torch.save(Model.state_dict(), checkpoint_path + str(epoch_idx) + '_' + str(batch_idx) + '.pth')
                
                # torch.cuda.empty_cache()


            train_epoch_accuracy = batch_accuracy/(batch_idx+1)
            train_avg_batch_loss = batch_loss/(batch_idx+1)
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
                data, onehot_labels,labels = test_data.to(device), onehot_labels.to(device), labels.to(device)

                indices, x_recon, y_pred, concept = Model(data, onehot_labels)
                
                labels = labels.squeeze()
                # loss = cross_entropy_loss(labels, y_pred, data, x_recon)
                
                loss = Loss.caps_loss(onehot_labels, y_pred, data, x_recon, False)
                batch_loss += loss.item()
                batch_accuracy += accuracy_calc(predictions=indices, labels=labels)
                

                    

            test_epoch_accuracy = batch_accuracy/(batch_idx+1)
            test_avg_batch_loss = batch_loss/(batch_idx+1)
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
            torch.save(Model.state_dict(), checkpoint_path + str(epoch_idx) + '_' + str(round(test_epoch_accuracy, 2)) + '.pth')
            # print("Saved best model at epoch %d"%(epoch_idx))
             
            # early_stopping(test_avg_batch_loss)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            
            total_params = count_parameters(Model)
            print(f'Total Parameters: {total_params}')
            
            torch.cuda.empty_cache()



if __name__ == '__main__':
    
    # world_size = 8  # 假设我们有4个GPU
    # rank = 0  # 此处应由启动脚本动态提供正确的值


    # os.environ['MASTER_ADDR'] = 'localhost' # 0号机器的IP
    # os.environ['MASTER_PORT'] = '19198' # 0号机器的可用端口
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # world_size = torch.cuda.device_count()
    # os.environ['WORLD_SIZE'] = str(world_size)
    
    # time_start = time.time()    
    # mp.spawn(fn=main, args=(args, ), nprocs=world_size)    
    # time_elapsed = time.time() - time_start
    # print(f'\ntime elapsed: {time_elapsed:.2f} seconds.')

    train(device=get_device(), learning_rate=cfg.LEARNING_RATE, num_epochs=cfg.NUM_EPOCHS, decay_step=cfg.DECAY_STEP, gamma=cfg.DECAY_GAMMA,
          checkpoint_folder=cfg.CHECKPOINT_FOLDER,checkpoint_name=cfg.CHECKPOINT_NAME, load_checkpoint=cfg.LOAD_TRAIN, graphs_folder=cfg.GRAPHS_FOLDER,
          finetune=cfg.FINETUNE, finetune_layers=cfg.FINETUNE_LAYER)

