import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from thop import profile
import numpy as np
from icecream import ic
import torch
torch.backends.cudnn.enabled = False
# from torch import autograd
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,MultiStepLR
from torch.optim import AdamW,Adam,SGD

# from models.deepcaps_crelu import DeepCapsModel_crelu
from models.deepcaps import DeepCapsModel
from models.capsnet import CapsNet
from models.convcaps import *
from models.parsecaps import *
from Utils.dataloader import *
from Utils.helpers import *
from Utils.plot import plot_loss_acc2, plot_reconstruction
import cfg
from layers import Loss


if cfg.DATASET == 'Cifar10':
    train_loader, test_loader, img_size, num_class, in_channel, image_shape = Cifar10(data_path=cfg.DATASET_FOLDER,
                                                               batch_size=cfg.BATCH_SIZE,
                                                               shuffle=True)()
elif cfg.DATASET == 'MNIST':
    train_loader, test_loader, img_size, num_class, in_channel, image_shape = MNIST(data_path=cfg.DATASET_FOLDER,
                                                               batch_size=cfg.BATCH_SIZE,
                                                               shuffle=True)()
elif cfg.DATASET == 'BrainTumor':
    train_loader, test_loader, img_size, num_class, in_channel, image_shape = BrainTumor(data_path=cfg.DATASET_FOLDER,
                                                               batch_size=cfg.BATCH_SIZE,
                                                               shuffle=True)()  
else:
    assert False, "Dataset not found!"
    
def check_device(model):
    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            print('Parameter not on CUDA:', name, param.device)
    for name, buf in model.named_buffers():
        if buf.device.type != 'cuda':
            print('Buffer not on CUDA:', name, buf.device)

def test(img_size=32, device=torch.device('cpu'), learning_rate=1e-3, num_epochs=500, decay_step=20, gamma=0.98,
          num_classes=10, checkpoint_folder=None, checkpoint_name=None, load_checkpoint=False, graphs_folder=None, finetune=False, finetune_layers=None):

            print('--------------Beginning testing--------------')
            
            if cfg.MODEL == 'CapsNet':
                Model = nn.DataParallel(CapsNet(num_class=num_classes,img_height=img_size, img_width=img_size, in_channel=in_channel, device=device,routing=cfg.ROUTING)).to(device)
                print("Successfully loaded CapsNet model")
            elif cfg.MODEL == 'OrthCaps_shallow':
                Model = nn.DataParallel(ShallowNet(num_class=num_classes, input_shape=image_shape, layernum=cfg.LAYERNUM, i_c=in_channel, similarity=cfg.SIMILARITY,activation = cfg.ACTIVATION,device=device)).to(device)
                print("Successfully loaded OrthCaps_shallow model")
            elif cfg.MODEL == 'ParseCaps':
                Model = nn.DataParallel(ParseCaps(num_class=num_classes, input_shape=image_shape, dim=cfg.DIM, similarity_threshold=cfg.SIMILARITY, kwargs_dict=cfg.KWARGS_DICT, device=device)).to(device)
                print("Successfully loaded ParseCaps model")
            elif cfg.MODEL == 'DeepUCaps':
                Model = nn.DataParallel(DeepUCaps(num_class=num_classes, input_shape=image_shape, dim=cfg.DIM, similarity_threshold=cfg.SIMILARITY, kwargs_dict=cfg.KWARGS_DICT, device=device)).to(device)
                print("Successfully loaded DeepUCaps model")
                # print(Model)
            else:
                assert False, "Model not found!"
            
        
            checkpoint_path =  checkpoint_folder + checkpoint_name 
            path = "/home/gengxinyu/codes/TMI/saves/saved_models/ParseCaps_dynamic_BrainTumor_99.375.pth"

            
            # SEED        
            torch.manual_seed(42)
            np.random.seed(42) 
            
            check_device(Model)

            # total_params = count_parameters(Model)
            # test_data, labels = next(iter(test_loader))
            # onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
            # data, onehot_labels, labels= test_data.to(device), onehot_labels.to(device), labels.to(device)
            
            # flops, param = profile(Model.to(device), inputs=(data, onehot_labels))
            
            # print(f'Total FLOPs: {flops}, Trainable Parameters: {param}')
            # print(f'Total Parameters: {total_params}')
            
            
            # ic(path)
            if load_checkpoint:
                try:
                    state_dict = torch.load(path)
                    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    Model.load_state_dict(state_dict)
                    print(f"{path} Checkpoint loaded!")
                except Exception as e:
                    print(e)
                    sys.exit()
                    
            batch_accuracy = 0
            batch_idx = 0
            
            Model.eval() 
            for batch_idx, (test_data, labels) in tqdm(enumerate(test_loader)): #from testing dataset

                labels = labels.type(torch.int64)
                onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
                data, onehot_labels,labels = test_data.to(device), onehot_labels.to(device), labels.to(device)
                indices, x_recon, y_pred, concept = Model(data, onehot_labels)
                batch_accuracy += accuracy_calc(predictions=indices, labels=labels)
            
            test_epoch_accuracy = batch_accuracy/(batch_idx+1)
            print(f" Testing Accuracy : {test_epoch_accuracy}")
            
if __name__ == '__main__':
    
    

    test(img_size=img_size, device=get_device(), learning_rate=cfg.LEARNING_RATE, num_epochs=cfg.NUM_EPOCHS, decay_step=cfg.DECAY_STEP, gamma=cfg.DECAY_GAMMA,
          num_classes=num_class, checkpoint_folder=cfg.CHECKPOINT_FOLDER,checkpoint_name=cfg.CHECKPOINT_NAME, load_checkpoint=cfg.LOAD_TEST, graphs_folder=cfg.GRAPHS_FOLDER,
          finetune=cfg.FINETUNE, finetune_layers=cfg.FINETUNE_LAYER)
   