import torch
import torch.nn as nn
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
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


from sklearn.metrics import precision_recall_fscore_support, confusion_matrix



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


def compute_per_class_values(cm, num_classes):
    # 计算每类的TN, FP, FN, TP
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    return TP, FP, FN, TN

def compute_metrics(TP, FP, FN, TN):
    # 计算查准率、召回率、F1得分和真负率
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, specificity, f1_score


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
            elif cfg.MODEL == 'DeepCaps':
                Model = nn.DataParallel(DeepCapsModel(num_class=num_classes, img_height=image_shape[-1], img_width=image_shape[-1], in_channel=in_channel, device=device)).to(device)
                print("Successfully loaded DeepCaps model")
        # print(Model)
            else:
                assert False, "Model not found!"
            
        
            checkpoint_path =  checkpoint_folder + checkpoint_name 
            path = "saves/saved_models/ParseCaps_dynamic_BrainTumor_99.375.pth"
            
            # SEED        
            torch.manual_seed(42)
            np.random.seed(42) 
            
            check_device(Model)
            
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
            
            # Initialize metrics
            total_precision = np.zeros(num_classes)
            total_recall = np.zeros(num_classes)
            total_f1 = np.zeros(num_classes)
            total_specificity = np.zeros(num_classes)
            num_batches = 0

            for batch_idx, (test_data, labels) in tqdm(enumerate(test_loader)):  # from testing dataset
                
                labels = labels.type(torch.int64)
                onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
                data, onehot_labels, labels = test_data.to(device), onehot_labels.to(device), labels.to(device)
                
                indices, _, _, _,  = Model(data, onehot_labels)
                predictions = indices
                
                cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=np.arange(num_classes))
                TP, FP, FN, TN = compute_per_class_values(cm, num_classes)
                precision, recall, specificity, f1 = compute_metrics(TP, FP, FN, TN)
                accuracy = (predictions == labels).float().mean()

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_specificity += specificity
                num_batches += 1

            # Calculate averages
            avg_precision = total_precision / num_batches
            avg_recall = total_recall / num_batches
            avg_f1 = total_f1 / num_batches
            avg_specificity = total_specificity / num_batches

            for i in range(num_classes):
                print(f'Class {i}: Precision: {avg_precision[i]:.4f}, Recall: {avg_recall[i]:.4f}, Specificity: {avg_specificity[i]:.4f}, F1 Score: {avg_f1[i]:.4f}')
            
            
            # AUC, F1, Precision, Recall
            # total_accuracy = 0
            # total_precision = 0
            # total_recall = 0
            # total_f1 = 0
            # total_specificity = 0
            # num_batches = 0

            # for batch_idx, (test_data, labels) in tqdm(enumerate(test_loader)):  # from testing dataset
            #     labels = labels.type(torch.int64)
            #     onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
            #     data, onehot_labels,labels = test_data.to(device), onehot_labels.to(device), labels.to(device)
            #     indices, _,_,_, = Model(data, onehot_labels)
            #     predictions = indices 
                
            #     # ic(indices.shape, labels.shape)
                
            #     # 计算混淆矩阵
            #     cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=np.arange(num_classes))
            #     TP, FP, FN, TN = compute_per_class_values(cm, num_classes)
            #     precision, recall, specificity, f1 = compute_metrics(TP, FP, FN, TN)
            #     accuracy = (predictions == labels).float().mean()

            #     total_accuracy += accuracy.item()
            #     total_precision += precision
            #     total_recall += recall
            #     total_f1 += f1
            #     total_specificity += specificity
            #     num_batches += 1

            # # 计算总体平均值
            # avg_accuracy = total_accuracy / num_batches
            # avg_precision = np.mean(total_precision / num_batches)
            # avg_recall = np.mean(total_recall / num_batches)
            # avg_f1 = np.mean(total_f1 / num_batches)
            # avg_specificity = np.mean(total_specificity / num_batches)

            # print(f'Average Accuracy: {avg_accuracy:.4f}')
            # print(f'Average Precision: {avg_precision:.4f}')
            # print(f'Average Recall: {avg_recall:.4f}')
            # print(f'Average F1 Score: {avg_f1:.4f}')
            # print(f'Average Specificity: {avg_specificity:.4f}')


if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    
    test(img_size=img_size, device=get_device(), learning_rate=cfg.LEARNING_RATE, num_epochs=cfg.NUM_EPOCHS, decay_step=cfg.DECAY_STEP, gamma=cfg.DECAY_GAMMA,
          num_classes=num_class, checkpoint_folder=cfg.CHECKPOINT_FOLDER,checkpoint_name=cfg.CHECKPOINT_NAME, load_checkpoint=True, graphs_folder=cfg.GRAPHS_FOLDER,
          finetune=cfg.FINETUNE, finetune_layers=cfg.FINETUNE_LAYER)



    # for batch_idx, (test_data, labels) in tqdm(enumerate(test_loader)):  # from testing dataset
    #     labels = labels.type(torch.int64)
    #     onehot_labels = torch.zeros(labels.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.)
    #     data, onehot_labels,labels = test_data.to(device), onehot_labels.to(device), labels.to(device)
    #     indices, _,_,_, = Model(data, onehot_labels)
    #     predictions = indices 
        
    #     # ic(indices.shape, labels.shape)
        
    #     # 计算混淆矩阵
    #     cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=np.arange(num_classes))
    #     TP, FP, FN, TN = compute_per_class_values(cm, num_classes)
    #     precision, recall, specificity, f1 = compute_metrics(TP, FP, FN, TN)
    #     accuracy = (predictions == labels).float().mean()

    #     total_accuracy += accuracy.item()
    #     total_precision += precision
    #     total_recall += recall
    #     total_f1 += f1
    #     total_specificity += specificity
    #     num_batches += 1

    #     # 计算总体平均值
    #     avg_accuracy = total_accuracy / num_batches
    #     avg_precision = np.mean(total_precision / num_batches)
    #     avg_recall = np.mean(total_recall / num_batches)
    #     avg_f1 = np.mean(total_f1 / num_batches)
    #     avg_specificity = np.mean(total_specificity / num_batches)

    #     print(f'Average Accuracy: {avg_accuracy:.4f}')
    #     print(f'Average Precision: {avg_precision:.4f}')
    #     print(f'Average Recall: {avg_recall:.4f}')
    #     print(f'Average F1 Score: {avg_f1:.4f}')
    #     print(f'Average Specificity: {avg_specificity:.4f}')