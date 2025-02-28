import torch
import os
from torch.utils.data import DataLoader,Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ColorJitter
import numpy as np
from icecream import ic
import math
import json
import pandas as pd
from PIL import Image
import cv2
from torchvision.io import read_image
from pathlib import Path
import medmnist
from medmnist import INFO, Evaluator

from Utils.transform import *
from Utils.randaug import RandAugment,CutoutDefault

## DDP packages
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.cuda.amp import GradScaler
# from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

"""输入为：样本的size和生成的随机lamda值"""
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)
 
    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    #限制坐标区域不超过样本大小
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2


class Cifar10:

    def __init__(self, data_path, batch_size, shuffle, num_workers=16, N=1, M=1, rotation_degrees=15, translate=(0.1,0.1), scale=(0.95, 1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.N = N
        self.M = M
        # self.rotation = rotation_degrees
        # self.translate = translate
        # self.scale = scale
        self.img_size = 32
        self.num_class = 10
        self.mean = [0.4914, 0.4822, 0.4465]  # Mean for RGB channels
        self.std = [0.2023, 0.1994, 0.2010]   # Std for RGB channels
        self.in_channel = 3
        self.image_shape = (-1, self.in_channel, self.img_size, self.img_size)

    def __call__(self):

        # Check if dataset exists, if not download
        download_dataset = not os.path.exists(os.path.join(self.data_path, 'cifar-10-batches-py'))

        train_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=self.rotation,scale=self.scale), #translate=self.translate),
            transforms.RandomCrop(32, padding=4),#, padding_mode='reflect'), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            Cutout(n_holes=1, length=16)
        ])
        train_transforms.transforms.insert(0, RandAugment(self.N, self.M))
        train_transforms.transforms.append(CutoutDefault(16))
        
        
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        train_loader = DataLoader(datasets.CIFAR10(root=self.data_path,
                                                   train=True,
                                                   download=download_dataset,
                                                   transform=train_transforms),
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=self.num_workers)

        test_loader = DataLoader(datasets.CIFAR10(root=self.data_path,
                                                  train=False,
                                                  download=download_dataset,
                                                  transform=test_transforms),
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle,
                                 num_workers=self.num_workers)

        print('Successfully load Cifar10 dataset')

        return train_loader, test_loader, self.img_size, self.num_class, self.in_channel, self.image_shape
    
class MNIST:

    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.2), scale=(0.95,1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 32
        self.num_class = 10
        self.in_channel = 1
        self.mean = [0.5]
        self.std = [0.5]
        self.image_shape = (-1, 1, self.img_size, self.img_size)

    def __call__(self):

        train_loader = DataLoader(datasets.MNIST(root=self.data_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale),
                                                                                      transforms.Pad(4,padding_mode='reflect'),
                                                                                      transforms.Resize(self.img_size),
                                                                                      transforms.ToTensor(),
                                                                                      transforms.Normalize(self.mean, self.std)])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.MNIST(root=self.data_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                            transforms.Pad(4,padding_mode='reflect'), 
                                                            transforms.Resize(self.img_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(self.mean, self.std)])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        print('Successfully load MNIST dataset')
        return train_loader, test_loader, self.img_size, self.num_class, self.in_channel,self.image_shape
    
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, masks, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long) - 1  # Convert labels from 1-indexed to 0-indexed
        self.masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # ic(image.shape)
        image = elastic_transform(image, alpha=15, sigma=3)
        if self.transform:
            image = self.transform(image)
        
        #TODO: Add mask to return if needed
        #return image, self.labels[idx], self.masks[idx]
        return image, self.labels[idx]

class BrainTumor:
    def __init__(self, data_path, batch_size, shuffle, num_workers=4, test_split=0.2):
        self.data_path = data_path + 'BrainTumor/'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.test_split = test_split

        # Load data
        self.labels = np.load(self.data_path + 'labels.npy')
        self.images = np.load(self.data_path + 'images_cropped.npy')
        self.images = self.normalize_images(self.images)
        self.masks = np.load(self.data_path + 'masks.npy')
        
        # print(self.images.shape)
        # print(f"Labels range:  {np.min(self.labels)}  to  {np.max(self.labels)}")
        # print(f"Images range:  {np.min(self.images)}  to  {np.max(self.images)}")
        # print(f"Masks range:  {np.min(self.masks)}  to  {np.max(self.masks)}")

        self.img_size = self.images.shape[1]
        self.num_class = len(np.unique(self.labels))
        self.in_channel = 1  # Grayscale images
        self.image_shape = (-1, self.in_channel, self.img_size, self.img_size)

        # Normalize data
        self.mean = [0.5]
        self.std = [0.5]
    
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomAffine(degrees=self.rotation,scale=self.scale), #translate=self.translate),
            # transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomCrop(self.img_size, padding=16, padding_mode='reflect'), 
            # transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            # Cutout(n_holes=1, length=16)
            # RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=self.mean)
        ])

        self.test_transform = transforms.Compose([
            # transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def standardize_images(self, images):
        images = images.astype(np.float32)
        mean = images.mean()
        std = images.std()
        images = (images - mean) / std
        return images
    
    def normalize_images(self, images):
        images = images.astype(np.float32)
        images_min = images.min()
        images_max = images.max()
        images = (images - images_min) / (images_max - images_min)
        return images

    def __call__(self):
        
        train_dataset = BrainTumorDataset(self.images, self.labels, self.masks, transform=self.train_transform)
        test_dataset = BrainTumorDataset(self.images, self.labels, self.masks, transform=self.test_transform)

        test_size = int(self.test_split * len(train_dataset))
        train_size = len(train_dataset) - test_size
        train_dataset, _ = random_split(train_dataset, [train_size, test_size])
        _, test_dataset = random_split(test_dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        print('Successfully loaded Brain Tumor dataset')
        return train_loader, test_loader, self.img_size, self.num_class, self.in_channel, self.image_shape
    

class RobustImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (IOError, OSError) as e:
            path, _ = self.samples[index]  # 获取出错图片的路径和标签
            print(f"Error loading image at index {index}, path: {path}, skipping. Error: {e}")
            return None

class ImageNet:
    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0, 0.2), scale=(0.95, 1.2)):
        
        self.data_path = os.path.join(data_path, 'ImageNet/')
        self.train_path = os.path.join(self.data_path, 'train/')
        self.test_path = os.path.join(self.data_path, 'validate/')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 224
        self.num_class = len(os.listdir(os.path.join(self.data_path, 'train/')))
        self.in_channel = 3
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.image_shape = (-1, self.in_channel, self.img_size, self.img_size)

    def __call__(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=self.rotation, translate=self.translate, scale=self.scale),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_dataset = RobustImageFolder(self.train_path, transform=train_transform)
        test_dataset = RobustImageFolder(self.test_path, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collate_fn)

        print('Successfully loaded ImageNet dataset')
        # ic(self.num_class)
        return train_loader, test_loader, self.img_size, self.num_class, self.in_channel, self.image_shape

    def collate_fn(self, batch):
        # Filter out None samples
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)
 
 
 
 
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class mini_imagenet:
    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rank=0):
        self.data_path = os.path.join(data_path, 'imagenet_mini/')
        self.data_root = os.path.join(self.data_path, 'images/')
        self.json_path = os.path.join(self.data_path,"classes_name.json")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        # self.rotation = rotation_degrees
        # self.translate = translate
        # self.scale = scale
        self.img_size = 224
        self.in_channel = 3
        self.image_shape = (-1, self.in_channel, self.img_size, self.img_size)
        self.rank = rank
        
    def __call__(self):
        
        data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.1,0.1)),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])}

        
        # 实例化训练数据集
        train_dataset = MyDataSet(root_dir=self.data_path,
                                csv_name="new_train.csv",
                                json_path=self.json_path,
                                transform=data_transform["train"])
        # print("self.json_path", self.json_path)
        # print("len of train_dataset", train_dataset.__len__())
        # input()

        num_class = len(train_dataset.labels)

        # 实例化验证数据集
        val_dataset = MyDataSet(root_dir=self.data_path,
                                csv_name="new_val.csv",
                                json_path=self.json_path,
                                transform=data_transform["val"])
        
        # print("num_class: ", num_class)
        # print("len of val_dataset: ", val_dataset.__len__())
        # input()
        

        # # 将样本索引每batch_size个元素组成一个list
        # train_batch_sampler = torch.utils.data.BatchSampler(
        #     train_sampler, self.batch_size, drop_last=True)

        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                # batch_sampler=train_batch_sampler,
                                                batch_size=self.batch_size,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn,
                                                shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=self.batch_size,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=val_dataset.collate_fn,
                                                shuffle=False)
        return train_loader, val_loader, self.img_size, num_class, self.in_channel, self.image_shape


def get_ddp_generator(seed=3407):
    '''
    对每个进程使用不同的随机种子，增强训练的随机性
    '''
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g
   
class mini_imagenet_ddp:
    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rank=0):
        self.data_path = os.path.join(data_path, 'imagenet_mini/')
        self.data_root = os.path.join(self.data_path, 'images/')
        self.json_path = os.path.join(self.data_path,"classes_name.json")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        # self.rotation = rotation_degrees
        # self.translate = translate
        # self.scale = scale
        self.img_size = 224
        self.in_channel = 3
        self.image_shape = (-1, self.in_channel, self.img_size, self.img_size)
        self.rank = rank
        
    def __call__(self):
        
        data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.1,0.1)),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        
        # 实例化训练数据集
        train_dataset = MyDataSet(root_dir=self.data_path,
                                csv_name="new_train.csv",
                                json_path=self.json_path,
                                transform=data_transform["train"])
        # print("self.json_path", self.json_path)
        # print("len of train_dataset", train_dataset.__len__())
        # input()

        num_class = len(train_dataset.labels)

        # 实例化验证数据集
        val_dataset = MyDataSet(root_dir=self.data_path,
                                csv_name="new_val.csv",
                                json_path=self.json_path,
                                transform=data_transform["val"])
        
        # print("num_class: ", num_class)
        # print("len of val_dataset: ", val_dataset.__len__())
        # input()

        # 给每个rank对应的进程分配训练的样本索引
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

        g = get_ddp_generator()

        # # 将样本索引每batch_size个元素组成一个list
        # train_batch_sampler = torch.utils.data.BatchSampler(
        #     train_sampler, self.batch_size, drop_last=True)

        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        if dist.get_rank() == 0:
            print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                # batch_sampler=train_batch_sampler,
                                                batch_size=self.batch_size,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn,
                                                shuffle=False,
                                                sampler=train_sampler,
                                                generator=g)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=self.batch_size,
                                                sampler=val_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=val_dataset.collate_fn,
                                                shuffle=False)
        return train_loader, val_loader, self.img_size, num_class, self.in_channel, self.image_shape
  
  
### skin ###
  
NUM_WORKERS = os.cpu_count()

class CCBMDataset(Dataset):
    def __init__(self, annotations_file, dir_path, masks_dir, extension, transform=None, target_transform=None):
        """Initializes a custom dataset given a CSV file.

        :param annotations_file: The CSV file containing images and labels.
        :param dir_path: The path directory of the images.
        :param masks_dir: The path directory of the segmentation masks.
        :param extension: The file extension of the images (jpg, png).
        :param transform: A PyTorch Transform to be applied to the images.
        :param target_transform: A PyTorch Transform to be applied to the labels
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.dir_path = dir_path
        self.masks_dir = masks_dir
        self.extension = extension
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.dir_path, self.img_labels.iloc[idx, 0])

        # Read image
        if self.extension == "":
            #image = read_image(img_path)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.img_labels.iloc[idx, 0].find("/") != -1:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0][4:-4] + ".png"), cv2.IMREAD_UNCHANGED
                )
            else:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0][:-4] + ".png"), cv2.IMREAD_UNCHANGED
                )
        else:
            #image = read_image(img_path + "." + self.extension)
            image = cv2.imread(img_path + "." + self.extension)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.img_labels.iloc[idx, 0].find("/") != -1:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0][4:] + ".png"), cv2.IMREAD_UNCHANGED
                )
            else:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0] + ".png"), cv2.IMREAD_UNCHANGED
                )


        # Get image mask
        #img_mask = read_image(os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0] + ".png"))

        # Get respective label
        label = self.img_labels.iloc[idx, 1]

        # Get Indicator Vectors
        indicator_vectors_dict = np.load(params.INDICATOR_VECTORS, allow_pickle=True).item()
        indicator_vectors_dict = {k.replace('.jpg', '').replace('.JPG', ''): v for k, v in indicator_vectors_dict.items()}

        filename = Path(img_path).stem
        indicator_vector = indicator_vectors_dict[filename]

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.target_transform:
            label = self.target_transform(label)

        return {
                'image': image,
                'label': label,
                'ind_vec': indicator_vector,
                'mask': mask,
                'img_path': img_path
        }



def create_dataloaders(path, batch_size,
        train_transform: transforms.Compose,
        val_transform: transforms.Compose
):
    """Creates training and validation DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      params: a file containing model parameters.
      train_transform: torchvision transforms to perform on training data.
      val_transform: torchvision transforms to perform on validation data.

    Returns:
      A tuple of (train_dataloader, validation_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, validation_dataloader, class_names = \
          = create_dataloaders(params=ph2_params,
                               train_transform=some_transform,
                               val_transform=some_transform,
                               )
    """
    # Use ImageFolder to create dataset(s)
    train_data = CCBMDataset(annotations_file= path+"skin/data/PH2_train.csv",
                             dir_path= path+"skin/PH2Dataset",
                             masks_dir= None,
                             extension= "png",
                             transform=train_transform)

    validation_data = CCBMDataset(annotations_file= path+"skin/data/PH2_validation.csv",
                                  dir_path= path+"skin/PH2Dataset",
                                  masks_dir= None,
                                  extension= "png",
                                  transform=val_transform)

    # Get class names
    class_names = ["Nevus", "Melanoma"]

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size= batch_size,
        shuffle=True,
        num_workers= 8,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size= batch_size,
        shuffle=False,
        num_workers= 8,
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader, class_names


def create_dataloader_for_evaluation(path,
                                     transform: transforms.Compose
                                     ):
    """Creates testing DataLoaders.

    Takes in a testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      params: a file containing model parameters.
      transform: torchvision transforms to perform on test data.


    Returns:
      A tuple of (test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        test_dataloader, class_names = \
          = create_dataloaders(params=ph2_params,
                               transform=some_transform
                               )
    """
    test_data = CCBMDataset(annotations_file=params.TEST_FILENAME,
                            dir_path=params.IMAGES_DIR,
                            masks_dir=params.MASKS_DIR,
                            extension=params.FILE_EXTENSION,
                            transform=transform)

    # Get class names
    class_names = ["Nevus", "Melanoma"]

    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0, #params.NUM_WORKERS,
        pin_memory=True,
    )

    return test_dataloader, class_names



class medmnist_dataset:
    '''
        medmnist dataset, use data_flag to select specific dataset,
        see https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py for candidate dataset.
    '''
    def __init__(self, data_flag, batch_size, downdload=False):
        self.data_flag = data_flag
        self.batch_size = batch_size
        # self.shuffle = shuffle
        # self.num_workers = num_workers
        self.download = downdload

        if self.data_flag == "tissuemnist":
            self.rotation = 30
            self.translate = (0,0.2)
            self.scale = (0.95,1.2)
            self.img_size = 28
            self.mean = [0.5]
            self.std = [0.5]
            self.train_transform = transforms.Compose([transforms.RandomAffine(degrees=self.rotation, 
                                                                translate=self.translate, 
                                                                scale=self.scale),
                                        transforms.Pad(4,padding_mode='reflect'),
                                        transforms.Resize(self.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])
            self.test_transform = transform=transforms.Compose([transforms.Pad(4,padding_mode='reflect'), 
                                                    transforms.Resize(self.img_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(self.mean, self.std)])
        
    def __call__(self):
        
        info = INFO[self.data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        DataClass = getattr(medmnist, info['python_class'])

        # load the data
        train_dataset = DataClass(split='train', transform=self.train_transform, download=self.download)
        test_dataset = DataClass(split='test', transform=self.test_transform, download=self.download)
    
        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        
        # encapsulate data into dataloader form
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=nw)
        # train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
        # here use test as validation
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=nw)
        
        self.image_shape = (-1, n_channels, self.img_size, self.img_size)
        
        return train_loader, test_loader, self.img_size, n_classes, n_channels, self.image_shape