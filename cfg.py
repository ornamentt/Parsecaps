
MODEL = 'ParseCaps'                  # Choice:CapsNet/OrthCaps_shallow/ParseCaps/DeepCaps
ROUTING = 'dynamic'                     # Choice: attention/dynamic
DATASET = 'BrainTumor'                      # Choice: Cifar10/BrainTumor/MNIST/Skin/mini_imagenet/tissuemnist
ACTIVATION = 'Squash'                     # Choice: Capsule_ReLU/Squash

### model parameters ###
SIMILARITY = 0.7  
ATTENTION_THRESHOLD = 1000
KWARGS_DICT = {
        'patch_size': [3, 3, 3],
        'patch_stride': [2, 2, 2],
        'patch_padding': [1, 1, 1],
        'embed_dim': [16, 36, 64], # [64, 192, 384],
        'depth': [1, 2, 5], # [1, 2, 10],
        'num_heads': [1, 1, 1], #[1, 3, 6],
        'mlp_ratio': [4.0, 4.0, 4.0],
        'qkv_bias': [True, True, True],
        'drop_rate': [0.0, 0.0, 0.0],
        'attn_drop_rate': [0.0, 0.0, 0.0],
        'drop_path_rate': [0.0, 0.0, 0.1],
        'kernel_size': [3, 3, 3],
        'padding_q': [1, 1, 1],
        'padding_kv': [1, 1, 1],
        'stride_kv': [2, 2, 2],
        'stride_q': [1, 1, 1],
    }  
# DIM = [None, 16, 64, 192, 384]
DIM = [None, 8, 16, 36, 64]
LAYERNUM = 3

### train ###
LEARNING_RATE = 2.5e-3 # 5e-6 # 5e-4 # 5e-3
# LEARNING_RATE = 5e-3
NUM_EPOCHS = 300
BATCH_SIZE = 128 # 128 # 256 # 20 # 16 # 128
# BATCH_SIZE = 64
WEIGHT_DECAY = 5e-4
DECAY_STEP = 25
DECAY_GAMMA = 0.95
PATIENCE = 10
FACTOR = 0.5
GRAPH = NUM_EPOCHS-1

### Loss ###
IFR = False                            # Whether to reconstruct or not, Choice: True/False
RECONSTRUCTION = 0.0005

### augmentation ###
BETA=0.25
CUTMIX_PROB=0.5

### mode
MODE = 'train'                          # Choice: train/test
LOAD_TRAIN = False
LOAD_TEST = True
FINETUNE = False
FINETUNE_LAYER = ['convlayer','Primary_Cap','cell1']                  

### data path ###
CHECKPOINT_FOLDER = 'saves/saved_models/'
CHECKPOINT_NAME = f'{DATASET}/{MODEL}_'
DATASET_FOLDER = '/data/dataset_folder/'
GRAPHS_FOLDER = 'saves/graphs/'
PLOT_NAME = MODEL + '_' + DATASET + '_accuracy_epoch'
