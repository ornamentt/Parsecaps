import json
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid
from icecream import ic
import os
from torch.utils.data import DataLoader
import random
from PIL import Image
import torchvision.transforms as transforms
torch.backends.cudnn.enabled = False

RESULTS_DIR = 'results'
CONFIG_DIR = 'configs'
RESULTS_FILENAME = 'accuracies_losses_valid.csv'

plt.style.use('seaborn-paper')


def get_comparison_plot(images, model):
    """Creates a plot that shows similar prototypes with their relevance scores and concept values.

    Parameters
    ----------
    images: torch.Tensor
       An array with the images to be compared
    model: models.senn
       A senn model to be used for the visualizations

    Returns
    ----------
    fig: matplotlib.pyplot
        The figure that contains the plots
    """

    def get_colors(values):
        colors = ['b' if v > 0 else 'r' for v in values]
        colors.reverse()
        return colors

    model.eval()
    with torch.no_grad():
        y_pred, (concepts, relevances), _ = model(images)
    y_pred = y_pred.argmax(1)

    fig, axes = plt.subplots(nrows=3, ncols=len(images))

    PROTOTYPE_ROW = 0
    RELEVANCE_ROW = 1
    CONCEPT_ROW = 2

    concepts_min = concepts.min().item()
    concepts_max = concepts.max().item()
    concept_lim = -concepts_min if -concepts_min > concepts_max else concepts_max

    for i in range(len(images)):
        prediction_index = y_pred[i].item()
        concept_names = [f'C{i + 1}' for i in range(concepts.shape[1] - 1, -1, -1)]

        # plot the input image
        axes[PROTOTYPE_ROW, i].imshow(images[i].permute(1, 2, 0).squeeze(), cmap='gray')
        axes[PROTOTYPE_ROW, i].set_title(f"Prediction: {prediction_index}")
        axes[PROTOTYPE_ROW, i].axis('off')

        # plot the relevance scores
        rs = relevances[i, :, prediction_index]
        colors_r = get_colors(rs)
        axes[RELEVANCE_ROW, i].barh(np.arange(len(rs)),
                                    np.flip(rs.detach().numpy()),
                                    align='center', color=colors_r)

        axes[RELEVANCE_ROW, i].set_yticks(np.arange(len(concept_names)))
        axes[RELEVANCE_ROW, i].set_yticklabels(concept_names)
        axes[RELEVANCE_ROW, i].set_xlim(-1.1, 1.1)

        # plot the concept values
        cs = concepts[i].flatten()
        colors_c = get_colors(cs)
        axes[CONCEPT_ROW, i].barh(np.arange(len(cs)),
                                  np.flip(cs.detach().numpy()),
                                  align='center', color=colors_c)

        axes[CONCEPT_ROW, i].set_yticks(np.arange(len(concept_names)))
        axes[CONCEPT_ROW, i].set_yticklabels(concept_names)
        axes[CONCEPT_ROW, i].set_xlim(-concept_lim - 0.2, concept_lim + 0.2)

        # Only show titles for the leftmost plots
        if i == 0:
            axes[CONCEPT_ROW, i].set_ylabel("Concepts scores")
            axes[RELEVANCE_ROW, i].set_ylabel("Relevance scores")

    return fig


def create_barplot(ax, relevances, y_pred, x_lim=1.1, title='', x_label='', concept_names=None, **kwargs):
    """Creates a bar plot of relevances.

    Parameters
    ----------
    ax : pyplot axes object
        The axes on which the bar plot should be created.
    relevances: torch.tensor
        The relevances for which the bar plot should be generated. shape: (1, NUM_CONCEPTS, NUM_CLASSES)
    y_pred: torch.tensor (int)
        The prediction of the model for the corresponding relevances. shape: scalar value
    x_lim: float
        the limits of the plot
    title: str
        the title of the plot
    x_label: str
        the label of the X-axis of the plot
    concept_names: list[str]
        the names of each feature on the plot
    """
    # Example data
    y_pred = y_pred.item()
    if len(relevances.squeeze().size()) == 2:
        relevances = relevances[:, y_pred]
    relevances = relevances.squeeze()
    if concept_names is None:
        concept_names = ['C. {}'.format(i + 1) for i in range(len(relevances))]
    else:
        concept_names = concept_names.copy()
    concept_names.reverse()
    y_pos = np.arange(len(concept_names))
    colors = ['b' if r > 0 else 'r' for r in relevances]
    colors.reverse()

    ax.barh(y_pos, np.flip(relevances.detach().cpu().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_title(title, fontsize=18)


def plot_lambda_accuracy(config_list, save_path=None, num_seeds=1, valid=False, **kwargs):
    """Plots the lambda (robustness regularizer) vs accuracy of SENN

    Parameters
    ----------
    config_list: list
        List of experiment configs files used to vary the lambda.
        If multiple seeds are used then this is a list of lists where the inner lists have a length
        equal to the number of different seeds used and contain the corresponding configs files.
    save_path: str
        Path to the location where the plot should be saved.
    num_seeds : int
        The number of different seeds that are used.
    valid : bool
        If true create plots based on saved validation accuracy (fast approach,
        only recommended if validation set was not used to tune hyper parameters).
        If false (default) the best model is loaded and evaluated on the test set (more runtime extensive).

    """
    assert type(num_seeds) is int and num_seeds > 0, "num_seeds must be an integer > 0 but is {}".format(num_seeds)
    lambdas = []
    accuracies = []
    std_seeds = []

    path = Path(CONFIG_DIR)
    for config_file in config_list:
        seed_accuracies = []
        for seed in range(num_seeds):
            config_path = path / config_file if num_seeds == 1 else path / config_file[seed]
            # if test mode: instanciate trainer that evaluates model on the test set
            if not valid:
                t = trainer.init_trainer(config_path, best_model=True)
                seed_accuracies.append(t.test())
            with open(config_path, 'r') as f:
                config = json.load(f)
                # if validation mode: read top validation accuracy from csv file (a lot faster)
                if valid:
                    result_dir = Path(RESULTS_DIR)
                    results_csv = result_dir / config["exp_name"] / RESULTS_FILENAME
                    seed_accuracies.append(pd.read_csv(results_csv, header=0)['Accuracy'].max())
        lambdas.append(config["robust_reg"])
        std_seeds.append(np.std(seed_accuracies))
        accuracies.append(sum(seed_accuracies) / num_seeds)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.errorbar(np.arange(len(lambdas)), accuracies, std_seeds, color='r', marker='o')
    ax.set_xticks(np.arange(len(lambdas)))
    ax.tick_params(labelsize=12)
    ax.set_xticklabels(lambdas, fontsize=12)
    ax.set_xlabel('Robustness Regularization Strength', fontsize=18)
    ax.set_ylabel('Prediction Accuracy', fontsize=18)
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    return fig



def show_explainations(model, test_loader, num_classes, device, save_path, num_explanations=2, batch_size=128, concept_names=None,
                       **kwargs):
    """Generates some explanations of model predictions.

    Parameters
    ----------
    model : torch nn.Module
        model to visualize
    test_loader: Dataloader object
        Test set dataloader to iterate over test set.
    dataset : str
        Name of the dataset used.
    num_explanations : int
        生成解释的数量。
    save_path : str
        Directory where the figures are saved. If None a figure is showed instead.
    batch_size : int
        batch_size of test loader
    concept_names : list of str 或 None
        概念的名称（如果有）。
        
    """
    model.eval()

    # # 选择测试样本
    (test_batch, test_labels) = next(iter(test_loader))
    test_batch = test_batch.float().to(device)
    labels = test_labels.type(torch.int64).to(device)
    onehot_labels = torch.zeros(labels.size(0), num_classes, device=device)
    onehot_labels.scatter_(1, labels.view(-1, 1), 1.)
    

    # 将测试批次送入模型获取解释
    _, _, y_pred, concepts = model(test_batch, onehot_labels)
    # y_pred, (concepts, relevances), _ = model(test_batch, onehot_labels)

    
    if len(y_pred.size()) > 1:    # 如果预测维度超过一维，取 argmax 获取预测的类别索引
        y_pred = y_pred.argmax(1)
    
    # 确定可视化概念的范围
    concepts_min = concepts.min().item()     # 概念激活中的最小值。
    concepts_max = concepts.max().item()     # 概念激活中的最大值。
    concept_lim = abs(concepts_min) if abs(concepts_min) > abs(concepts_max) else abs(concepts_max)    # 绘图的最大限度

    plt.style.use('seaborn-paper')
    batch_idx = np.random.randint(0, batch_size - 1, num_explanations)      # 随机选择可视化的索引。
    for i in range(num_explanations):                                       # 循环生成解释的数量。
        if concept_names is not None:
            gridsize = (1, 2)
            fig = plt.figure(figsize=(12, 6))
            ax1 = plt.subplot2grid(gridsize, (0, 0))
            ax2 = plt.subplot2grid(gridsize, (0, 1))
            # 为相关性创建条形图。
            create_barplot(ax1, relevances[batch_idx[i]], y_pred[batch_idx[i]], x_label='Relevances (theta)',
                           concept_names=concept_names, **kwargs)
            ax1.xaxis.set_label_position('top')
            ax1.tick_params(which='major', labelsize=12)
            # 为概念激活创建条形图。
            create_barplot(ax2, concepts[batch_idx[i]], y_pred[batch_idx[i]], x_lim=concept_lim,
                           x_label='Concepts/Raw Inputs', concept_names=concept_names, **kwargs)
            ax2.xaxis.set_label_position('top')
            ax2.tick_params(which='major', labelsize=12)

        else:       # 如果没有提供概念名称， 
            gridsize = (1, 2)  # 更新网格尺寸为1行2列
            fig = plt.figure(figsize=(6, 3))  # 调整画布大小
            ax1 = plt.subplot2grid(gridsize, (0, 0))
            ax2 = plt.subplot2grid(gridsize, (0, 1))

            # 显示输入图像的代码块保持不变
            image = test_batch[batch_idx[i]].cpu()
            if image.dim() == 3 and image.shape[0] == 3:  # 假定第一个维度是通道且为RGB图像
                image = image.permute(1, 2, 0)  # 重排维度从 (C, H, W) 到 (H, W, C)
            ax1.imshow(image)
            ax1.set_axis_off()
            ax1.set_title(f'Input Prediction: {y_pred[batch_idx[i]].item()}', fontsize=18)

            # 更新为仅创建概念激活条形图的代码块
            create_barplot(ax2, concepts[batch_idx[i]], y_pred[batch_idx[i]], x_lim=concept_lim,
                        x_label='Concept activations', **kwargs)
            ax2.xaxis.set_label_position('top')
            ax2.tick_params(which='major', labelsize=12)

            plt.tight_layout()
            plt.savefig(path.join(save_path, f'explanation_{i}.png'),bbox_inches='tight', dpi=300, transparent=True)
            plt.show()
            plt.close('all')  # 关闭图形以释放内存。

# ----------------------------------------------------------------------------------------------------------------------------

def highest_activations(model, test_loader, num_class, device, save_path, num_concepts=10, num_prototypes=9):
    """通过最高激活创建概念表示。

    概念通过最具原型性的数据样本表示。
    （即产生每个概念最高激活的样本）

    参数
    ----------
    model: torch.nn.Module
      训练完毕且具有所有参数的模型。
    test_loader: DataLoader 对象
       遍历测试集的数据加载器。
    num_concepts: int
       模型的概念数量。
    num_prototypes: int
        每个概念应显示的原型示例数量。
    save_path: str
        保存条形图的路径位置。
    """
    model.eval()  # 将模型设置为评估模式。
    activations = []  # 初始化激活列表。
    for test_data, labels in test_loader:  # 遍历测试加载器。
        test_data = test_data.float()
        labels = labels.type(torch.int64).to(device)
        onehot_labels = torch.zeros(labels.size(0), num_class, device=device)
        onehot_labels.scatter_(1, labels.view(-1, 1), 1.)
        data, onehot_labels,labels = test_data.to(device), onehot_labels.to(device), labels.to(device)
        
        with torch.no_grad():  # 不计算梯度。
            _, _, _, concepts = model(data, onehot_labels)  # 获取模型的概念输出。
            activations.append(concepts.squeeze())  # 收集概念激活并去掉多余的维度。
    activations = torch.cat(activations)  # 将激活列表连接成一个张量。

    _, top_test_idx = torch.topk(activations, num_prototypes, 0)  # 获取每个概念激活最高的索引。
    # top_test_idx = top_test_idx.cpu()
    data_tensor = torch.tensor(test_loader.dataset.data).to(device)
    top_examples = [data_tensor[top_test_idx[:, concept]] for concept in range(num_concepts)]  # 收集最高激活的样本。
    # 保证图像具有正确的形状
    
    # ic(top_examples[0].shape)
    top_examples = [img.unsqueeze(0) if len(img.size()) == 2 else img for sublist in top_examples for img in sublist]
    top_examples = [img.permute(2, 0, 1) for img in top_examples]  # 从 (H, W, C) 转换为 (C, H, W)
    # ic(top_examples[0].shape)      #([3, 32, 32])
    # ic(len(top_examples))          # 90
    top_examples_tensor = torch.stack(top_examples)

    plt.rcdefaults()  # 重置matplotlib默认设置。
    fig, ax = plt.subplots()  # 创建图形和坐标轴。
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]  # 创建概念名称列表。

    start = 0.0
    end = num_concepts * data.size(-1)
    stepsize = abs(end - start) / num_concepts  # 计算步长。
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))  # 设置y轴刻度。
    ax.set_yticklabels(concept_names)  # 设置y轴标签。
    plt.xticks([])  # 隐藏x轴刻度。
    ax.set_xlabel('{} most prototypical data examples per concept'.format(num_prototypes))  # 设置x轴标签。
    ax.set_title('Concept Prototypes: ')  # 设置标题。
    
    
    save_or_show(make_grid(top_examples_tensor, nrow=num_prototypes, pad_value=1), save_path)  # 显示或保存图像。
    plt.rcdefaults()  # 重置matplotlib默认设置。


###


def save_or_show(img, save_path):
    """Saves an image or displays it.
    
    Parameters
    ----------
    img: torch.Tensor
        Tensor containing the image data that should be saved.
    save_path: str
        Path to the location where the bar plot should be saved. If None is passed image is showed instead.
    """
    
    img = img.clone().squeeze()
    npimg = img.cpu().numpy()
    if len(npimg.shape) == 2:
        if save_path is None:
            plt.imshow(npimg, cmap='Greys')
            plt.show()
        else:
            plt.imsave(save_path, npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
    plt.clf()
    
# 加载并选择特定图像
def load_select_images(loader, indices, device):
    """根据提供的索引从DataLoader加载和选择图像"""
    selected_images = []
    dataset = loader.dataset
    for idx in indices:
        img, _ = dataset[idx]  # 假设__getitem__返回的是(img, label)
        img = img.unsqueeze(0).to(device)  # 添加批处理维度并转移到设备
        selected_images.append(img)
    return torch.cat(selected_images)  # 返回堆叠的张量


###

# def highest_contrast(model, test_loader, num_class, device, save_path, num_concepts=8, num_prototypes=8):
    
#     model.eval()  # 将模型设置为评估模式。
#     activations = []  # 初始化激活列表。
#     for test_data, labels in test_loader:  # 遍历测试加载器。
#         try:
#             test_data = test_data.float()
#             labels = labels.type(torch.int64).to(device)
#             onehot_labels = torch.zeros(labels.size(0), num_class, device=device)
#             onehot_labels.scatter_(1, labels.view(-1, 1), 1.)
#             data, onehot_labels, labels = test_data.to(device), onehot_labels.to(device), labels.to(device)
#             with torch.no_grad():  # 不计算激度。
#                 _, _, _, concepts = model(data, onehot_labels)  # 获取模型的概念输出。
#                 activations.append(concepts.squeeze())  # 收集概念激活并去掉多余的维度。
#         except FileNotFoundError as e:
#             print(f"Skipping missing file: {e}")
#             continue  # 跳过当前缺失的文件，继续处理下一个文件
    
#     activations = torch.cat(activations)  # 将激活列表连接成一个张量。

#     contrast_scores = torch.empty_like(activations).to(device)  # 初始化对比度得分张量。
#     for c in range(num_concepts - 1):  # 对每个概念计算对比度。
#         # 对比度是当前概念的激活值与其他所有概念的激活值之和的差。
#         contrast_scores[:, c] = activations[:, c] - (activations[:, :c].sum(dim=1) + activations[:, c + 1:].sum(dim=1))
#     # 处理最后一个概念的对比度。
#     contrast_scores[:, num_concepts - 1] = activations[:, num_concepts - 1] - activations[:, :num_concepts - 1].sum(dim=1)

#     _, top_test_idx = torch.topk(contrast_scores, num_prototypes, 0)  # 获取对比度最高的索引。
    
#     # data_tensor = torch.tensor(test_loader.dataset.data).to(device)
#     # top_examples = [data_tensor[top_test_idx[:, concept]] for concept in range(num_concepts)]  # 收集最高对比度的样本。
#     # 使用前面定义的函数加载和选择图像
#     top_examples = []
#     for concept in range(num_concepts):
#         indices = top_test_idx[:, concept].tolist()  # 获取每个概念的索引列表
#         concept_images = load_select_images(test_loader, indices, device)
#         top_examples.append(concept_images)
    
#     # 保证图像具有正确的形状
#     top_examples = [img.unsqueeze(0) if len(img.size()) == 2 else img for sublist in top_examples for img in sublist]
#     top_examples_tensor = torch.stack(top_examples)
    
#     # 获取原始张量的形状
#     num_images = top_examples_tensor.shape[0]

#     # 计算每列应有的图像数，这里假设 num_prototypes 为每列的图像数
#     num_columns = num_images // num_prototypes

#     # 重新排列图像张量以符合列优先顺序
#     # 先 reshape 成二维 (num_columns, num_prototypes)，然后转置
#     indices = torch.arange(num_images).reshape(num_columns, num_prototypes).t().flatten()
#     reordered_tensor = top_examples_tensor[indices]
    
    
#     # 使用 make_grid 生成图像网格，并调整图像大小
#     top_examples_grid = make_grid(reordered_tensor, nrow=num_columns, padding=2, pad_value=1)

#     # 将张量从 (C, H, W) 转换为 (H, W, C) 以供显示或保存
#     top_examples_grid = top_examples_grid.permute(1, 2, 0).cpu().numpy()
    
#     # top_examples_grid = np.transpose(top_examples_grid, (1, 0, 2))
    
#     # # 网格的新形状
#     # new_shape = (num_concepts, num_prototypes, top_examples_grid.shape[0] // num_concepts, top_examples_grid.shape[1] // num_prototypes, top_examples_grid.shape[2])
#     # # 调整形状并转置
#     # top_examples_grid = top_examples_grid.reshape(new_shape).transpose(1, 0, 2, 3, 4)
#     # top_examples_grid = top_examples_grid.reshape(top_examples_grid.shape[1], top_examples_grid.shape[0], top_examples_grid.shape[2])

#     # 确保数据在0到1之间
#     if top_examples_grid.dtype == np.float32 or top_examples_grid.dtype == np.float64:
#         top_examples_grid = np.clip(top_examples_grid, 0, 1)
#     else:
#         top_examples_grid = np.clip(top_examples_grid / 255.0, 0, 1)

#     # 调整图像大小，并在图像之间留出一些空间
#     fig, ax = plt.subplots(figsize=(8, 10))  
#     fig.patch.set_facecolor('white')
#     ax.imshow(top_examples_grid)
    
#     # 隐藏坐标轴
#     ax.axis('off')
#     # 移除所有边框
#     for spine in ax.spines.values():
#         spine.set_visible(False)
    
#     column_width = top_examples_grid.shape[1] / num_prototypes
#     for i in range(num_concepts):
#         ax.text(i * (top_examples_grid.shape[1] / num_concepts) + (top_examples_grid.shape[1] / num_concepts) / 2,
#             -100,  # 增加与图片的垂直间隔
#             '$p_{}$'.format(i + 1), ha='center', va='top', fontsize=20, color='black')

        
#     plt.subplots_adjust(top=2.0)  # 调整顶部边距以避免标签重叠
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight',dpi=300,transparent=True)
#     else:
#         plt.show()

#     plt.rcdefaults()  



def filter_concepts(model, num_concepts=5, num_prototypes=10, save_path=None):
    """通过过滤器可视化创建概念表示。

    概念通过概念编码器最后一层的过滤器表示。
    （此可视化选项需要在配置中将 concept_visualization 字段设置为 'filter'。详情请参阅 ConvConceptizer 文档）

    参数
    ----------
    model: torch.nn.Module
        训练完毕且具有所有参数的模型。
    num_concepts: int
        模型的概念数量。
    num_prototypes: int
        表示一个概念的过滤器具有的通道数量。
    save_path: str
        保存条形图的路径位置。
    """
    model.eval()  # 将模型设置为评估模式。
    plt.rcdefaults()  # 重置matplotlib默认设置。
    fig, ax = plt.subplots()  # 创建图形和坐标轴。
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]  # 创建概念名称列表。

    filters = [f for f in model.conceptizer.encoder[-2][0].weight.data.clone()]  # 获取概念编码器最后一层的过滤器权重。
    imgs = [dim.unsqueeze(0) for f in filters for dim in f]  # 调整每个过滤器的维度以便可视化。

    start = 0.0
    end = num_concepts * filters[0].size(-1) + 2  # 计算绘图范围。
    stepsize = abs(end - start) / num_concepts  # 计算步长。
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))  # 设置y轴刻度。
    ax.set_yticklabels(concept_names)  # 设置y轴标签。
    plt.xticks([])  # 隐藏x轴刻度。
    ax.set_xlabel('{} dimensions of concept filters'.format(num_prototypes))  # 设置x轴标签。
    ax.set_title('Concept Prototypes: ')  # 设置标题。
    save_or_show(make_grid(imgs, nrow=num_prototypes, normalize=True, padding=1, pad_value=1), save_path)  # 显示或保存图像。
    plt.rcdefaults()  # 重置matplotlib默认设置。

def show_prototypes(model, test_loader, num_class, device, save_path, representation_type='activation', **kwargs):
    """Generates prototypes for concept representation.

    Parameters
    ----------
    model : torch nn.Module
        model to visualize
    test_loader: Dataloader object
        Test set dataloader to iterate over test set.
    representation_type : str
        Name of the representation type used.
    save_path : str
        Directory where the figures are saved. If None a figure is showed instead.
    """
    if representation_type == 'activation':
        highest_activations(model, test_loader, num_class, device, save_path=save_path)          # 调用 highest_activations 函数显示最高激活。
    elif representation_type == 'contrast':
        highest_contrast(model, test_loader, num_class, device, save_path=save_path)             # 调用 highest_contrast 函数显示对比度最高的特征。
    elif representation_type == 'filter':
        filter_concepts(model, num_class, device, save_path=save_path)                           # 调用 filter_concepts 函数来过滤并展示概念。
 