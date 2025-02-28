

import cfg
import matplotlib.pyplot as plt
from icecream import ic
import pandas as pd
import numpy as np
from Utils import helpers
import seaborn as sns

fig=plt.figure(figsize=(20, 5))

def plot_loss_acc2(path, num_epoch, train_accuracies, train_losses, test_accuracies, test_losses, plot_name,dpi = 500):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''

    plt.clf()

    epochs = np.arange(num_epoch+1)

    train_accuracy_data = {"Epochs": epochs, "Accuracy": train_accuracies}
    test_accuracy_data = {"Epochs": epochs, "Accuracy": test_accuracies}

    train_loss_data = {"Epochs": epochs, "Loss": train_losses}
    test_loss_data = {"Epochs": epochs, "Loss": test_losses}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # plot accuracy graph
    ax[0].plot(train_accuracy_data["Epochs"], train_accuracy_data["Accuracy"], label="Train")
    ax[0].plot(test_accuracy_data["Epochs"], test_accuracy_data["Accuracy"], label="Test")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy Graph")
    ax[0].legend(loc="best")

    # plot loss graph
    ax[1].plot(train_loss_data["Epochs"], train_loss_data["Loss"], label="Train")
    ax[1].plot(test_loss_data["Epochs"], test_loss_data["Loss"], label="Test")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Loss Graph")
    ax[1].legend(loc="best")
    
    helpers.check_path(path)
        
    # print(f'Saved image to{path}')
    plt.savefig(path+plot_name+f'loss_epoch_{num_epoch}.png',dpi=dpi)
    plt.close()

    return None


def plot_reconstruction(path, num_epoch, original_images, reconstructed_images,dpi = 500):
    '''
    Plot a grid of original and reconstructed images.
    '''

    plt.clf()

    original_images = original_images.detach().cpu()
    reconstructed_images = reconstructed_images.detach().cpu()
    reconstructed_images = reconstructed_images.view(-1,1,32,32)
    
    # ic(original_images.size())
    # ic(reconstructed_images.size())
    
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))

    for i in range(5):
        
        axs[0, i].imshow(original_images[i].squeeze(), cmap='gray')
        axs[0, i].set_title("Original")
        axs[1, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
        axs[1, i].set_title("Reconstructed")

    plt.suptitle(f"Reconstruction Results (Epoch: {num_epoch})")
    #TODO: delete shallow
    plt.savefig(path+f'shallow_reconstruction_epoch_{num_epoch}.png',dpi=dpi)
    plt.close()

    return None

def plot_histogram(avg_similarities,path,plot_name,dpi = 500):
    # Compute average similarity for each capsule
    # avg_similarities = np.mean(similarity_matrix, axis=1)
    
    # Plot histogram
    plt.figure(figsize=[8,6])
    # counts, _, _ = plt.hist(avg_similarities, bins=12, edgecolor='black', linewidth=0.7)
    sns.swarmplot(x=avg_similarities)
    plt.xticks(np.arange(0.2, 0.8, step=0.1))
    # plt.yticks(np.arange(0, max(counts) + 1, step=max(counts) // 10), 
    #            np.arange(0, max(counts) // 6 + 1, step=max(counts) // 60).astype(int))
    # total_capsules = avg_similarities.size
    # percentages = (np.arange(0, max(counts) + 1, step=max(counts) // 10) / total_capsules) * 100
    # plt.yticks(np.arange(0, max(counts) + 1, step=max(counts) // 10), 
    #            [f'{percentage:.1f}%' for percentage in percentages])
    
    plt.xlabel('Average Cosine Similarity')
    plt.ylabel('Percentage of Capsules')
    plt.title('Histogram of Average Cosine Similarities for Capsules')
    
    plt.savefig(path+plot_name+'.png',dpi=dpi)
    print(f'Saved image to{path}')
    
    
    
    
def plot_orth(epoch, orthogonality ,path, plot_name, dpi = 500):
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch, orthogonality, label='Orthogonality')
    plt.xlabel('Epoch')
    plt.ylabel('Orthogonality')
    plt.title('Orthogonality over Epochs')
    
    plt.savefig(path+plot_name+'.png',dpi=dpi)
    print(f'Saved image to{path}')