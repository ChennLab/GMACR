import os
import numpy as np
import seaborn as sns
import pickle
import torch
import random
from tqdm import tqdm, trange
#  序列化与反序列化
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 计算分类输出结果的准确率
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_code(predicted, target):
    # 确保预测和目标张量的形状相同
    assert predicted.shape == target.shape, "Predicted and target shapes must be the same"

    # 计算正确预测的数量
    correct_predictions = (predicted == target).sum().item()

    # 计算准确率
    accuracy = correct_predictions / target.numel()

    return accuracy


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def show_it(imgs, output_dir, epoch):
    for i, img in enumerate(imgs):
        img = img.squeeze(0)
        img = img.cpu().detach().numpy()

        file_name = f'epoch[{epoch}]_i[{i}].png'

        output_path = os.path.join(output_dir, file_name)

        plt.figure(figsize=(8, 8), dpi=200)
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample {i + 1} - Grad-CAM Heatmap")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # 关闭图形以释放内存
        # plt.show()

def tv_loss(generated_counterfactual):
    """
    Total Variation (TV) 损失。

    Args:
        generated_counterfactual: 生成的反事实图 (Tensor).

    Returns:
        TV 损失值 (Tensor).
    """
    batch_size = generated_counterfactual.size()[0]
    h_x = generated_counterfactual.size()[2]
    w_x = generated_counterfactual.size()[3]
    h_tv = torch.mean(torch.abs(generated_counterfactual[:,:,1:,:] - generated_counterfactual[:,:,:h_x-1,:])) # 水平方向 TV
    w_tv = torch.mean(torch.abs(generated_counterfactual[:,:,:,1:] - generated_counterfactual[:,:,:,:w_x-1])) # 垂直方向 TV
    loss_tv = h_tv + w_tv
    return loss_tv

def visualize_generated_images(image_batch):
    """
    可视化一个批次的生成图像 (假设为灰度图像).

    Args:
        image_batch (torch.Tensor): 形状为 (batch_size, channels, height, width) 的图像批次张量.
                                    假设 channels=1 代表灰度图像，batch_size <= 16.
    """
    batch_size = image_batch.size(0)
    if batch_size > 16:
        print(f"警告: 批次大小超过 16.  只显示前 16 张图像.")
        batch_size = 16

    # 将 PyTorch 张量转换为 NumPy 数组，并移动到 CPU (如果需要)
    image_batch_np = image_batch.detach().cpu().numpy()

    # 创建一个 4x4 的子图网格来显示最多 16 张图像
    fig, axes = plt.subplots(4, 4, figsize=(8, 8)) # 可以调整 figsize 来改变整体图片大小
    fig.subplots_adjust(wspace=0.1, hspace=0.1) # 调整子图之间的间距

    for i in range(batch_size):
        row_idx = i // 4 # 计算行索引
        col_idx = i % 4  # 计算列索引
        ax = axes[row_idx, col_idx]

        # 获取单张图像，并去除通道维度 (假设为灰度图像，通道数为 1)
        image = image_batch_np[i].squeeze(0) # 从 (1, 128, 128) 变为 (128, 128)

        # 显示图像，使用 'gray' colormap 显示灰度图像
        ax.imshow(image, cmap='gray') # 使用 'gray' colormap 显示灰度图像

        # 移除坐标轴刻度和标签
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off') # 也可以直接关闭坐标轴

    plt.tight_layout() # 自动调整子图参数, 使得子图填充整个画布区域，避免重叠
    plt.show()

def visualize_first_image_in_batch(image_batch):
    """
    可视化一个批次中的第一张图像 (假设为灰度图像).

    Args:
        image_batch (torch.Tensor): 形状为 (batch_size, channels, height, width) 的图像批次张量.
                                    假设 channels=1 代表灰度图像.
    """
    # 检查批次大小是否至少为 1
    if image_batch.size(0) < 1:
        print("警告: 批次大小为空，无法显示图像.")
        return

    # 获取批次中的第一张图像 (索引为 0)
    first_image = image_batch[0]

    # 将 PyTorch 张量转换为 NumPy 数组，并移动到 CPU (如果需要)
    first_image_np = first_image.detach().cpu().numpy()

    # 去除通道维度 (假设为灰度图像，通道数为 1)
    image = np.squeeze(first_image_np) # 从 (1, 128, 128) 变为 (128, 128) or from (128, 128) if channel dim was already removed

    # 显示图像，使用 'gray' colormap 显示灰度图像
    plt.imshow(image, cmap='gray')  # 使用 'gray' colormap 显示灰度图像

    # 移除坐标轴刻度和标签
    plt.xticks([])
    plt.yticks([])
    plt.axis('off') # 也可以直接关闭坐标轴

    plt.title("First Image in Batch") # 添加标题，可选
    plt.tight_layout()
    plt.show()

def visualize_latent_space(latents, labels, method_name):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap

    plt.figure(figsize=(8, 6))
    if method_name == 'PCA':
        reducer = PCA(n_components=2)
    elif method_name == 't-SNE':
        reducer = TSNE(n_components=2, random_state=0)
    elif method_name == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=0)
    else:
        raise ValueError("Invalid method_name. Choose from 'PCA', 't-SNE', 'UMAP'.")

    reduced_latents = reducer.fit_transform(latents)

    # 分类别颜色
    scatter = plt.scatter(reduced_latents[:, 0], reduced_latents[:, 1], c=labels, cmap=plt.cm.RdBu, alpha=0.7) # RdBu 色彩映射，可以根据需要调整
    plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1']) # 添加图例
    plt.title(f'Latent Space Visualization using {method_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(scatter, ticks=[0, 1], label='Class Label') # 添加颜色条
    plt.show()
