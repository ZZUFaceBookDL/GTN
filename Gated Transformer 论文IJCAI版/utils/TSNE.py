from dataset_process.dataset_process import MyDataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# path = 'E:\\PyCharmWorkSpace\\mtsdata\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2
#
# dataset = MyDataset(path, 'train')
# X = dataset.train_dataset
# X = torch.mean(X, dim=1).numpy()
# # X = X.reshape(X.shape[0], -1).numpy()
# y = dataset.train_label.numpy()

from matplotlib import cm


def plot_with_labels(lowDWeights, labels, kinds, file_name):
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255/kinds * s)) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, s, backgroundcolor=c, fontsize=6)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())

    plt.title('Clustering Step-Wise after Embedding')
    plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 设置figure_size尺寸

    if os.path.exists(f'gather_figure/{file_name.split(" ")[0]}') == False:
        os.makedirs(f'gather_figure/{file_name.split(" ")[0]}')

    plt.savefig(f'gather_figure/{file_name.split(" ")[0]}/{file_name}.jpg', dpi=600)
    # plt.show()
    plt.close()

def plot_only(lowDWeights, labels, index, file_name):
    """
    绘制聚类图并为标签打上颜色
    :param lowDWeights: 将为之后的用于绘制聚类图的数据
    :param labels: lowDWeights对应的标签
    :param index: 用于命名文件是进行区分 防止覆盖
    :param file_name: 文件名称和聚类的方式
    :return: None
    """
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    # 聚类图中自定义的颜色的绘制请在下面for循环中完成
    for x, y, s in zip(X, Y, labels):
        position = 255
        if x < -850:
            position = 255
        elif 0.5*x - 225 < y:
            position = 0
        elif x < 1500:
            position = 50
        else:
            position = 100

        # c = cm.rainbow(int(255/9 * s)) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        c = cm.rainbow(position)  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, s, backgroundcolor=c, fontsize=6)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    # plt.title('Clustering Step-Wise after Embedding')

    plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 设置figure_size尺寸

    if os.path.exists(f'gather_figure/{file_name.split(" ")[0]}') == False:
        os.makedirs(f'gather_figure/{file_name.split(" ")[0]}')

    plt.savefig(f'gather_figure/{file_name.split(" ")[0]}/{file_name} {index}.jpg', dpi=600)
    # plt.show()
    plt.close()


from sklearn.manifold import TSNE


def gather_by_tsne(X: np.ndarray,
                   Y: np.ndarray,
                   index: int,
                   file_name: str):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=4000)  # TSNE降维，降到2
    low_dim_embs = tsne.fit_transform(X[:, :])
    labels = Y[:]
    plot_only(low_dim_embs, labels, index, file_name)


def gather_all_by_tsne(X: np.ndarray,
                   Y: np.ndarray,
                   kinds: int,
                   file_name: str):
    """
    对gate之后的二维数据进行聚类
    :param X: 聚类数据 2维数据
    :param Y: 聚类数据对应标签
    :param kinds: 分类数
    :param file_name: 用于文件命名
    :return: None
    """
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=4000)  # TSNE降维，降到2
    low_dim_embs = tsne.fit_transform(X[:, :])
    labels = Y[:]
    plot_with_labels(low_dim_embs, labels, kinds, file_name)


if __name__ == '__main__':
    path = 'E:\\PyCharmWorkSpace\\mtsdata\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2

    dataset = MyDataset(path, 'train')
    X = dataset.train_dataset
    X = torch.mean(X, dim=1).numpy()
    # X = X.reshape(X.shape[0], -1).numpy()
    Y = dataset.train_label.numpy()
    gather_by_tsne(X, Y)
