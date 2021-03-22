import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

def heatMap_all(score_input: torch.Tensor,  # 二维数据
                score_channel: torch.Tensor,  # 二维数据
                x: torch.Tensor,  # 二维数据
                save_root: str,
                file_name: str,
                accuracy: str,
                index: int) -> None:
    score_channel = score_channel.detach().numpy()
    score_input = score_input.detach().numpy()
    draw_data = x.detach().numpy()

    euclidean_norm = lambda x, y: np.abs(x - y)  # 用于计算DTW使用的函数,此处是一个计算欧氏距离的函数

    matrix_00 = np.ones((draw_data.shape[1], draw_data.shape[1]))  # 用于记录channel之间DTW值的矩阵
    # matrix_01 = np.ones((draw_data.shape[1], draw_data.shape[1]))  # 用于记录channel之间相差度的矩阵

    # matrix_10 = np.ones((draw_data.shape[0], draw_data.shape[0]))  # 用于记录input之间DTW值的矩阵
    matrix_11 = np.ones((draw_data.shape[0], draw_data.shape[0]))  # 用于记录input之间相差度值的矩阵

    for i in range(draw_data.shape[0]):
        for j in range(draw_data.shape[0]):
            x = draw_data[i, :]
            y = draw_data[j, :]
            # d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
            # matrix_10[i, j] = d
            matrix_11[i, j] = np.sqrt(np.sum((x - y) ** 2))

    draw_data = draw_data.transpose(-1, -2)
    for i in range(draw_data.shape[0]):
        for j in range(draw_data.shape[0]):
            x = draw_data[i, :]
            y = draw_data[j, :]
            d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
            matrix_00[i, j] = d
            # matrix_01[i, j] = np.mean((x - y) ** 2)

    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置figure_size尺寸
    plt.subplot(221)
    sns.heatmap(score_channel, cmap="YlGnBu", vmin=0)
    # plt.title('channel-wise attention')

    plt.subplot(222)
    sns.heatmap(matrix_00, cmap="YlGnBu", vmin=0)
    # plt.title('channel-wise DTW')

    plt.subplot(223)
    sns.heatmap(score_input, cmap="YlGnBu", vmin=0)
    # plt.title('step-wise attention')

    plt.subplot(224)
    sns.heatmap(matrix_11, cmap="YlGnBu", vmin=0)
    # plt.title('step-wise L2 distance')

    # plt.suptitle(f'{file_name.lower()}')

    if os.path.exists(f'{save_root}/{file_name}') == False:
        os.makedirs(f'{save_root}/{file_name}')
    plt.savefig(f'{save_root}/{file_name}/{file_name} accuracy={accuracy} {index}.jpg', dpi=400)

    # plt.show()
    plt.close()


if __name__ == '__main__':
    matrix = torch.Tensor(range(24)).reshape(2, 3, 4)
    print(matrix.shape)
    file_name = 'lall'
    epcoh = 1

    data_channel = matrix.detach()
    data_input = matrix.detach()

    plt.subplot(2, 2, 1)
    sns.heatmap(data_channel[0].data.cpu().numpy())
    plt.title("1")

    plt.subplot(2, 2, 2)
    sns.heatmap(data_input[0].data.cpu().numpy())
    plt.title("2")

    plt.subplot(2, 2, 3)
    sns.heatmap(data_input[0].data.cpu().numpy())
    plt.title("3")

    plt.subplot(2, 2, 4)
    sns.heatmap(data_input[0].data.cpu().numpy())
    plt.title("4")

    plt.suptitle("JapaneseVowels Attention Heat Map", fontsize='x-large', fontweight='bold')
    # plt.savefig('result_figure/JapaneseVowels Attention Heat Map.png')
    plt.show()
