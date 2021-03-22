from torch.utils.data import Dataset
import torch
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设

from scipy.io import loadmat

from torch.utils.data import Dataset
import torch

from scipy.io import loadmat


class MyDataset(Dataset):
    def __init__(self, path, dataset):
        super(MyDataset, self).__init__()
        self.dataset = dataset
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.test_dataset, \
        self.test_label = self.pre_option(path)

    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index] - 1
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index] - 1

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len

    def pre_option(self, path):
        m = loadmat(path)

        # m中是一个字典 有4个key 其中最后一个键值对存储的是数据
        x1, x2, x3, x4 = m
        data = m[x4]

        data00 = data[0][0]
        # print('data00.shape', data00.shape)  # ()  data00才到达数据的维度

        index_train = str(data.dtype).find('train\'')
        index_trainlabels = str(data.dtype).find('trainlabels')
        index_test = str(data.dtype).find('test\'')
        index_testlabels = str(data.dtype).find('testlabels')
        list = [index_test, index_train, index_testlabels, index_trainlabels]
        list = sorted(list)
        index_train = list.index(index_train)
        index_trainlabels = list.index(index_trainlabels)
        index_test = list.index(index_test)
        index_testlabels = list.index(index_testlabels)

        # [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]  O 表示数据类型为 numpy.object
        train_label = data00[index_trainlabels]
        train_data = data00[index_train]
        test_label = data00[index_testlabels]
        test_data = data00[index_test]

        train_label = train_label.squeeze()
        train_data = train_data.squeeze()
        test_label = test_label.squeeze()
        test_data = test_data.squeeze()

        train_len = train_data.shape[0]
        test_len = test_data.shape[0]
        output_len = len(tuple(set(train_label)))

        # 时间步最大值
        max_lenth = 0  # 93
        for item in train_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        for item in test_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        # train_data, test_data为numpy.object 类型，不能直接对里面的numpy.ndarray进行处理
        train_dataset = []
        test_dataset = []
        for x1 in train_data:
            x1 = torch.as_tensor(x1).float()
            if x1.shape[1] != max_lenth:
                padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
                x1 = torch.cat((x1, padding), dim=1)
            train_dataset.append(x1)

        for x2 in test_data:
            x2 = torch.as_tensor(x2).float()
            if x2.shape[1] != max_lenth:
                padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
                x2 = torch.cat((x2, padding), dim=1)
            test_dataset.append(x2)

        # 最后维度 [数据条数,时间步数最大值,时间序列维度]
        train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
        test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
        train_label = torch.Tensor(train_label)
        test_label = torch.Tensor(test_label)
        channel = test_dataset[0].shape[-1]
        input = test_dataset[0].shape[-2]

        return train_len, test_len, input, channel, output_len, train_dataset, train_label, test_dataset, test_label

def heatMap_channel(matrix, file_name, EPOCH):
    test_data = matrix[0].detach().numpy()
    euclidean_norm = lambda x, y: np.abs(x - y)
    matrix_0 = np.ones((test_data.shape[1], test_data.shape[1]))
    matrix_1 = np.ones((test_data.shape[1], test_data.shape[1]))  # 相差度
    for i in range(test_data.shape[1]):
            for j in range(test_data.shape[1]):
                x = test_data[:, i]
                y = test_data[:, j]
                d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                matrix_0[i, j] = d
                matrix_1[i, j] = np.mean((x - y) ** 2)

    sns.set()
    # f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix_0, annot=True)
    plt.title('CHANNEL DTW')
    if os.path.exists(f'../heatmap_figure/{file_name}'):
        plt.savefig(f'../heatmap_figure/{file_name}/channel DTW EPOCH:{EPOCH}.jpg')
    else:
        os.makedirs(f'../heatmap_figure/{file_name}')
        plt.savefig(f'../heatmap_figure/{file_name}/channel DTW EPOCH:{EPOCH}.jpg')
    sns.heatmap(matrix_1, annot=True)
    plt.title('CHANNEL difference')
    plt.savefig(f'../heatmap_figure/{file_name}/channel difference EPOCH:{EPOCH}.jpg')

def heatMap_input(matrix, file_name, EPOCH):
    test_data = matrix[0].detach().numpy()
    euclidean_norm = lambda x, y: np.abs(x - y)
    matrix_0 = np.ones((test_data.shape[0], test_data.shape[0]))  # DTW
    matrix_1 = np.ones((test_data.shape[0], test_data.shape[0]))  # 相差度
    for i in range(test_data.shape[0]):
            for j in range(test_data.shape[0]):
                x = test_data[i, :]
                y = test_data[j, :]
                d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                matrix_0[i, j] = d
                matrix_1[i, j] = np.mean((x - y) ** 2)

    sns.set()
    # f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix_0, annot=True)
    plt.title('INPUT DTW')
    if os.path.exists(f'../heatmap_figure/{file_name}'):
        plt.savefig(f'../heatmap_figure/{file_name}/input DTW EPOCH:{EPOCH}.jpg')
    else:
        os.makedirs(f'../heatmap_figure/{file_name}')
        plt.savefig(f'../heatmap_figure/{file_name}/input DTW EPOCH:{EPOCH}.jpg')
    sns.heatmap(matrix_1, annot=True)
    plt.title('INPUT difference')
    plt.savefig(f'../heatmap_figure/{file_name}/input difference EPOCH:{EPOCH}.jpg')

def heatMap_score(matrix_input, matrix_channel, file_name, EPOCH):
    score_input = matrix_input[0].detach().numpy()
    score_channel = matrix_channel[0].detach().numpy()
    sns.set()
    # f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(score_input, annot=True)
    plt.title('SCORE INPUT')
    if os.path.exists(f'../heatmap_figure/{file_name}'):
        plt.savefig(f'../heatmap_figure/{file_name}/score input EPOCH:{EPOCH}.jpg')
    else:
        os.makedirs(f'../heatmap_figure/{file_name}')
        plt.savefig(f'../heatmap_figure/{file_name}/ score input EPOCH:{EPOCH}.jpg')
    sns.heatmap(score_channel, annot=True)
    plt.title('SCORE CHANNEL')
    plt.savefig(f'../heatmap_figure/{file_name}/score channel EPOCH:{EPOCH}.jpg')


if __name__ == '__main__':
    path = 'E:\\PyCharmWorkSpace\\mtsdata\\JapaneseVowels\\JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9

    dataset = MyDataset(path, 'train')
    train_dataset = dataset.train_dataset
    print(train_dataset.shape)
    test_data = train_dataset[0].numpy()
    print(test_data[12, :])
    # step_3 = train_dataset[0, 3, :].numpy()
    # step_12 = train_dataset[0, 12, :].numpy()
    # step_15 = train_dataset[0, 15, :].numpy()
    # step_25 = train_dataset[0, 25, :].numpy()
    # step_21 = train_dataset[0, 21, :].numpy()
    # step_27 = train_dataset[0, 27, :].numpy()
    #
    # # print(step_25.shape)
    # print(step_15)
    # print(step_25)
    # print(step_27)

    euclidean_norm = lambda x, y: np.abs(x - y)

    # d1, cost_matrix, acc_cost_matrix, path = dtw(step_3, step_12, dist=euclidean_norm)
    # d2, cost_matrix, acc_cost_matrix, path = dtw(step_15, step_25, dist=euclidean_norm)
    # d3, cost_matrix, acc_cost_matrix, path = dtw(step_3, step_21, dist=euclidean_norm)
    # d4, cost_matrix, acc_cost_matrix, path = dtw(step_15, step_27, dist=euclidean_norm)
    #
    # print(d1)
    # print(d2)
    # print(d3)
    # print(d4)

    print(test_data.shape)
    # matrix = np.ones((test_data.shape[0], test_data.shape[0]))  # DTW
    # matrix_1 = np.ones((test_data.shape[0], test_data.shape[0]))  # 欧氏距离
    # matrix_2 = np.ones((test_data.shape[0], test_data.shape[0]))  # 相差度
    # matrix_3 = np.ones((test_data.shape[0], test_data.shape[0]))  # 相差度
    matrix = np.ones((test_data.shape[1], test_data.shape[1]))  # DTW
    matrix_1 = np.ones((test_data.shape[1], test_data.shape[1]))  # 欧氏距离
    matrix_2 = np.ones((test_data.shape[1], test_data.shape[1]))  # 相差度
    matrix_3 = np.ones((test_data.shape[1], test_data.shape[1]))  # 点成
    for i in range(test_data.shape[1]):
            for j in range(test_data.shape[1]):
                # x = test_data[i, :]
                # y = test_data[j, :]
                x = test_data[:, i]
                y = test_data[:, j]
                d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                matrix[i, j] = d
                matrix_1[i, j] = np.sum(np.abs(x-y))
                matrix_2[i, j] = np.mean((x - y) ** 2)
                matrix_3[i, j] = float(x.dot(y))

    # print(matrix)

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    # np.random.seed(0)
    # uniform_data = np.random.rand(10, 12)
    # ax = sns.heatmap(matrix)
    # f, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(matrix, annot=True)
    plt.title('input')
    # os.mkdir('../heatmap_figure/lala')
    if os.path.exists('../heatmap_figure/lala'):
        plt.savefig(f'../heatmap_figure/lala/1.jpg')
    else:
        os.makedirs('../heatmap_figure/lala')
        plt.savefig(f'../heatmap_figure/lala/1.jpg')
    plt.show()

    plt.title('channel')
    sns.heatmap(matrix_3, annot=True)
    plt.savefig('2.jpg')
    plt.show()
