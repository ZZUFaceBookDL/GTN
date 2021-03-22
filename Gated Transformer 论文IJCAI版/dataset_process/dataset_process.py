from torch.utils.data import Dataset
import torch

from scipy.io import loadmat


class MyDataset(Dataset):
    def __init__(self,
                 path: str,
                 dataset: str):
        """
        训练数据集与测试数据集的Dataset对象
        :param path: 数据集路径
        :param dataset: 区分是获得训练集还是测试集
        """
        super(MyDataset, self).__init__()
        self.dataset = dataset  # 选择获取测试集还是训练集
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.test_dataset, \
        self.test_label, \
        self.max_length_sample_inTest, \
        self.train_dataset_with_no_paddding = self.pre_option(path)

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

    # 数据预处理
    def pre_option(self, path: str):
        """
        数据预处理  由于每个样本的时间步维度不同，在此使用最长的时间步作为时间步的维度，使用0进行填充
        :param path: 数据集路径
        :return: 训练集样本数量，测试集样本数量，时间步维度，通道数，分类数，训练集数据，训练集标签，测试集数据，测试集标签，测试集中时间步最长的样本列表，没有padding的训练集数据
        """
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
                # max_length_index = train_data.tolist().index(item.tolist())

        for item in test_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        # 填充Padding  使用0进行填充
        # train_data, test_data为numpy.object 类型，不能直接对里面的numpy.ndarray进行处理
        train_dataset_with_no_paddding = []
        test_dataset_with_no_paddding = []
        train_dataset = []
        test_dataset = []
        max_length_sample_inTest = []
        for x1 in train_data:
            train_dataset_with_no_paddding.append(x1.transpose(-1, -2).tolist())
            x1 = torch.as_tensor(x1).float()
            if x1.shape[1] != max_lenth:
                padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
                x1 = torch.cat((x1, padding), dim=1)
            train_dataset.append(x1)

        for index, x2 in enumerate(test_data):
            test_dataset_with_no_paddding.append(x2.transpose(-1, -2).tolist())
            x2 = torch.as_tensor(x2).float()
            if x2.shape[1] != max_lenth:
                padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
                x2 = torch.cat((x2, padding), dim=1)
            else:
                max_length_sample_inTest.append(x2.transpose(-1, -2))
            test_dataset.append(x2)

        # 最后维度 [数据条数,时间步数最大值,时间序列维度]
        # train_dataset_with_no_paddding = torch.stack(train_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
        # test_dataset_with_no_paddding = torch.stack(test_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
        train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
        test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
        train_label = torch.Tensor(train_label)
        test_label = torch.Tensor(test_label)
        channel = test_dataset[0].shape[-1]
        input = test_dataset[0].shape[-2]

        return train_len, test_len, input, channel, output_len, train_dataset, train_label, test_dataset, test_label, max_length_sample_inTest, train_dataset_with_no_paddding