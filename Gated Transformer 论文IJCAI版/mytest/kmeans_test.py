# import cPickle
# X,y = cPickle.load(open('data.pkl','r'))  #X和y都是numpy.ndarray类型
# X.shape  #输出(1000,2)
# y.shape  #输出(1000,)对应每个样本的真实标签
from dataset_process.dataset_process import MyDataset

# path = 'E:\\PyCharmWorkSpace\\mtsdata\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\JapaneseVowels\\JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9
path = 'E:\\PyCharmWorkSpace\\mtsdata\\KickvsPunch\\KickvsPunch.mat'  # lenth=10  input=841 channel=62 output=2



dataset = MyDataset(path, 'train')
X = dataset.train_dataset
# X = torch.mean(X, dim=1).numpy()
X = X.reshape(X.shape[0], -1).numpy()
y = dataset.train_label.numpy()


import numpy as np
import matplotlib.pyplot as plt
from utils.kmeans import KMeans
def draw(X, Y):
    clf = KMeans(n_clusters=2, initCent='random' ,max_iter=100)
    clf.fit(X)
    cents = clf.centroids#质心
    labels = clf.labels#样本点被分配到的簇的索引
    sse = clf.sse
    #画出聚类结果，每一类用一种颜色
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    n_clusters = 2
    for i in range(n_clusters):
        index = np.nonzero(labels==i)[0]
        x0 = X[index,0]
        x1 = X[index,1]
        y_i = Y[index]
        for j in range(len(x0)):
            plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
                    fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],linewidths=12)
    plt.title("SSE={:.2f}".format(sse))
    plt.axis([-30,30,-30,30])
    plt.show()