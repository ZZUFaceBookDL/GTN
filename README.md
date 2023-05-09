# Gated-Transformer-on-MTS
基于Pytorch，使用改良的Transformer模型应用于多维时间序列的分类任务上

## 实验结果
对比模型选择 Fully Convolutional Networks (FCN) and Residual Net works (ResNet) <br>


DataSet|MLP|FCN|ResNet|Encoder|MCNN|t-LeNet|MCDCNN|Time-CNN|TWIESN|Gated Transformer|
-------|---|---|------|-------|----|-------|------|--------|------|-----------------|
ArabicDigits|96.9|99.4|**99.6**|98.1|10.0|10.0|95.9|95.8|85.3|98.8|
AUSLAN|93.3|**97.5**|97.4|93.8|1.1|1.1|85.4|72.6|72.4|**97.5**|
CharacterTrajectories|96.9|**99.0**|**99.0**|97.1|5.4|6.7|93.8|96.0|92.0|97.0|
CMUsubject16|60.0|**100**|99.7|98.3|53.1|51.0|51.4|97.6|89.3|**100**|
ECG|74.8|87.2|86.7|87.2|67.0|67.0|50.0|84.1|73.7|**91.0**|
JapaneseVowels|97.6||**99.3**|99.2|97.6|9.2|23.8|94.4|95.6|96.5|98.7|
Libras|78.0|**96.4**|95.4|78.3|6.7|6.7|65.1|63.7|79.4|88.9|
UWave|90.1|**93.4**|92.6|90.8|12.5|12.5|84.5|8.9|75.4|91.0|
KickvsPunch|61.0|54.0|51.0|61.0|54.0|50.0|56.0|62.0|67.0|**90.0**|
NetFlow|55.0|89.1|62.7|77.7|77.9|72.3|63.0|89.0|94.5|**100**|
PEMS|-|-|-|-|-|-|-|-|-|93.6|
Wafer|89.4|98.2|98.9|98.6|89.4|89.4|65.8|94.8|94.9|**99.1**|
WalkvsRun|70.0|**100**|**100**|**100**|75.0|60.0|45.0|**100.0**|94.4|**100**|

## 实验环境
环境|描述|
---|---------|
语言|Python3.7|
框架|Pytorch1.6|
IDE|Pycharm and Colab|
设备|CPU and GPU|

## 数据集
多元时间序列数据集, 文件为.mat格式，训练集与测试集在一个文件中，且预先定义为了测试集数据，测试集标签，训练集数据与训练集标签。 <br>
数据集下载使用百度云盘，连接如下：<br>
  链接：https://pan.baidu.com/s/1u2HN6tfygcQvzuEK5XBa2A <br> 
  提取码：dxq6 <br>
Google drive link:https://drive.google.com/drive/folders/1QFadJOmbOLWMjLrcebZQR_w2fBX7x0Vm?usp=share_link

UEA and UCR dataset:http://www.timeseriesclassification.com/index.php

---

数据集维度描述
DataSet|Number of Classes|Size of training Set|Size of testing Set|Max Time series Length|Channel|
-------|-----------------|--------------------|-------------------|----------------------|-------|
ArabicDigits|10|6600|2200|93|13|
AUSLAN|95|1140|1425|136|22|
CharacterTrajectories|20|300|2558|205|3|
CMUsubject16|2|29|29|580|62|
ECG|2|100|100|152|2|
JapaneseVowels|9|270|370|29|12|
Libras|15|180|180|45|2|
UWave|8|200|4278|315|3|
KickvsPunch|2|16|10|841|62|
NetFlow|2|803|534|997|4|
PEMS|7|267|173|144|963|
Wafer|2|298|896|198|6|
WalkvsRun|2|28|16|1918|62|
 
## 数据预处理
详细数据集处理过程参看 dataset_process.py文件。<br>
- 创建torch.utils.data.Dataset对象，在类中对数据集进行处理，其成员变量定义的有训练集数据，训练集标签，测试集数据，测试集标签等。创建torch.utils.data.DataLoader对象，生成训练过程中的mini-batch与数据集的随机shuffle<br>
- 数据集中不同样本的Time series Length不同，处理时使用所有样本中（测试集与训练集中）**最长**的时间步作为Time series Length，使用**0**进行填充。<br>
- 数据集处理过程中保存未添加Padding的训练集数据与测试集数据，还有测试集中最长时间步的样本列表以供探索模型使用。<br>
- NetFlow数据集中标签为**1和13**，在使用此数据集时要对返回的标签值进行处理。<br>

## 模型描述

<img src="https://github.com/SY-Ma/Gated-Transformer-on-MTS/blob/main/images/GTN%20structure.png" style="zoom:50%">

- 仅使用Encoder：由于是分类任务，模型删去传统Transformer中的decoder，**仅使用Encoder**进行分类
- Two Tower：在不同的Step之间或者不同Channel之间，显然存在着诸多联系，传统Transformer使用Attentino机制用来关注不同的step或channel之间的相关程度，但仅选择一个进行计算。不同于CNN模型处理时间序列，它可以使用二维卷积核同时关注step-wise和channel-wise，在这里我们使**双塔**模型，即同时计算step-wise Attention和channel-wise Attention。
- Gate机制：对于不同的数据集，不同的Attention机制有好有坏，对于双塔的特征提取的结果，简单的方法，是对两个塔的输出尽心简单的拼接，不过在这里，我们使用模型学习两个权重值，为每个塔的输出进行权重的分配，公式如下。<br>
    `h = W · Concat(C, S) + b` <br>
    `g1, g2 = Softmax(h)` <br>
    `y = Concat(C · g1, S · g2)` <br>
- 在step-wise,模型如传统Transformer一样，添加位置编码与mask机制，而在channel-wise，模型舍弃位置编码与mask，因为对于没有时间特性的channel之间，这两个机制没有实际的意义。

## 超参描述
超参|描述|
----|---|
d_model|模型处理的为时间序列而非自然语言，所以省略了NLP中对词语的编码，仅使用一个线性层映射成d_model维的稠密向量，此外，d_model保证了在每个模块衔接的地方的维度相同|
d_hidden|Position-wise FeedForword 中隐藏层的维度| 
d_input|时间序列长度，其实是一个数据集中最长时间步的维度 **固定**的，直接由数据集预处理决定|
d_channel|多元时间序列的时间通道数，即是几维的时间序列 **固定**的，直接由数据集预处理决定|
d_output|分类类别数 **固定**的，直接由数据集预处理决定|
q,v|Multi-Head Attention中线性层映射维度|
h|Multi-Head Attention中头的数量|
N|Encoder栈中Encoder的数量|
dropout|随机失活|
EPOCH|训练迭代次数|
BATCH_SIZE|mini-batch size|
LR|学习率 定义为1e-4|
optimizer_name|优化器选择 建议**Adagrad**和Adam|

## 文件描述
文件名称|描述|
-------|----|
dataset_process|数据集处理|
font|存储字体，用于结果图中的文字|
gather_figure|聚类结果图|
heatmap_figure_in_test|测试模型时绘制的score矩阵的热力图|
module|模型的各个模块|
mytest|各种测试代码|
reslut_figure|准确率结果图|
saved_model|保存的pkl文件|
utils|工具类文件|
run.py|训练模型|
run_with_saved_model.py|使用训练好的模型（保存为pkl文件）测试结果|

## utils工具描述
简单介绍几个
- random_seed:用于设置**随机种子**，使每一次的实验结果可复现。
- heatMap.py:用于绘制双塔的score矩阵的**热力图**，用来分析channel与channel之间或者step与step之间的相关程度，用于比较的还有**DTW**矩阵和欧氏距离矩阵，用来分析决定权重分配的因素。
- draw_line:用于绘制折线图，一般需要根据需要自定义新的函数进行绘制。
- visualization:用于绘制训练模型的loss变化曲线和accuracy变化曲线，判断是否收敛与过拟合。
- TSNE：**降维聚类算法**并绘制聚类图，用来评估模型特征提取的效果或者时间序列之间的相似性。

## Tips
- .pkl文件需要先训练，并在训练结束时进行保存(设置参数为True)，由于github对文件大小的限制，上传文件中不包含训练好的.pkl文件。
- .pkl文件使用pycharm上的1.6版本的pytorch和colab上1.7的pytorch保存，若想load模型直接进行测试，需要测试使用的pytorch版本尽可能高于等于**1.6版本**。
- 根目录文件如saved_model,reslut_figure为保存的默认路径，请勿删除或者修改名称，除非直接在源代码中对路径进行修改。
- 请使用百度云盘提供的数据集，不同的MTS数据集文件格式不同，本数据集处理的是.mat文件。
- utils中的工具类，在绘制彩色曲线和聚类图时，对于图中颜色的划分，由于需求不能泛化，请在函数中自行编写代码定义。
- save model保存的.pkl文件在迭代过程中不断更新，在最后保存最高准确率的模型并命名，命名格式请勿修改，因为在run_with_saved_model.py中，对文件命名中的信息会加以利用，若干绘图结果的命名也会参考其中的信息。
- 优先选择GPU，没有则使用CPU。

## 参考
```
[Wang et al., 2017] Z. Wang, W. Yan, and T. Oates. Time series classification from scratch with deep neural networks:A strong baseline. In 2017 International Joint Conference on Neural Networks (IJCNN), pages 1578–1585, 2017.
```

## 本人学识浅薄，代码和文字若有不当之处欢迎批评与指正！
## 联系方式：masiyuan007@qq.com
