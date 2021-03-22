import matplotlib.pyplot as plt


def draw_line(list_1, list_2, list_3):
    plt.style.use('seaborn')
    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(list_3, color='red', label='Target_Weight_for_step')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('value')
    ax1.set_title('One Sample Weight for Step')

    ax2.plot(list_1, color='red', label='Weight_for_step')
    ax2.plot(list_2, color='blue', label='Weight_for_channel')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('value')
    ax2.set_title('Mean Weight Allocation')

    plt.legend(loc='best')
    plt.show()

def draw_heatmap_anylasis(sample):
    channel_00 = sample[:, 0].numpy()
    channel_11 = sample[:, 1].numpy()
    sample = sample.transpose(-1, -2)
    channel_0 = sample[0, :].numpy().tolist()
    channel_1 = sample[1, :].numpy().tolist()
    channel_6 = sample[6, :].numpy().tolist()
    channel_4 = sample[4, :].numpy().tolist()
    channel_3 = sample[3, :].numpy().tolist()
    channel_7 = sample[7, :].numpy().tolist()
    channel_11 = sample[11, :].numpy().tolist()
    channel_9 = sample[9, :].numpy().tolist()

    ax1 = plt.subplot(511)
    ax2 = plt.subplot(512)
    ax3 = plt.subplot(513)
    ax4 = plt.subplot(514)
    ax5 = plt.subplot(515)
    ax1.plot(channel_11, color='red', label='channel_11')
    ax1.plot(channel_7, label='channel_7')
    ax2.plot(channel_11, color='red', label='channel_11')
    ax2.plot(channel_6, label='channel_6')
    ax3.plot(channel_11, color='red', label='channel_11')
    ax3.plot(channel_3, label='channel_3')
    ax4.plot(channel_3, color='red', label='channel_3')
    ax4.plot(channel_4, label='channel_4')
    ax5.plot(channel_9, color='red', label='channel_6')
    ax5.plot(channel_0, label='channel_0')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')
    plt.suptitle('Feature-wise Series Compare')
    plt.show()