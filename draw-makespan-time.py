import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
if __name__ == "__main__":
    x_labels = ['la01', 'la02', 'la03', 'la04', 'la05', 'la06', 'la07', 'la08', 'la09', 'la10',
                'la11', 'la12', 'la13', 'la14', 'la15', 'la16', 'la17', 'la18', 'la19', 'la20',
                'la21', 'la22', 'la23', 'la24', 'la25', 'la26', 'la27', 'la28', 'la29', 'la30',
                'la31', 'la32', 'la33', 'la34', 'la35', 'la36', 'la37', 'la38', 'la39', 'la40', ]
    x_labels1 = ['MK1', 'MK2', 'MK3', 'MK4', 'MK5', 'MK6', 'MK7', 'MK8', 'MK9', 'MK10']

    trajectories = pd.read_excel("draw_makespan_time/simple-trajectory.xls")
    traj = trajectories.values[:, 1:6]
    time = pd.read_excel("draw_makespan_time/simple-time.xls")
    total_time = time.values[:, 1:6]

    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    palette = pyplot.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    avg_traj = np.mean(traj, axis=1)
    std_traj = np.std(traj, axis=1, dtype=np.float64)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg_traj, std_traj)))  # 上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg_traj, std_traj)))  # 下方差

    avg_t = np.mean(total_time, axis=1)
    std_t = np.std(total_time, axis=1, dtype=np.float64)
    u_t = list(map(lambda x: x[0] - x[1], zip(avg_t, std_t)))  # 上方差
    l_t = list(map(lambda x: x[0] + x[1], zip(avg_t, std_t)))  # 下方差

    fig = plt.figure()
    ax_list = []
    index = x_labels
    color = palette(0)  # 算法1颜色
    ax = fig.add_subplot(1, 1, 1)
    # ax_list += ax.plot(index, avg_traj, color=color, label="训练轨迹数", linewidth=1.5)
    ax_list += ax.plot(index, avg_traj, color=color, label="training trajectories", linewidth=1.5)
    ax.fill_between(index, r1, r2, color=color, alpha=0.2)
    # ax.legend(loc='upper left', prop=font1)
    # ax.set_xlabel('调度实例', fontdict=font1)
    # ax.set_ylabel('轨迹数', fontdict=font1)
    ax.set_xlabel('Instances', fontdict=font1)
    ax.set_ylabel('The number of trajectories', fontdict=font1)

    color2 = palette(1)  # 算法2颜色

    ax2 = ax.twinx()
    # ax_list += ax2.plot(index, avg_t, color=color2, label="训练时间", linewidth=1.5)
    ax_list += ax2.plot(index, avg_t, color=color2, label="training time", linewidth=1.5)
    ax2.fill_between(index, u_t, l_t, color=color2, alpha=0.2)
    # ax2.set_ylabel('训练时间  /s ', fontdict=font1)
    ax2.set_ylabel('Training time  /s ', fontdict=font1)
    labels = [x.get_label() for x in ax_list]
    ax2.legend(ax_list, labels, loc='upper left', prop=font1)

    plt.show()
    print("hello")


