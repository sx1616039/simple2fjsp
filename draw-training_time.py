import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
if __name__ == "__main__":
    x_labels = ['la01', 'la02', 'la03', 'la04', 'la05', 'la06', 'la07', 'la08', 'la09', 'la10',
                'la11', 'la12', 'la13', 'la14', 'la15', 'la16', 'la17', 'la18', 'la19', 'la20',
                'la21', 'la22', 'la23', 'la24', 'la25', 'la26', 'la27', 'la28', 'la29', 'la30',
                'la31', 'la32', 'la33', 'la34', 'la35', 'la36', 'la37', 'la38', 'la39', 'la40', ]
    # x_labels1 = ['MK1', 'MK2', 'MK3', 'MK4', 'MK5', 'MK6', 'MK7', 'MK8', 'MK9', 'MK10']
    time_e = pd.read_excel("draw_makespan_time/simple-time - edata.xls")
    total_time_e = time_e.values[:, 1:6]
    time_r = pd.read_excel("draw_makespan_time/simple-time - rdata.xls")
    total_time_r = time_r.values[:, 1:6]
    time_v = pd.read_excel("draw_makespan_time/simple-time - vdata.xls")
    total_time_v = time_v.values[:, 1:6]

    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
    palette = pyplot.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    avg_t_e = np.mean(total_time_e, axis=1)
    std_t_e = np.std(total_time_e, axis=1, dtype=np.float64)
    u_t_e = list(map(lambda x: x[0] - x[1], zip(avg_t_e, std_t_e)))  # 上方差
    l_t_e = list(map(lambda x: x[0] + x[1], zip(avg_t_e, std_t_e)))  # 下方差

    avg_t_r = np.mean(total_time_r, axis=1)
    std_t_r = np.std(total_time_r, axis=1, dtype=np.float64)
    u_t_r = list(map(lambda x: x[0] - x[1], zip(avg_t_r, std_t_r)))  # 上方差
    l_t_r = list(map(lambda x: x[0] + x[1], zip(avg_t_r, std_t_r)))  # 下方差

    avg_t_v = np.mean(total_time_v, axis=1)
    std_t_v = np.std(total_time_v, axis=1, dtype=np.float64)
    u_t_v = list(map(lambda x: x[0] - x[1], zip(avg_t_v, std_t_v)))  # 上方差
    l_t_v = list(map(lambda x: x[0] + x[1], zip(avg_t_v, std_t_v)))  # 下方差

    fig = plt.figure()
    ax_list = []
    index = x_labels
    color = palette(0)  # 算法1颜色
    color1 = palette(1)  # 算法1颜色
    color2 = palette(2)  # 算法2颜色
    ax = fig.add_subplot(1, 1, 1)
    ax_list += ax.plot(index, avg_t_e, color=color, label="edata", linewidth=1.5)
    ax.fill_between(index, u_t_e, l_t_e, color=color, alpha=0.2)
    ax_list += ax.plot(index, avg_t_r, color=color1, label="rdata", linewidth=1.5)
    ax.fill_between(index, u_t_r, l_t_r, color=color1, alpha=0.2)
    ax_list += ax.plot(index, avg_t_v, color=color2, label="vdata", linewidth=1.5)
    ax.fill_between(index, u_t_v, l_t_v, color=color2, alpha=0.2)
    ax.set_xlabel('Instances', fontdict=font1)
    ax.set_ylabel('Training time  /s ', fontdict=font1)

    labels = [x.get_label() for x in ax_list]
    ax.legend(ax_list, labels, loc='upper left', prop=font1)
    plt.show()
    print("hello")


