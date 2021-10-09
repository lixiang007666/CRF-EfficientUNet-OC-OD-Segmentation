import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plt.xlabel("Batch", fontsize=14)
plt.ylabel("DICE", fontsize=14)
net2 = pd.read_csv('run-fc-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net2.Step, net2.Value, lw=1.5, label='Dice+Focal Loss,$alpha$', color='red')
net3 = pd.read_csv('run-dc_fc_alpha_0.25_r_2-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net3 .Step, net3 .Value, lw=1.5, label='$公式符号$', color='blue')
net4 = pd.read_csv('run-dc-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net4 .Step, net4 .Value, lw=1.5, label='Net-4', color='green')
net = pd.read_csv('run-.-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net .Step, net .Value, lw=1.5, label='Net-4', color='black')
net5 = pd.read_csv('run-1_0.25-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net5 .Step, net5 .Value, lw=1.5, label='Net-4', color='orange')
net6 = pd.read_csv('run-0_0.75-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net6 .Step, net6 .Value, lw=1.5, label='Net-4', color='#662FA3')
net7 = pd.read_csv('run-0.5_0.5-tag-batch_categorical_accuracy.csv', usecols=['Step', 'Value'])
ax.plot(net7 .Step, net7 .Value, lw=1.5, label='Net-4', color='#A37B0D')
ax.legend(labels=[r"Dice+Focal Loss, $\alpha$=0.25, $\gamma$=2", r"Focal Loss, $\alpha$=0.25, $\gamma$=2","Dice Loss",r"Dice+Focal Loss, $\alpha$=5, $\gamma$=0.25",r"Dice+Focal Loss, $\alpha$=0.25, $\gamma$=1",r"Dice+Focal Loss, $\alpha$=0.75, $\gamma$=0",r"Dice+Focal Loss, $\alpha$=0.5, $\gamma$=0.5"])
axins = inset_axes(ax, width="45%", height="25%", loc='lower left',
                   bbox_to_anchor=(0.2, 0.62, 1, 1),
                   bbox_transform=ax.transAxes)

axins.plot(net2.Step, net2.Value, lw=1.5, label='Ranger', color='red')

axins.plot(net3 .Step, net3 .Value, lw=1.5, label='Net-3', color='blue')

axins.plot(net4 .Step, net4 .Value, lw=1.5, label='Net-4', color='green')
axins.plot(net .Step, net .Value, lw=1.5, label='Net-4', color='black')
axins.plot(net5 .Step, net5 .Value, lw=1.5, label='Net-4', color='#cba0e6')
axins.plot(net6 .Step, net6 .Value, lw=1.5, label='Net-4', color='#662FA3')
axins.plot(net7 .Step, net7 .Value, lw=1.5, label='Net-4', color='#A37B0D')

# 设置放大区间
zone_left = 10
zone_right = 40

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5  # x轴显示范围的扩展比例
y_ratio = 0.5  # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = net2.Step[zone_left] - (net2.Step[zone_right] - net2.Step[zone_left]) * x_ratio
xlim1 = net2.Step[zone_right] + (net2.Step[zone_right] - net2.Step[zone_left]) * x_ratio

# Y轴的显示范围
y = np.hstack((net2.Value[zone_left:zone_right], net2.Value[zone_left:zone_right], net2.Value[zone_left:zone_right]))
ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
plt.show()