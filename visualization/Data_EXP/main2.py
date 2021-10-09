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
plt.ylabel("Loss", fontsize=14)
net3 = pd.read_csv('run-radam-tag-batch_loss.csv', usecols=['Step', 'Value'])
ax.plot(net3 .Step, net3 .Value, lw=1.5, label='Net-3', color='blue')
net2 = pd.read_csv('run-ranger-tag-batch_loss.csv', usecols=['Step', 'Value'])
ax.plot(net2.Step, net2.Value, lw=1.5, label='Ranger', color='red')

net4 = pd.read_csv('run-adam-tag-batch_loss.csv', usecols=['Step', 'Value'])
ax.plot(net4 .Step, net4 .Value, lw=1.5, label='Net-4', color='green')
net5 = pd.read_csv('run-SGD-tag-batch_loss.csv', usecols=['Step', 'Value'])
ax.plot(net5 .Step, net5 .Value, lw=1.5, label='Net-4', color='orange')
ax.legend(labels=["RAdam+LookAhead(Ranger)", "RAdam","Adam",'SGD'])
axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.4, 0.3, 1, 1),
                   bbox_transform=ax.transAxes)

axins.plot(net2.Step, net2.Value, lw=1.5, label='Ranger', color='red')

axins.plot(net3 .Step, net3 .Value, lw=1.5, label='Net-3', color='blue')

axins.plot(net4 .Step, net4 .Value, lw=1.5, label='Net-4', color='green')
axins.plot(net5 .Step, net5 .Value, lw=1.5, label='Net-4', color='#cba0e6')
# 设置放大区间
zone_left = 7
zone_right = 23

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
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
plt.show()