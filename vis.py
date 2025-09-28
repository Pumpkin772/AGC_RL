# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取 CSV 文件
# df = pd.read_csv("D:\桌面\simulink实验\\train\loss\sac_idp_v3_3seq_10beta_train20250725-224825.csv")
# df.columns = df.columns.str.strip()
#  # 根据你的格式使用制表符分隔
# print(df.columns)
# print(df.head())         # 查看前几行数据
# print(df.columns.tolist())  # 显示所有列名
# # 可视化 loss 随 step 的变化
# plt.figure(figsize=(10, 6))
# plt.plot(df['step'], df['loss'], marker='o', markersize=2, linestyle='-', color='#1f77b4', linewidth=2)
# plt.plot(df['step'], df['lossA'], marker='o', markersize=2, linestyle='-', color="#2ca03b", linewidth=2)
# plt.plot(df['step'], df['lossF'], marker='o', markersize=2, linestyle='-', color="#ffdf0e", linewidth=2)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# 读取 CSV 文件
df = pd.read_csv(r"D:\桌面\simulink实验\train\loss\sac_idp_v3_3seq_5a_500b_Kunder_5dis_train20250911-084323.csv")
df.columns = df.columns.str.strip()

# 原始数据
x = df['step']
ep_rd1 = df['ep_rd1']
ep_rdA1 = df['ep_rdA1']
ep_rdF1 = df['ep_rdF1']
ep_rd2 = df['ep_rd2']
ep_rdA2 = df['ep_rdA2']
ep_rdF2 = df['ep_rdF2']
y_loss = ep_rd1 + ep_rd2
y_lossA = ep_rdA1 + ep_rdA2
y_lossF = ep_rdF1 + ep_rdF2


# 创建更密集的 step 值用于插值
x_dense = np.linspace(x.min(), x.max(), 1000)

# 插值函数（可选 'linear', 'quadratic', 'cubic'）
f_loss = interp1d(x, y_loss, kind='cubic')
f_lossA = interp1d(x, y_lossA, kind='cubic')
f_lossF = interp1d(x, y_lossF, kind='cubic')

# 插值后的值
y_dense_loss = f_loss(x_dense)
y_dense_lossA = f_lossA(x_dense)
y_dense_lossF = f_lossF(x_dense)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_dense_loss, label='Reward', color='#1f77b4', linewidth=2)
plt.plot(x_dense, y_dense_lossA, label='Reward_ΔPtie', color='#2ca03b', linewidth=2)
plt.plot(x_dense, y_dense_lossF, label='Reward_Δf', color='#ffdf0e', linewidth=2)

plt.xlabel('Training episode')
plt.ylabel('Reward')
plt.title('Total Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("D:\桌面\simulink实验\\fig\M10B39\\train_5a_500b_5dis_loss.svg", format='svg', dpi=300, bbox_inches='tight')
plt.show()


