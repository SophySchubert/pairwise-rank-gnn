import re
import pandas as pd
import matplotlib.pyplot as plt

data_path1 = './experiment/fc1t/dataframe.pkl'
data_path2 = './experiment/fc2t/dataframe.pkl'
data_path3 = './experiment/fc3t/dataframe.pkl'

df_mean1 = pd.read_pickle('./fcf_mean.pkl')
df_mean2 = pd.read_pickle('./fct_mean.pkl')

df1 = pd.read_pickle(data_path1)
df2 = pd.read_pickle(data_path2)
df3 = pd.read_pickle(data_path3)

df_mean = pd.concat([df1, df2, df3])
df_mean = df_mean.groupby(level=0).mean()
df_mean.to_pickle('fct_mean.pkl')

fig, ax1 = plt.subplots(figsize=(6,4))

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss", color="tab:red")
ax1.plot(df_mean.index, df_mean['Valid Error'], label='Valid Loss (Mean)', color='tab:red', linestyle='solid', alpha=0.8)
# ax1.plot(df_mean.index, df_mean1['Valid Error'], label='Valid Loss (Mean)', color='tab:red', linestyle='solid', alpha=0.8)
# ax1.plot(df_mean.index, df_mean2['Valid Error'], label='Valid Loss (Mean bidirectional)', color='tab:orange', linestyle='solid', alpha=0.8)
ax1.tick_params(axis='y', labelcolor="tab:red", color="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color="tab:blue")
ax2.plot(df_mean.index, df_mean['Valid Acc'], label='Valid Acc (Mean)', color='tab:blue', linestyle='solid', alpha=0.8)
# ax2.plot(df_mean.index, df_mean1['Valid Acc'], label='Valid Acc (Mean)', color='tab:blue', linestyle='solid', alpha=0.8)
# ax2.plot(df_mean.index, df_mean2['Valid Acc'], label='Valid Acc (Mean bidirectional)', color='tab:green', linestyle='solid', alpha=0.8)
ax2.tick_params(axis='y', labelcolor="tab:blue", color="tab:blue")

# Legende und Layout
fig.tight_layout()
fig.legend(loc='center', bbox_transform=ax1.transAxes)
plt.savefig('fc_both.pdf', format='pdf')
plt.close()


