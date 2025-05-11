import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

df_pushblock_drawer = pd.read_csv("progress_pushblockdrawer.csv")
df_pickplace_drawer = pd.read_csv("progress_pickplace_drawer.csv")
df_pickplace_table = pd.read_csv("progress_pickplacetable.csv")
df_drawer = pd.read_csv("progress_drawer.csv")
df_drawer_nolayer = pd.read_csv("progress_drawer_nolayer.csv")
df_drawer_noaug = pd.read_csv("progress_noaug_2.csv")
df_drawer_noinit = pd.read_csv("progress_drawer_noinit.csv")
df_drawer_512_2 = pd.read_csv("progress_drawer_512_2.csv")
df_drawer_1024_2 = pd.read_csv("progress_1024_2.csv")

df_drawer_unstable = pd.read_csv("progress_drawer_unstable.csv")
df_pickplace_drawer_unstable = pd.read_csv("progress_pickplace_drawer_unstable.csv")
df_pickplace_table_unstable = pd.read_csv("progress_pickplace_table_unstable.csv")
df_pushblock_drawer_unstable = pd.read_csv("progress_pushblockdrawer_unstable.csv")


pushblock_drawer_mean = df_pushblock_drawer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_drawer_mean = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_table_mean = df_pickplace_table['eval/state_desired_goal/final/overall_success Mean'] 
drawer_mean = df_drawer['eval/state_desired_goal/final/overall_success Mean'] 
drawer_nolayer_mean = df_drawer_nolayer['eval/state_desired_goal/final/overall_success Mean'] 
drawer_noaug_mean = df_drawer_noaug['eval/state_desired_goal/final/overall_success Mean'] 
df_drawer_noinit_mean = df_drawer_noinit['eval/state_desired_goal/final/overall_success Mean'] 
df_drawer_512_2_mean = df_drawer_512_2['eval/state_desired_goal/final/overall_success Mean'] 
df_drawer_1024_2_mean = df_drawer_1024_2['eval/state_desired_goal/final/overall_success Mean'] 

drawer_unstable_mean = df_drawer_unstable['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_drawer_unstable_mean = df_pickplace_drawer_unstable['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_table_unstable_mean = df_pickplace_table_unstable['eval/state_desired_goal/final/overall_success Mean'] 
pushblock_drawer_unstable_mean = df_pushblock_drawer_unstable['eval/state_desired_goal/final/overall_success Mean'] 



# print(np.mean(drawer_mean))
# print(np.mean(drawer_nolayer_mean))
# print(np.mean(drawer_noaug_mean))
# print(np.mean(df_drawer_noinit_mean))
# print(drawer_mean.iloc[50])
# print(drawer_nolayer_mean.iloc[45])
# print(drawer_noaug_mean.iloc[45])

pushblock_drawer_std = df_pushblock_drawer['eval/state_desired_goal/final/overall_success Std'] 
pickplace_drawer_std = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Std'] 
pickplace_table_std = df_pickplace_table['eval/state_desired_goal/final/overall_success Std'] 
drawer_std = df_drawer['eval/state_desired_goal/final/overall_success Std'] 
drawer_nolayer_std = df_drawer_nolayer['eval/state_desired_goal/final/overall_success Std'] 
drawer_noaug_std = df_drawer_noaug['eval/state_desired_goal/final/overall_success Std'] 
drawer_noinit_std = df_drawer_noinit['eval/state_desired_goal/final/overall_success Std'] 
df_drawer_512_2_std = df_drawer_512_2['eval/state_desired_goal/final/overall_success Std'] 
df_drawer_1024_2_std = df_drawer_1024_2['eval/state_desired_goal/final/overall_success Std'] 

drawer_unstable_std = df_drawer_unstable['eval/state_desired_goal/final/overall_success Std'] 
pickplace_drawer_unstable_std = df_pickplace_drawer_unstable['eval/state_desired_goal/final/overall_success Std'] 
pickplace_table_unstable_std = df_pickplace_table_unstable['eval/state_desired_goal/final/overall_success Std'] 
pushblock_drawer_unstable_std = df_pushblock_drawer_unstable['eval/state_desired_goal/final/overall_success Std'] 

labels = ['drawer', 'pick & place \n (table)', 'pick & place \n (drawer)', 'push block \n open drawer']
x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

stable_means = [np.mean(drawer_mean), np.mean(pickplace_table_mean), np.mean(pickplace_drawer_mean), np.mean(pushblock_drawer_mean)]
unstable_means = [np.mean(drawer_unstable_mean), np.mean(pickplace_table_unstable_mean), np.mean(pickplace_drawer_unstable_mean), np.mean(pushblock_drawer_unstable_mean)]

stable_stds = [np.mean(drawer_std), np.mean(pickplace_table_std), np.mean(pickplace_drawer_std), np.mean(pushblock_drawer_std)]
unstable_stds = [np.mean(drawer_unstable_std), np.mean(pickplace_table_unstable_std), np.mean(pickplace_drawer_unstable_std), np.mean(pushblock_drawer_unstable_std)]


fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, stable_means, width, yerr=stable_stds, label='Stable Contrastive RL', capsize=5)
bars2 = ax.bar(x + width/2, unstable_means, width, yerr=unstable_stds, label='Contrastive RL', capsize=5)

# Labels and layout
ax.set_ylabel('Success Rate')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()

plt.show()


# categories = ["3-layer CNN + (1024, 4) MLP", "3-layer CNN + (1024, 2) MLP", "3-layer CNN + (512, 2) MLP"]
# values = [np.mean(drawer_mean), np.mean(df_drawer_1024_2_mean), np.mean(df_drawer_512_2_mean)]
# stds = [np.mean(drawer_std), np.mean(df_drawer_1024_2_std), np.mean(df_drawer_512_2_std)]
# colors = ['#1f77b4', '#a6c8e0', '#a6c8e0']


# print(np.mean(drawer_std))
# print(np.mean(drawer_nolayer_std))
# print(np.mean(drawer_noaug_std))
# print(np.mean(drawer_noinit_std))
# print(drawer_std.iloc[50])
# print(drawer_nolayer_std.iloc[45])
# print(drawer_noaug_std.iloc[45])


# categories = ["Stable Contrastive RL", "w/o cold initialization", "w/o layer normalization", "w/o data augmentation"]
# values = [np.mean(drawer_mean), np.mean(df_drawer_noinit_mean), np.mean(drawer_nolayer_mean.iloc), np.mean(drawer_noaug_mean.iloc)]
# stds = [np.mean(drawer_std), np.mean(drawer_noinit_std), np.mean(drawer_nolayer_std.iloc), np.mean(drawer_noaug_std.iloc)]

# categories = ["3-layer CNN + (1024, 4) MLP", "3-layer CNN + (1024, 2) MLP", "3-layer CNN + (512, 2) MLP"]
# values = [np.mean(drawer_mean), np.mean(df_drawer_1024_2_mean), np.mean(df_drawer_512_2_mean)]
# stds = [np.mean(drawer_std), np.mean(df_drawer_1024_2_std), np.mean(df_drawer_512_2_std)]
# colors = ['#1f77b4', '#a6c8e0', '#a6c8e0']


# plt.figure(figsize=(12,6))
# plt.bar(categories, values,  yerr = stds, color = colors, capsize = 5)

# plt.ylabel('Success Rate')
# plt.title('Network Architecture')

# # Show plot
# plt.show()
