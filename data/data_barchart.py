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

pushblock_drawer_mean = df_pushblock_drawer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_drawer_mean = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_table_mean = df_pickplace_table['eval/state_desired_goal/final/overall_success Mean'] 
drawer_mean = df_drawer['eval/state_desired_goal/final/overall_success Mean'] 
drawer_nolayer_mean = df_drawer_nolayer['eval/state_desired_goal/final/overall_success Mean'] 
drawer_noaug_mean = df_drawer_noaug['eval/state_desired_goal/final/overall_success Mean'] 

print(np.mean(drawer_mean))
print(np.mean(drawer_nolayer_mean))
print(np.mean(drawer_noaug_mean))
# print(drawer_mean.iloc[50])
# print(drawer_nolayer_mean.iloc[45])
# print(drawer_noaug_mean.iloc[45])

pushblock_drawer_std = df_pushblock_drawer['eval/state_desired_goal/final/overall_success Std'] 
pickplace_drawer_std = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Std'] 
pickplace_table_std = df_pickplace_table['eval/state_desired_goal/final/overall_success Std'] 
drawer_std = df_drawer['eval/state_desired_goal/final/overall_success Std'] 
drawer_nolayer_std = df_drawer_nolayer['eval/state_desired_goal/final/overall_success Std'] 
drawer_noaug_std = df_drawer_noaug['eval/state_desired_goal/final/overall_success Std'] 

print(np.mean(drawer_std))
print(np.mean(drawer_nolayer_std))
print(np.mean(drawer_noaug_std))
# print(drawer_std.iloc[50])
# print(drawer_nolayer_std.iloc[45])
# print(drawer_noaug_std.iloc[45])


categories = ["Stable Contrastive RL", "w/o layer normalization", "w/o data augmentation"]
values = [np.mean(drawer_mean), np.mean(drawer_nolayer_mean), np.mean(drawer_noaug_mean)]
stds = [np.mean(drawer_std), np.mean(drawer_nolayer_std), np.mean(drawer_noaug_std)]
colors = ['#1f77b4', '#a6c8e0', '#a6c8e0']


plt.figure(figsize=(20,6))
plt.bar(categories, values,  yerr = stds, color = colors, capsize = 5)

plt.ylabel('Success Rate')
plt.title('Stabilization and Initialization')

# Show plot
plt.show()
