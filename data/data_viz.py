import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

df_pushblock_drawer = pd.read_csv("progress_pushblockdrawer.csv")
df_pickplace_drawer = pd.read_csv("progress_pickplace_drawer.csv")
df_pickplace_table = pd.read_csv("progress_pickplacetable.csv")
df_drawer = pd.read_csv("progress_drawer.csv")
df_drawer_nolayer = pd.read_csv("progress_drawer_nolayer.csv")
df_pickplace_drawer_entropy = pd.read_csv("progress_pickplace_drawer_entropy0.1.csv")
df_pickplace_drawer_0_25 = pd.read_csv("progress_pickplace_0.25.csv")
df_drawer_unstable = pd.read_csv("progress_drawer_unstable.csv")
df_pickplace_drawer_unstable = pd.read_csv("progress_pickplace_drawer_unstable.csv")
df_pickplace_table_unstable = pd.read_csv("progress_pickplace_table_unstable.csv")
df_pushblock_drawer_unstable = pd.read_csv("progress_pushblockdrawer_unstable.csv")



pushblock_drawer_mean = df_pushblock_drawer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_drawer_mean = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_table_mean = df_pickplace_table['eval/state_desired_goal/final/overall_success Mean'] 
drawer_mean = df_drawer['eval/state_desired_goal/final/overall_success Mean'] 
drawer_nolayer_mean = df_drawer_nolayer['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_drawer_entropy_mean = df_pickplace_drawer_entropy['eval/state_desired_goal/final/overall_success Mean']
pickplace_drawer_0_25_mean = df_pickplace_drawer_0_25['eval/state_desired_goal/final/overall_success Mean']

drawer_unstable_mean = df_drawer_unstable['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_drawer_unstable_mean = df_pickplace_drawer_unstable['eval/state_desired_goal/final/overall_success Mean'] 
pickplace_table_unstable_mean = df_pickplace_table_unstable['eval/state_desired_goal/final/overall_success Mean'] 
pushblock_drawer_unstable_mean = df_pushblock_drawer_unstable['eval/state_desired_goal/final/overall_success Mean'] 


drawer_actor_loss = df_drawer['trainer/eval/Policy Loss'] 
drawer_unstable_actor_loss = df_drawer_unstable['trainer/eval/Policy Loss'] 

pushblock_drawer_actor_loss = df_pushblock_drawer['trainer/eval/Policy Loss']
pushblock_drawer_unstable_actor_loss = df_pushblock_drawer_unstable['trainer/eval/Policy Loss']

plt.figure(figsize=(6, 4))
plt.plot(pushblock_drawer_actor_loss, label = "Stable Contrastive RL")
plt.plot(pushblock_drawer_unstable_actor_loss, label = "Contrastive RL")
plt.title("Policy Loss over Epochs for Push Block, Open Drawer Environment")
plt.xlabel("Epochs")
plt.ylabel("Policy Loss")
plt.xlim(0,200)
plt.legend()
plt.show()

# pushblock_drawer_mean_overall = df_pushblock_drawer['expl/Returns Mean'] 
# pickplace_drawer_mean_overall = df_pickplace_drawer['eval/state_desired_goal/overall_success Mean'] 
# pickplace_table_mean_overall = df_pickplace_table['expl/Returns Mean'] 
# drawer_mean_overall = df_drawer['eval/state_desired_goal/overall_success Mean'] 

# pushblock_drawer_std = df_pushblock_drawer['eval/state_desired_goal/final/overall_success Std'] 
# pickplace_drawer_std = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Std'] 
# pickplace_table_std = df_pickplace_table['eval/state_desired_goal/final/overall_success Std'] 
# drawer_std = df_drawer['eval/state_desired_goal/final/overall_success Std'] 
# drawer_nolayer_std = df_drawer_nolayer['eval/state_desired_goal/final/overall_success Std'] 

# pushblock_drawer_std_overall = df_pushblock_drawer['expl/Returns Std'] 
# pickplace_drawer_std_overall = df_pickplace_drawer['eval/state_desired_goal/final/overall_success Std'] 
# pickplace_table_std_overall = df_pickplace_table['expl/Returns Std'] 
# drawer_std_overall = df_drawer['eval/state_desired_goal/final/overall_success Std'] 

# drawer_nolayer_mean_smoothed = gaussian_filter1d(drawer_nolayer_mean, 3)
# drawer_nolayer_std_smoothed = gaussian_filter1d(drawer_nolayer_std, 3)

# grad_steps_per_epoch = 2000
# new_indices = np.linspace(0, len(drawer_nolayer_mean_smoothed) - 1, len(drawer_nolayer_mean_smoothed) * grad_steps_per_epoch)
# drawer_nolayer_mean_smoothed_extended = np.interp(new_indices, np.arange(len(drawer_nolayer_mean_smoothed)), drawer_nolayer_mean_smoothed)
# drawer_nolayer_std_smoothed_extended = np.interp(new_indices, np.arange(len(drawer_nolayer_std_smoothed)), drawer_nolayer_std_smoothed)

# x = np.linspace(0, len(drawer_nolayer_mean_smoothed) * grad_steps_per_epoch, len(drawer_nolayer_mean_smoothed) * grad_steps_per_epoch)

# plt.figure(figsize=(6, 4))
# plt.plot(x, drawer_nolayer_mean_smoothed_extended, label = 'push block, open drawer')
# plt.fill_between(x,
#                  drawer_nolayer_mean_smoothed_extended - drawer_nolayer_std_smoothed_extended,
#                  drawer_nolayer_mean_smoothed_extended + drawer_nolayer_std_smoothed_extended,
#                  alpha=0.2)
# plt.xlabel('Gradient Steps')
# plt.ylabel('Evaluation Success Rate')
# plt.title('Drawer (No Layer)')
# plt.tight_layout()
# plt.show()

# x = np.linspace(0, len(pickplace_drawer_mean), len(pickplace_drawer_mean))
# plt.figure(figsize=(6, 4))
# plt.plot(x, pickplace_drawer_mean, label = 'push block, open drawer')
# plt.fill_between(x,
#                  pickplace_drawer_mean - pickplace_drawer_std,
#                  pickplace_drawer_mean + pickplace_drawer_std,
#                  alpha=0.2)
# plt.xlabel('Epochs')
# plt.ylabel('Evaluation Success Rate')
# plt.title('Pick & Place (Drawer)')
# plt.tight_layout()
# plt.show()
# x = np.linspace(0, len(drawer_mean), len(drawer_mean))
# plt.figure(figsize=(6, 4))
# plt.plot(x, drawer_mean, label = 'push block, open drawer')
# plt.fill_between(x,
#                  drawer_mean - drawer_std,
#                  drawer_mean + drawer_std,
#                  alpha=0.2)
# plt.xlabel('Epochs')
# plt.ylabel('Evaluation Success Rate')
# plt.title('Drawer')
# plt.tight_layout()
# plt.show()