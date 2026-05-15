import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the log file
df = pd.read_csv('training_log_e2e.csv')

# Set plotting style
sns.set(style="darkgrid")
plt.rcParams.update({'font.size': 12})

def plot_metric(metric, color, title, ylabel, filename, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df[metric], marker='o', linewidth=2.5, markersize=6, color=color)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# 1. Plot SRCC
plot_metric('val_mos_srcc', 'royalblue', 'Validation MOS SRCC vs Epoch', 'Spearman Rank Correlation Coefficient', '/home/user3/.gemini/antigravity/brain/7c3563ce-f275-4408-bf62-b4bb5472bef5/val_mos_srcc.png', ylim=(0, 1))

# 2. Plot PLCC
plot_metric('val_mos_plcc', 'darkorange', 'Validation MOS PLCC vs Epoch', 'Pearson Linear Correlation Coefficient', '/home/user3/.gemini/antigravity/brain/7c3563ce-f275-4408-bf62-b4bb5472bef5/val_mos_plcc.png', ylim=(0, 1))

# 3. Plot RMSE
plot_metric('val_mos_rmse', 'crimson', 'Validation MOS RMSE vs Epoch', 'Root Mean Square Error', '/home/user3/.gemini/antigravity/brain/7c3563ce-f275-4408-bf62-b4bb5472bef5/val_mos_rmse.png')

print("All 3 plots generated successfully.")
