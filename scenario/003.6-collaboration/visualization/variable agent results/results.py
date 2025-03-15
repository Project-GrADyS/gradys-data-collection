import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
all_collected = pd.read_csv("./all_collected_rate.csv")
completion_time = pd.read_csv("./completion_time.csv")

# Plotting all_collected
sns.set_theme()
sns.set_style("whitegrid")
ax = sns.lineplot(data=all_collected, x='Step', y='Value')
ax.set(xlabel='Training step', ylabel='Collected rate (0-1)')
plt.tight_layout()

# Save the plot
plt.savefig("all_collected_rate.png")

plt.clf()
# Plotting completion_time
sns.set_theme()
sns.set_style("whitegrid")
ax = sns.lineplot(data=completion_time, x='Step', y='Value')
ax.set(xlabel='Training step', ylabel='Completion time (s)')
plt.tight_layout()

# Save the plot
plt.savefig("completion_time.png")