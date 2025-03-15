import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [
    (2, "better results_12 to 36 sensors 2 agents-1728932364.4491496.csv"),
    (4, "better results_12 to 36 sensors 4 agents-1728987793.6354911.csv"),
    (8, "better results_12 to 36 sensors 8 agents-1729168237.176651.csv")
]

all_dfs = []
for agents, file in data:
    df = pd.read_csv(file)
    df["Number of Agents"] = agents
    all_dfs.append(df)

training_data = pd.concat(all_dfs)

training_data = training_data[training_data['Step'] <= 3_000_000]

sns.set_theme()
sns.set_style("whitegrid")
ax = sns.lineplot(data=training_data, x='Step', y='Value', hue='Number of Agents', palette='tab10')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set(xlabel='Step', ylabel='Value')
plt.tight_layout()
plt.savefig("unstable.png")







