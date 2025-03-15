import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df1 = pd.read_csv("./agent scale bench_active_agents False-phantom_agentsFalse__1730610092.csv")
df2 = pd.read_csv("./agent scale bench_active_agents False-phantom_agentsTrue__1730610092.csv")
df3 = pd.read_csv("./agent scale bench_active_agents True-phantom_agentsFalse__1730610092.csv")
df4 = pd.read_csv("./agent scale bench_active_agents True-phantom_agentsTrue__1730610092.csv")

# Identify data
df1['State filling approach'] = 'Zeroes'
df1['Agent count on state'] = 'No'

df2['State filling approach'] = 'Phantom agents'
df2['Agent count on state'] = 'No'

df3['State filling approach'] = 'Zeroes'
df3['Agent count on state'] = 'Yes'

df4['State filling approach'] = 'Phantom agents'
df4['Agent count on state'] = 'Yes'

# Combine the data
df = pd.concat([df3, df4, df1, df2])

df.rename(columns={
    "Step": "Training step",
    "Value": "Average Rewards"
}, inplace=True)

# Plot the data
sns.set_theme()
sns.set_style("whitegrid")
ax = sns.lineplot(data=df, x='Training step', y='Average Rewards', hue='State filling approach', style='Agent count on state')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

# Save the plot
plt.savefig("rewards.png")
