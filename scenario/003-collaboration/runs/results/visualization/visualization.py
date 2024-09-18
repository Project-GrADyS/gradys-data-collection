import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = [
    (2, 12, pd.read_csv("./data/run-results__a_2-s_12__1__1723034080-tag-eval_avg_reward.csv")),
    (2, 24, pd.read_csv("./data/run-results__a_2-s_24__1__1723034081-tag-eval_avg_reward.csv")),
    (2, 36, pd.read_csv("./data/run-results__a_2-s_36__1__1723034081-tag-eval_avg_reward.csv")),
    (4, 12, pd.read_csv("./data/run-results__a_4-s_12__1__1723034081-tag-eval_avg_reward.csv")),
    (4, 24, pd.read_csv("./data/run-results__a_4-s_24__1__1723034081-tag-eval_avg_reward.csv")),
    (4, 36, pd.read_csv("./data/run-results__a_4-s_36__1__1723034081-tag-eval_avg_reward.csv")),
    (8, 12, pd.read_csv("./data/run-results__a_8-s_12__1__1723034081-tag-eval_avg_reward.csv")),
    (8, 24, pd.read_csv("./data/run-results__a_8-s_24__1__1723034081-tag-eval_avg_reward.csv")),
    (8, 36, pd.read_csv("./data/run-results__a_8-s_36__1__1723034081-tag-eval_avg_reward.csv")),
]

for num_agents, num_sensors, df in results:
    df['Number of Agents'] = num_agents
    df['Number of Sensors'] = num_sensors
    df['Experiment'] = f"{num_agents} agents, {num_sensors} sensors"
    df['Reward'] = df['Value']

result_df = pd.concat([result[2] for result in results])
sns.set_theme()
sns.set_style("whitegrid")
ax = sns.lineplot(result_df, x='Step', y='Reward', hue='Experiment')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("rewards.png")