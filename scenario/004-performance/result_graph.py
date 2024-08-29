import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.DataFrame.from_records([
    {
        'Solution': 'MADDPG',
        'Number of Agents': 2,
        'Success': 1
    },
    {
        'Solution': 'MADDPG',
        'Number of Agents': 4,
        'Success': 1
    },
    {
        'Solution': 'MADDPG',
        'Number of Agents': 6,
        'Success': 1
    },
    {
        'Solution': 'Greedy Heuristic',
        'Number of Agents': 2,
        'Success': 1
    },
        {
        'Solution': 'Greedy Heuristic',
        'Number of Agents': 4,
        'Success': 1
    },
        {
        'Solution': 'Greedy Heuristic',
        'Number of Agents': 6,
        'Success': 1
    }
])

sns.barplot(x='Number of Agents', y='Success', hue='Solution', data=results, gap=0.1)

# Set x axis label
plt.xlabel("Number of Agents")

# Set y axis label
plt.ylabel("Success rate")

# Format y axis as percentage
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

# Place the legend outside the plot, on top of plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Expand the plot to fit the legend
plt.tight_layout()

# Show the plot
plt.show()


# Scenario 2
# 2 drone
# MADDPG - 15.81s
# Heuristic - 20.33s
#
# 4 drone
# MADDPG - 13.42s
# Heuristic - 19.63s
#
# 6 drone
# MADDPG - 12.21
# Heuristic - 19.46s

results = pd.DataFrame.from_records([
    {
        'Solution': 'MADDPG',
        'Number of Agents': 2,
        'Average Collection Time': 15.81
    },
    {
        'Solution': 'MADDPG',
        'Number of Agents': 4,
        'Average Collection Time': 13.42
    },
    {
        'Solution': 'MADDPG',
        'Number of Agents': 6,
        'Average Collection Time': 12.21
    },
    {
        'Solution': 'Greedy Heuristic',
        'Number of Agents': 2,
        'Average Collection Time': 20.33
    },
        {
        'Solution': 'Greedy Heuristic',
        'Number of Agents': 4,
        'Average Collection Time': 19.63
    },
        {
        'Solution': 'Greedy Heuristic',
        'Number of Agents': 6,
        'Average Collection Time': 19.46
    }
])

ax = sns.barplot(x='Number of Agents', y='Average Collection Time', hue='Solution', data=results, gap=0.1)

# Set x axis label
plt.xlabel("Number of Agents")

# Set y axis label
plt.ylabel("Average Collection Time (s)")

# Show numbers on top of the bars
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])

# Place the legend outside the plot, on top of plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Expand the plot to fit the legend
plt.tight_layout()

# Show the plot
plt.show()