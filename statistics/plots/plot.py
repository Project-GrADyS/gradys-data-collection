# Métricas do non-collaborative
# Total successful runs: 4434
# Total runs: 4500
# Success rate: 98.53%
# Average completion time: 140.16s
# Training time: 7.78h
# Total training time: 7.78h

# Métricas do collaborative:
# Total successful runs: 4494
# Total runs: 4500
# Success rate: 99.87%
# Average completion time: 17.42s
# Training time: (2, 12) - 13.75h | (2,24) - 69.05h | (2, 36) - 121.30h | (4,12) - 21.42h | (4,24) - 46.29h | (4,36) - 37.71h | (8, 12) - 22.69h | (8, 24) - 111.54h | (8, 36) - 199.03h
# Total training time: 642.78h
#
# Métricas do flexible sensor:
# Total successful runs: 4497
# Total runs: 4500
# Success rate: 99.93%
# Average completion time: 17.75s
# Training time: 2 - 24.14h | 4 - 9.65h | 8 - 90.64h
# Total training time: 124.43h
#
# Métricas do flexible agent:
# Total successful runs: 4464
# Total runs: 4500
# Success rate: 99.20%
# Average completion time: 26.65s
# Training time: 2 - 28,56h
# Total training time: 28.56h

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dados extraídos dos comentários
algorithms = [
    'Non-collaborative',
    'Collaborative',
    'Flexible Sensor',
    'Flexible Agent'
]
success_rate = [98.53, 99.87, 99.93, 99.20]
avg_completion_time = [140.16, 17.42, 17.75, 26.65]
total_training_time = [7.78, 642.78, 124.43, 28.56]

# Configuração do tema
sns.set_theme(style="whitegrid", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

# Plot 1: Success Rate
ax1 = sns.barplot(x=algorithms, y=success_rate, palette="crest", )
for i, container in enumerate(ax1.containers):
    ax1.bar_label(container, labels=[f'{success_rate[i]:.2f}%'], fontsize=16)
ax1.set_ylabel('Taxa de Sucesso (%)')
ax1.set_xlabel('Algoritmo')
ax1.set_ylim(95, 100.8)  # Ajuste do limite do eixo y
plt.tight_layout()
plt.savefig('./success_rate_barplot.png')
plt.clf()

# Plot 2: Tempo Médio de Conclusão
ax2 = sns.barplot(x=algorithms, y=avg_completion_time, palette="crest")
for i, container in enumerate(ax2.containers):
    ax2.bar_label(container, labels=[f'{avg_completion_time[i]:.2f}s'], fontsize=16)
ax2.set_ylabel('Tempo Médio de Conclusão (s)')
ax2.set_xlabel('Algoritmo')
plt.tight_layout()
plt.savefig('./completion_time_barplot.png')
plt.clf()

# Plot 3: Tempo Total de Treinamento
ax3 = sns.barplot(x=algorithms, y=total_training_time, palette="crest")
for i, container in enumerate(ax3.containers):
    ax3.bar_label(container, labels=[f'{total_training_time[i]:.2f}h'], fontsize=16)
ax3.set_ylabel('Tempo Total de Treinamento (h)')
ax3.set_xlabel('Algoritmo')
plt.tight_layout()
plt.savefig('./training_time_barplot.png')
plt.clf()