import matplotlib.pyplot as plt

# Data extracted from image_5803e4.png
# Exact search is isolated as a baseline, leaving nprobe as the tunable X-axis
nprobes = ['1', '2', '5', '10', '20']
qps = [929.63, 260.34, 81.87, 43.81, 27.51]
recall = [94.0, 98.4, 100.0, 100.0, 100.0]

exact_qps = 16.51
exact_recall = 100.0

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot QPS (Speed) curve on the primary y-axis
color1 = 'tab:blue'
ax1.set_xlabel('nprobe (Clusters Searched)', fontweight='bold')
ax1.set_ylabel('Queries Per Second (QPS)', color=color1, fontweight='bold')
line1 = ax1.plot(nprobes, qps, marker='o', color=color1, linewidth=2, label='IVF Speed (QPS)')

# Add Exact baseline for QPS
line2 = ax1.axhline(y=exact_qps, color=color1, linestyle='--', alpha=0.7, label=f'Exact Baseline ({exact_qps} QPS)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Recall curve on the secondary y-axis
ax2 = ax1.twinx()  
color2 = 'tab:orange'
ax2.set_ylabel('Recall Accuracy (%)', color=color2, fontweight='bold')  
line3 = ax2.plot(nprobes, recall, marker='s', color=color2, linewidth=2, label='IVF Accuracy (%)')

# Add Exact baseline for Recall
line4 = ax2.axhline(y=exact_recall, color=color2, linestyle='--', alpha=0.7, label='Exact Baseline (100% Recall)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(90, 101)

# Manually combine legends from both axes
lines = line1 + [line2] + line3 + [line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Vector Database: Speed vs. Accuracy Tradeoff', fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('benchmark_results.png', dpi=300)
plt.show()