"""
ReasonGraph-ABSA hyperparameter sensitivity figures
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 1.2


# ============================================================
# Figure: GATv2 layer count + LoRA rank sensitivity (combined)
# ============================================================
def plot_hyperparameter_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- GATv2 Layers ---
    layers = [0, 1, 2, 3, 4]
    layer_labels = ['0\n(no graph)', '1', '2', '3', '4']
    f1_layers = [79.10, 79.68, 80.22, 79.85, 79.41]

    bars = ax1.bar(layers, f1_layers, color=['#95a5a6'] + ['#3498db']*4,
                   edgecolor='white', width=0.6)
    bars[2].set_color('#e67e22')  # highlight best
    bars[2].set_edgecolor('#d35400')
    bars[2].set_linewidth(2)

    for i, v in enumerate(f1_layers):
        ax1.text(i, v + 0.15, f'{v:.2f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold' if i == 2 else 'normal')

    ax1.set_xlabel('Number of GATv2 Layers', fontsize=12)
    ax1.set_ylabel('F1 (%)', fontsize=12)
    ax1.set_title('(a) GATv2 Layer Depth', fontsize=13, fontweight='bold')
    ax1.set_xticks(layers)
    ax1.set_xticklabels(layer_labels, fontsize=10)
    ax1.set_ylim(78, 81)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- LoRA Rank ---
    ranks = [0, 1, 2, 3]
    rank_labels = ['16', '32', '64', '128']
    f1_ranks = [78.53, 79.41, 80.22, 80.08]

    bars2 = ax2.bar(ranks, f1_ranks, color=['#3498db']*4,
                    edgecolor='white', width=0.6)
    bars2[2].set_color('#e67e22')
    bars2[2].set_edgecolor('#d35400')
    bars2[2].set_linewidth(2)

    for i, v in enumerate(f1_ranks):
        ax2.text(i, v + 0.15, f'{v:.2f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold' if i == 2 else 'normal')

    ax2.set_xlabel('LoRA Rank', fontsize=12)
    ax2.set_ylabel('F1 (%)', fontsize=12)
    ax2.set_title('(b) LoRA Rank', fontsize=13, fontweight='bold')
    ax2.set_xticks(ranks)
    ax2.set_xticklabels(rank_labels, fontsize=10)
    ax2.set_ylim(77.5, 81)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/hyperparameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print('Saved: figures/hyperparameter_sensitivity.png')


# ============================================================
# Figure: Stage2 epoch sensitivity (line chart)
# ============================================================
def plot_stage2_epochs():
    epochs = [1, 2, 3, 5, 7, 10]
    f1_scores = [78.86, 79.55, 79.92, 80.22, 80.15, 79.78]
    precision = [80.12, 79.98, 79.75, 79.38, 78.90, 78.25]
    recall =    [77.64, 79.13, 80.09, 81.09, 81.44, 81.35]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, f1_scores, 'o-', color='#e67e22', linewidth=2.5, markersize=8,
            label='F1', zorder=5)
    ax.plot(epochs, precision, 's--', color='#3498db', linewidth=1.5, markersize=6,
            alpha=0.7, label='Precision')
    ax.plot(epochs, recall, '^--', color='#2ecc71', linewidth=1.5, markersize=6,
            alpha=0.7, label='Recall')

    # Highlight best F1
    best_idx = f1_scores.index(max(f1_scores))
    ax.annotate(f'Best: {f1_scores[best_idx]:.2f}',
               xy=(epochs[best_idx], f1_scores[best_idx]),
               xytext=(epochs[best_idx]+1.2, f1_scores[best_idx]+0.5),
               fontsize=10, fontweight='bold', color='#d35400',
               arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.5))

    ax.set_xlabel('Stage 2 Training Epochs', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Impact of Stage 2 Training Duration (Rest14, Qwen3-8B)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(epochs)
    ax.set_ylim(77, 83)
    ax.legend(fontsize=11, frameon=False, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/stage2_epochs.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/stage2_epochs.png', dpi=300, bbox_inches='tight')
    print('Saved: figures/stage2_epochs.png')


# ============================================================
if __name__ == '__main__':
    plot_hyperparameter_sensitivity()
    plot_stage2_epochs()
    print('\nAll sensitivity figures saved!')
