"""Four canonical causal graphs (chain, fork, collider, diamond)."""

import matplotlib.pyplot as plt
import networkx as nx


CAUSAL_GRAPHS = {
    'chain': {
        'edges': [('X', 'Y'), ('Y', 'Z')],
        'description': 'Chain (Mediation)',
        'variables': ['X', 'Y', 'Z'],
        'mechanism': 'X causes Z only through mediator Y.',
    },
    'fork': {
        'edges': [('C', 'X'), ('C', 'Z')],
        'description': 'Fork (Common Cause)',
        'variables': ['C', 'X', 'Z'],
        'mechanism': (
            'C is a common cause (confounder) of X and Z. '
            'X and Z are correlated but neither causes the other.'
        ),
    },
    'collider': {
        'edges': [('X', 'M'), ('Z', 'M')],
        'description': 'Collider (Explaining Away)',
        'variables': ['X', 'Z', 'M'],
        'mechanism': (
            'X and Z are independent causes of M. '
            'Conditioning on M induces spurious dependence between X and Z.'
        ),
    },
    'diamond': {
        'edges': [('X', 'Y'), ('X', 'Z'), ('Y', 'W'), ('Z', 'W')],
        'description': 'Diamond (Multiple Pathways)',
        'variables': ['X', 'Y', 'Z', 'W'],
        'mechanism': 'X influences W through two parallel mediators Y and Z.',
    },
}

# Manual layouts -- spring layout makes these messy.
_LAYOUTS = {
    'chain':    {'X': (0, 0), 'Y': (1, 0), 'Z': (2, 0)},
    'fork':     {'C': (1, 1), 'X': (0, 0), 'Z': (2, 0)},
    'collider': {'X': (0, 1), 'Z': (2, 1), 'M': (1, 0)},
    'diamond':  {'X': (0, 1), 'Y': (1, 2), 'Z': (1, 0), 'W': (2, 1)},
}


def plot_causal_graphs(output_path=None):
    n = len(CAUSAL_GRAPHS)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, CAUSAL_GRAPHS.items()):
        G = nx.DiGraph()
        G.add_edges_from(info['edges'])
        pos = _LAYOUTS[name]
        node_colors = ['#BAE1FF'] * G.number_of_nodes()

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=800, edgecolors='black', linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=13, font_weight='bold')
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#333',
                               arrows=True, arrowsize=20, width=2,
                               connectionstyle='arc3,rad=0.1')
        ax.set_title(info['description'], fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Four Canonical Causal Structures', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
