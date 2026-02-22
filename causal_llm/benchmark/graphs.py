"""Definitions of five canonical causal graph structures.

Each structure is a fundamental topology from causal inference theory,
used to test distinct aspects of causal reasoning.
"""

CAUSAL_GRAPHS = {
    "chain": {
        "edges": [("X", "Y"), ("Y", "Z")],
        "description": "Chain (Mediation)",
        "variables": ["X", "Y", "Z"],
        "mechanism": "X causes Z only through mediator Y.",
    },
    "fork": {
        "edges": [("C", "X"), ("C", "Z")],
        "description": "Fork (Common Cause)",
        "variables": ["C", "X", "Z"],
        "mechanism": (
            "C is a common cause (confounder) of X and Z. "
            "X and Z are correlated but neither causes the other."
        ),
    },
    "collider": {
        "edges": [("X", "M"), ("Z", "M")],
        "description": "Collider (Explaining Away)",
        "variables": ["X", "Z", "M"],
        "mechanism": (
            "X and Z are independent causes of M. "
            "Conditioning on M induces spurious dependence between X and Z."
        ),
    },
    "diamond": {
        "edges": [("X", "Y"), ("X", "Z"), ("Y", "W"), ("Z", "W")],
        "description": "Diamond (Multiple Pathways)",
        "variables": ["X", "Y", "Z", "W"],
        "mechanism": "X influences W through two parallel mediators Y and Z.",
    },
    "instrument": {
        "edges": [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")],
        "description": "Instrumental Variable",
        "variables": ["Z", "X", "Y", "U"],
        "mechanism": (
            "Z is an instrument for the X->Y effect. "
            "U is an unobserved confounder between X and Y."
        ),
    },
}

LAYOUT_OVERRIDES = {
    "chain": {"X": (0, 0), "Y": (1, 0), "Z": (2, 0)},
    "fork": {"C": (1, 1), "X": (0, 0), "Z": (2, 0)},
    "collider": {"X": (0, 1), "Z": (2, 1), "M": (1, 0)},
    "diamond": {"X": (0, 1), "Y": (1, 2), "Z": (1, 0), "W": (2, 1)},
    "instrument": {"Z": (0, 1), "X": (1, 1), "Y": (2, 1), "U": (1.5, 2)},
}
