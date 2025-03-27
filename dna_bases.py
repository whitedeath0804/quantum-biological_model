"""
This module contains DNA base data and utility functions.
"""

BASE_DATA = {
    'A': {'comp': 'T', 'D': 0.20, 'a': 1.5e10, 'x0': 0.5e-10, 'C': 0.15},
    'T': {'comp': 'A', 'D': 0.21, 'a': 1.4e10, 'x0': 0.5e-10, 'C': 0.15},
    'G': {'comp': 'C', 'D': 0.25, 'a': 1.6e10, 'x0': 0.45e-10, 'C': 0.14},
    'C': {'comp': 'G', 'D': 0.24, 'a': 1.55e10, 'x0': 0.47e-10, 'C': 0.15}
}

def get_complement(base):
    """
    Returns the complementary base for a given base (e.g., A -> T).
    """
    return BASE_DATA.get(base, {}).get('comp', None)
