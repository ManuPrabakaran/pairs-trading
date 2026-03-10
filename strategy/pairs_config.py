"""
Canonical list of validated pairs and their trading parameters.

This is the single source of truth for which pairs the strategy trades.
All notebooks and the live signal system import from here.

To add a new pair after validation in a future notebook:
1. Append (t1, t2) to VALIDATED_PAIRS
2. Optionally add a custom entry in PAIR_CONFIGS if the pair uses
   different parameters — otherwise the default (2.0, 0.0) applies.

Parameter meaning:
    entry_z  — z-score threshold to enter a position (validated robust at 2.0–2.5)
    exit_z   — z-score threshold to exit a position (0.0 = exit at the mean)
"""

VALIDATED_PAIRS = [
    ('KO',  'PEP'),   # Beverages       — added notebook 06
    ('NUE', 'STLD'),  # Steel           — added notebook 06
    ('V',   'MA'),    # Payments        — added notebook 06
    ('GS',  'MS'),    # Investment banks — added notebook 06
    ('HD',  'LOW'),   # Home improvement — added notebook 06
    ('TRV', 'CB'),    # Insurance       — added notebook 14 (BH-corrected screening)
]

DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z  = 0.0

# Per-pair (entry_z, exit_z). Falls back to defaults if a pair is not listed.
_CUSTOM_CONFIGS = {}

PAIR_CONFIGS = {
    pair: _CUSTOM_CONFIGS.get(pair, (DEFAULT_ENTRY_Z, DEFAULT_EXIT_Z))
    for pair in VALIDATED_PAIRS
}
