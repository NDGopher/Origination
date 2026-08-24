# How to add a new context adjustment
#
# 1. Implement `ContextAdjustment` in
#    `src/origination/features/context_adjustments.py`
# 2. Register the class in `ADJUSTMENT_REGISTRY`
# 3. Enable under `features.context_adjustments.<name>.enabled: true`
# 4. Ensure `apply()` uses only pre-kickoff information
# 5. Add a unit test for enable/disable + no leakage
# 6. Re-run walk-forward; promote only if market log-loss / CLV improve
#
# See module docstring in context_adjustments.py for full contract.
#
# Currently LIVE (non-scaffold):
#   referee — lagged cards/fouls + asymmetric card_bias_coef
#     (docs/REFEREE_CONTEXT.md)
#   coaching_change — appointment CSV + new-coach bounce
#     (docs/COACHING_CONTEXT.md; data/interim/coaching_changes.csv)
