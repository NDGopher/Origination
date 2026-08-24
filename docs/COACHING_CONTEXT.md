# Coaching-change context
#
# Source: data/interim/coaching_changes.csv
#   columns: team, change_date[, notes]
#   Team names canonicalized via TeamNameMapper.
#   File is explicit and reviewable — extend when adding seasons.
#
# Features (pre-match only):
#   coach_days_in_charge_{home,away}
#   coach_games_in_charge_{home,away}
#   new_coach_{home,away} = 1 if days <= new_coach_days (60) OR games <= new_coach_games (8)
#
# Optional intensity:
#   bounce_coef: λ_side *= exp(bounce_coef) when new_coach_side=1
#   Positive = new-manager bounce; negative = disruption. Default 0 until measured.
#
# Implementation: origination.features.context_adjustments.CoachingChangeAdjustment
# Enable: features.context_adjustments.coaching_change.enabled: true
