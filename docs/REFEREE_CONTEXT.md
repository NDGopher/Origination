# How referee tendencies are computed
#
# Source: football-data.co.uk columns on aligned matches —
#   referee, home_yellow, away_yellow, home_red, away_red, home_fouls, away_fouls
#
# Pre-match only: for each referee, expanding mean of prior fixtures
# (min_prior_games default 5). Features:
#   ref_cards_avg, ref_fouls_avg, ref_home_card_share, ref_home_card_bias,
#   ref_games_prior, ref_cards_vs_league
#
# Intensity channels (YAML):
#   tempo_coef: equal λ/μ bump from ref_cards_vs_league (O/U; measured ≈null)
#   card_bias_coef: asymmetric 1X2 channel —
#     bias = ref_home_card_share - 0.5
#     λ_home *= exp(-card_bias_coef * bias)
#     λ_away *= exp(+card_bias_coef * bias)
#
# Enable:
#   features.context_adjustments.enabled: true
#   features.context_adjustments.referee.enabled: true
#
# Implementation: origination.features.context_adjustments.RefereeAdjustment
