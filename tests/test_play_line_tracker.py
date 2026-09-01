"""Tests for live play line-move / CLV helpers."""

from origination.gameday.play_line_tracker import (
    bet_timing_action,
    odds_clv_pct,
    steam_label,
    timing_note,
)


def test_odds_clv_positive_when_got_longer_price():
    # Bet over at 1.98, close 1.90 → positive CLV (steam toward over)
    assert odds_clv_pct(1.98, 1.90) == round((1.98 / 1.90 - 1) * 100, 2)


def test_odds_clv_negative_when_line_moved_against():
    assert odds_clv_pct(1.80, 1.82) < 0


def test_steam_toward_us():
    assert steam_label(1.98, 1.90) == "toward_us"
    assert steam_label(1.80, 1.82) == "against_us"
    assert steam_label(2.06, 2.06) == "flat"


def test_timing_note_early_rewarded():
    note = timing_note(1.98, 1.98, 1.90)
    assert "early" in note.lower() or "rewarded" in note.lower()


def test_bet_timing_action():
    assert bet_timing_action(1.98, 1.90, clv_last_pct=4.2, n_obs=3, tier="PLAY") == "BET_NOW"
    assert bet_timing_action(1.80, 1.82, clv_last_pct=-1.1, n_obs=3, tier="PLAY") == "WAIT"
    assert bet_timing_action(2.06, 2.06, clv_last_pct=0.0, n_obs=2, tier="WATCH") == "MONITOR"
    assert bet_timing_action(2.06, 2.06, n_obs=1, tier="PLAY") == "INSUFFICIENT_DATA"
