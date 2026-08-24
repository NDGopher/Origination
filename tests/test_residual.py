"""Tests for additive logit residual."""

from __future__ import annotations

import numpy as np

from origination.models.residual import (
    _additive_apply_1x2,
    _additive_apply_ou,
    _target_deltas_1x2,
    _target_deltas_ou,
)


def test_additive_alpha_zero_unchanged():
    base = np.array([[0.5, 0.3, 0.2]])
    delta = np.array([[1.0, -0.5, -0.5]])
    out = _additive_apply_1x2(base, delta, 0.0)
    assert np.allclose(out, base / base.sum(), atol=1e-6)


def test_additive_1x2_sums_to_one():
    base = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    delta = np.array([[0.2, -0.1, -0.1], [-0.5, 0.2, 0.3]])
    out = _additive_apply_1x2(base, delta, 0.1)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_target_deltas_centered():
    p = np.array([[0.5, 0.3, 0.2]])
    y = np.array([0])
    d = _target_deltas_1x2(p, y, label_smoothing=0.02)
    assert abs(d.sum(axis=1)[0]) < 1e-8


def test_ou_additive_moves_toward_label():
    p = np.array([0.4, 0.6])
    # Force large positive delta → over probability rises
    out = _additive_apply_ou(p, np.array([2.0, 2.0]), 0.5)
    assert out[0] > p[0] and out[1] > p[1]


def test_ou_target_sign():
    d = _target_deltas_ou(np.array([0.3]), np.array([1.0]), label_smoothing=0.02)
    assert d[0] > 0  # observed over → positive correction vs under-confident base
