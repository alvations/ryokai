"""Unit tests for the SimAlign-inspired layer-selection and distortion-filter
options on ContextualTokenSimBackend. No network — uses the constructor only."""
from ryokai.sim.contextual import ContextualTokenSimBackend


def test_default_layer_is_minus_one():
    be = ContextualTokenSimBackend("xlm-r-base")
    assert be.layer == -1


def test_layer_argument_stored():
    be = ContextualTokenSimBackend("xlm-r-base", layer=8)
    assert be.layer == 8


def test_max_distortion_default_is_none():
    be = ContextualTokenSimBackend("xlm-r-base")
    assert be.max_distortion is None


def test_max_distortion_argument_stored():
    be = ContextualTokenSimBackend("xlm-r-base", max_distortion=0.3)
    assert be.max_distortion == 0.3
