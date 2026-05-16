"""Pure-Python tests: no model downloads, fast."""
from ryokai.labelconfig import LabelConfig


def test_canonical_propbank_args():
    lc = LabelConfig()
    assert lc.canonical("ARG0") == "A0"
    assert lc.canonical("ARG1") == "A1"
    assert lc.canonical("ARG2") == "A2"


def test_canonical_continuation_and_reference_tags():
    lc = LabelConfig()
    # continuation
    assert lc.canonical("C-ARG0") == "A0"
    assert lc.canonical("C-ARG1") == "A1"
    # reference
    assert lc.canonical("R-ARG0") == "A0"
    assert lc.canonical("R-ARG1") == "A1"


def test_canonical_ancora_lowercase():
    lc = LabelConfig()
    assert lc.canonical("arg0-agt") == "A0"
    assert lc.canonical("arg1-tem") == "A1"
    assert lc.canonical("arg2-atr") == "A2"


def test_canonical_chinese_propbank_short_forms():
    lc = LabelConfig()
    assert lc.canonical("A0") == "A0"
    assert lc.canonical("A1") == "A1"


def test_canonical_modifiers():
    lc = LabelConfig()
    assert lc.canonical("ARGM-TMP") == "TMP"
    assert lc.canonical("ARGM-LOC") == "LOC"
    assert lc.canonical("ARGM-MNR") == "MNR"
    assert lc.canonical("ARGM-NEG") == "NEG"


def test_canonical_case_insensitive():
    lc = LabelConfig()
    assert lc.canonical("arg0") == "A0"
    assert lc.canonical("Arg1") == "A1"


def test_canonical_unknown_label_falls_through_to_other():
    lc = LabelConfig()
    assert lc.canonical("THIS-DOES-NOT-EXIST") == "OTHER"
    assert lc.canonical("") == "OTHER"
    assert lc.canonical(None) == "OTHER"


def test_contains_dunder():
    lc = LabelConfig()
    assert "ARG0" in lc
    assert "arg0-agt" in lc
    assert "BOGUS-LABEL" not in lc
