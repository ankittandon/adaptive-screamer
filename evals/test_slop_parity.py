"""Guard that the Python slop port matches the Rust detector.

These mirror the unit tests in `crates/screamer-styling/src/slop.rs`. They need
no Eval Protocol install (plain pytest), so they are the hermetic gate that runs
in CI. If slop.rs changes, update slop.py and these together.
"""

from scorers.slop import slop_analyze, slop_score


def test_clean_prose_scores_low():
    s = "Let's look at how Grammarly handles onboarding and mirror that for our styling layer."
    assert slop_score(s) < 15.0, slop_score(s)


def test_slop_scores_high():
    s = (
        "Let's delve into this — it's not just a feature, it's a seamless tapestry, "
        "which showcases a robust landscape, so here's the thing: we must leverage it."
    )
    b = slop_analyze(s)
    assert b["em_dashes"] >= 1
    assert b["buzzwords"] >= 3
    assert b["not_but"] >= 1
    assert b["over_joins"] >= 1
    assert b["signposts"] >= 1
    assert slop_score(s) > 40.0, slop_score(s)


def test_empty_is_zero():
    assert slop_score("") == 0.0
