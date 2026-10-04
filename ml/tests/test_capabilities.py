"""
The platform's core promise is that it runs anywhere. These tests assert that
the optional-dependency layer behaves the same whether a package is present or
missing, and that nothing silently becomes mandatory.
"""

from __future__ import annotations

from ai.common import CAPABILITIES, capability_report, has, optional_import


def test_report_is_complete():
    report = capability_report()
    assert report["tier"] in {"pure-python", "accelerated", "full-inference"}
    assert report["total_count"] == len(CAPABILITIES)
    assert report["available_count"] <= report["total_count"]
    assert isinstance(report["gpu"]["available"], bool)


def test_unavailable_capabilities_explain_themselves():
    """A missing dependency must say what is lost and how to install it."""
    for capability in capability_report()["capabilities"]:
        assert capability["purpose"]
        if not capability["available"]:
            assert capability["install"], f"{capability['name']} has no install hint"
            assert capability["fallback"], f"{capability['name']} has no documented fallback"


def test_unknown_capability_is_false_not_an_error():
    assert has("definitely-not-a-real-package") is False
    assert optional_import("definitely-not-a-real-package") is None


def test_import_matches_availability():
    for name in CAPABILITIES:
        module = optional_import(name)
        if CAPABILITIES[name].module and has(name):
            assert module is not None, f"{name} reports available but did not import"
