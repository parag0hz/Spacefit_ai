"""Unit tests for spacefit_v2/model/geom.py and features.py.

Run as:  python -m spacefit_v2.tests.test_features

Covers:
  * point-segment distance correctness
  * smooth_min approaches true min as T→0
  * signed distance sign inside/outside a convex polygon
  * effective_half_extents matches closed-form at yaw=0, 90°
  * soft_aabb_overlap is 0 for disjoint boxes, positive for overlapping
  * extract_placement_features returns (FEATURE_DIM,) tensor
  * gradients flow through (x, z, yaw) to features and to a scalar sum
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from spacefit_v2.model.features import (
    FEATURE_DIM,
    Furniture,
    PlacementContext,
    extract_placement_features,
)
from spacefit_v2.model.geom import (
    effective_half_extents,
    point_segment_distance,
    signed_distance_to_convex_polygon,
    smooth_min,
    soft_aabb_overlap,
)


_PASS = 0
_FAIL = 0


def _case(name: str, fn):
    global _PASS, _FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        _PASS += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        _FAIL += 1
    except Exception as e:  # pragma: no cover
        print(f"  ERROR {name}: {e}")
        traceback.print_exc()
        _FAIL += 1


# ── Geometry primitives ─────────────────────────────────────────────────────


def test_point_segment_distance_endpoint():
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([1.0, 0.0])
    d = point_segment_distance(torch.tensor(-1.0), torch.tensor(0.0), a, b)
    assert abs(d.item() - 1.0) < 1e-3, d.item()


def test_point_segment_distance_perpendicular():
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([2.0, 0.0])
    d = point_segment_distance(torch.tensor(1.0), torch.tensor(1.5), a, b)
    assert abs(d.item() - 1.5) < 1e-3, d.item()


def test_smooth_min_converges():
    values = torch.tensor([5.0, 2.0, 10.0, 1.0])
    sm = smooth_min(values, temperature=0.01)
    assert abs(sm.item() - 1.0) < 0.1, sm.item()


def test_smooth_min_differentiable():
    values = torch.tensor([5.0, 2.0, 1.0], requires_grad=True)
    sm = smooth_min(values, temperature=0.1)
    sm.backward()
    assert values.grad is not None
    # min index 2 should dominate
    assert values.grad[2].item() > values.grad[0].item()


def test_signed_distance_inside_outside():
    # Unit square CCW: (0,0), (1,0), (1,1), (0,1)
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # interior point
    sd_in = signed_distance_to_convex_polygon(
        torch.tensor(0.5), torch.tensor(0.5), square, temperature=0.01
    )
    # exterior point
    sd_out = signed_distance_to_convex_polygon(
        torch.tensor(2.0), torch.tensor(0.5), square, temperature=0.01
    )
    assert sd_in.item() < 0, f"expected negative for inside, got {sd_in.item()}"
    assert sd_out.item() > 0, f"expected positive for outside, got {sd_out.item()}"


def test_effective_half_extents_axis_aligned():
    hx, hz = effective_half_extents(
        torch.tensor(2.0), torch.tensor(1.0), torch.tensor(0.0)
    )
    assert abs(hx.item() - 1.0) < 1e-3
    assert abs(hz.item() - 0.5) < 1e-3


def test_effective_half_extents_rotated_90():
    hx, hz = effective_half_extents(
        torch.tensor(2.0), torch.tensor(1.0), torch.tensor(math.pi / 2)
    )
    # After 90° rotation, x-extent is depth/2=0.5, z-extent is width/2=1.0
    assert abs(hx.item() - 0.5) < 1e-2, hx.item()
    assert abs(hz.item() - 1.0) < 1e-2, hz.item()


def test_aabb_overlap_disjoint_zero():
    ov = soft_aabb_overlap(
        torch.tensor(0.0), torch.tensor(0.0),
        torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0),
        torch.tensor(5.0), torch.tensor(5.0),
        torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0),
    )
    assert ov.item() == 0.0


def test_aabb_overlap_full_overlap():
    ov = soft_aabb_overlap(
        torch.tensor(0.0), torch.tensor(0.0),
        torch.tensor(2.0), torch.tensor(2.0), torch.tensor(0.0),
        torch.tensor(0.0), torch.tensor(0.0),
        torch.tensor(2.0), torch.tensor(2.0), torch.tensor(0.0),
    )
    assert abs(ov.item() - 4.0) < 1e-3, ov.item()


# ── Feature extraction ─────────────────────────────────────────────────────


def _sample_context():
    polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    existing = [Furniture(x=3.5, z=0.5, yaw=0.0, width=1.0, depth=0.6, height=1.8, category="nightstand")]
    return PlacementContext(floor_polygon=polygon, existing_furniture=existing)


def test_feature_shape():
    ctx = _sample_context()
    feats = extract_placement_features(
        x=torch.tensor(2.0), z=torch.tensor(1.5), yaw=torch.tensor(0.0),
        width=2.0, depth=1.0, category="double_bed", context=ctx,
    )
    assert feats.shape == (FEATURE_DIM,), feats.shape


def test_features_finite():
    ctx = _sample_context()
    feats = extract_placement_features(
        x=torch.tensor(2.0), z=torch.tensor(1.5), yaw=torch.tensor(0.3),
        width=2.0, depth=1.0, category="double_bed", context=ctx,
    )
    assert torch.isfinite(feats).all(), feats


def test_features_differentiable():
    ctx = _sample_context()
    x = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(1.5, requires_grad=True)
    yaw = torch.tensor(0.0, requires_grad=True)
    feats = extract_placement_features(
        x=x, z=z, yaw=yaw,
        width=2.0, depth=1.0, category="double_bed", context=ctx,
    )
    feats.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad), x.grad
    assert z.grad is not None and torch.isfinite(z.grad), z.grad
    assert yaw.grad is not None and torch.isfinite(yaw.grad), yaw.grad


def test_has_related_set_when_nightstand_near_bed():
    ctx = _sample_context()
    feats = extract_placement_features(
        x=torch.tensor(2.0), z=torch.tensor(1.5), yaw=torch.tensor(0.0),
        width=2.0, depth=1.8, category="double_bed", context=ctx,
    )
    # Index 12 = has_related. A nightstand is in existing furniture.
    assert feats[12].item() == 1.0


def test_overlap_feature_positive_when_overlapping():
    ctx = _sample_context()
    # Place the new bed on top of the nightstand
    feats = extract_placement_features(
        x=torch.tensor(3.5), z=torch.tensor(0.5), yaw=torch.tensor(0.0),
        width=2.0, depth=1.0, category="double_bed", context=ctx,
    )
    # Index 8 = total_overlap
    assert feats[8].item() > 0.0, feats[8].item()


def test_boundary_margin_inside_positive():
    ctx = _sample_context()
    feats = extract_placement_features(
        x=torch.tensor(2.0), z=torch.tensor(1.5), yaw=torch.tensor(0.0),
        width=0.5, depth=0.5, category="table", context=ctx,
    )
    # boundary_margin = -signed_distance, so inside → positive
    assert feats[6].item() > 0.0, feats[6].item()


def main():
    cases = [
        ("point_segment_distance_endpoint", test_point_segment_distance_endpoint),
        ("point_segment_distance_perpendicular", test_point_segment_distance_perpendicular),
        ("smooth_min_converges", test_smooth_min_converges),
        ("smooth_min_differentiable", test_smooth_min_differentiable),
        ("signed_distance_inside_outside", test_signed_distance_inside_outside),
        ("effective_half_extents_axis_aligned", test_effective_half_extents_axis_aligned),
        ("effective_half_extents_rotated_90", test_effective_half_extents_rotated_90),
        ("aabb_overlap_disjoint_zero", test_aabb_overlap_disjoint_zero),
        ("aabb_overlap_full_overlap", test_aabb_overlap_full_overlap),
        ("feature_shape", test_feature_shape),
        ("features_finite", test_features_finite),
        ("features_differentiable", test_features_differentiable),
        ("has_related_set_when_nightstand_near_bed", test_has_related_set_when_nightstand_near_bed),
        ("overlap_feature_positive_when_overlapping", test_overlap_feature_positive_when_overlapping),
        ("boundary_margin_inside_positive", test_boundary_margin_inside_positive),
    ]
    print(f"Running {len(cases)} feature/geometry tests…")
    for name, fn in cases:
        _case(name, fn)
    print(f"\n  {_PASS} passed, {_FAIL} failed")
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
