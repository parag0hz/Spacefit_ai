"""Unit tests for differentiable losses and refinement."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from spacefit_v2.model.losses import compute_boundary_loss, compute_collision_loss, loss_physics
from spacefit_v2.optim.diff_refine import DifferentiableRefiner
from spacefit_v2.single_target.methods import _constraint_loss


def _simple_scene():
    return {
        "floor_plan_vertices": [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)],
        "walls": [],
        "doors": [],
        "existing_furniture": [
            {
                "id": "desk-0",
                "category": "desk",
                "size": {"width": 1.0, "depth": 0.6, "height": 0.75},
                "position": {"x": 3.2, "z": 1.5},
                "rotation_y": 0.0,
            }
        ],
    }


def _constraint_scene():
    return {
        "floor": {"polygon": [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]},
        "walls": [],
        "doors": [],
        "windows": [],
        "objects": [
            {
                "id": "sofa-0",
                "category": "sofa",
                "position": [2.0, 0.0, 1.5],
                "size": [1.0, 0.8, 0.6],
                "yaw": 0.0,
            }
        ],
    }


def main() -> int:
    polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    pos_inside = [torch.tensor([2.0, 1.5], dtype=torch.float32)]
    pos_outside = [torch.tensor([4.5, 1.5], dtype=torch.float32)]
    rot = [torch.tensor(0.0)]
    size = [{"width": 1.0, "depth": 1.0, "height": 0.8}]

    l_in = compute_boundary_loss(pos_inside, rot, size, polygon)
    l_out = compute_boundary_loss(pos_outside, rot, size, polygon)
    assert l_out.item() > l_in.item(), (l_in.item(), l_out.item())

    obstacles = [{
        "position": {"x": 2.0, "z": 1.5},
        "rotation_y": 0.0,
        "size": {"width": 1.0, "depth": 1.0, "height": 0.8},
    }]
    l_col = compute_collision_loss(pos_inside, rot, size, obstacles)
    assert l_col.item() > 0.0

    refiner = DifferentiableRefiner(device="cpu")
    placements = refiner.refine(
        scene=_simple_scene(),
        new_furniture=[{
            "id": "bed-1",
            "category": "double_bed",
            "size": {"width": 1.8, "depth": 2.0, "height": 0.6},
        }],
        candidate_regions=[{
            "id": 0,
            "bounds": (0.0, 0.0, 2.0, 3.0),
            "centroid": (1.0, 1.5),
            "area": 6.0,
        }],
        selected_regions={"bed-1": 0},
        n_iters=50,
        lr=0.05,
    )
    assert placements and placements[0]["status"] == "placed", placements
    x = placements[0]["position"]["x"]
    z = placements[0]["position"]["z"]
    assert 0.0 <= x <= 2.0 and 0.0 <= z <= 3.0

    rel_scene = _constraint_scene()
    rel_size = {"width": 0.5, "depth": 0.5, "height": 0.5}
    yaw = torch.tensor(0.0)
    front_loss = _constraint_loss(
        torch.tensor([2.0, 2.4]), yaw, rel_scene, rel_size,
        [{"constraint_type": "in_front_of", "target_category": "sofa"}],
    )
    behind_loss = _constraint_loss(
        torch.tensor([2.0, 0.6]), yaw, rel_scene, rel_size,
        [{"constraint_type": "in_front_of", "target_category": "sofa"}],
    )
    assert front_loss.item() < behind_loss.item(), (front_loss.item(), behind_loss.item())

    left_loss = _constraint_loss(
        torch.tensor([1.2, 1.5]), yaw, rel_scene, rel_size,
        [{"constraint_type": "left_of", "target_category": "sofa"}],
    )
    right_loss = _constraint_loss(
        torch.tensor([2.8, 1.5]), yaw, rel_scene, rel_size,
        [{"constraint_type": "left_of", "target_category": "sofa"}],
    )
    assert left_loss.item() < right_loss.item(), (left_loss.item(), right_loss.item())

    beside_loss = _constraint_loss(
        torch.tensor([2.6, 1.5]), yaw, rel_scene, rel_size,
        [{"constraint_type": "beside", "target_category": "sofa"}],
    )
    far_loss = _constraint_loss(
        torch.tensor([3.8, 2.8]), yaw, rel_scene, rel_size,
        [{"constraint_type": "beside", "target_category": "sofa"}],
    )
    assert beside_loss.item() < far_loss.item(), (beside_loss.item(), far_loss.item())

    print("test_diffopt: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
