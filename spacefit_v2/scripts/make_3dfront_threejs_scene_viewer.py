"""Create a Three.js before/after viewer for a SpaceFit 3D-FRONT case.

The generated viewer shows:
- existing room furniture
- target furniture in a staging area (before)
- target furniture at predicted placement (after)
- a toggle and blend slider for scenario presentation
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "spacefit_v2" / "data" / "single_target_benchmark" / "cases"
FRONT_DIR = ROOT / "dataset" / "3D-FRONT"
FUTURE_DIRS = [
    ROOT / "dataset" / "3D-FUTURE-model",
    ROOT / "dataset" / "3D-FUTURE-model-part1",
]
DEFAULT_PREDICTIONS = (
    ROOT
    / "spacefit_v2"
    / "results"
    / "final_constraint_solver_human_rerank"
    / "test_gpt_intent"
    / "raw_predictions_human_reranked.json"
)
GPT_INTENT_CASES = ROOT / "spacefit_v2" / "data" / "single_target_benchmark" / "gpt_intent_cases_test.json"


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SpaceFit 3D Scenario Viewer</title>
  <style>
    :root {{
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      background: #f6f8fb;
      color: #111827;
    }}
    body {{
      margin: 0;
      overflow: hidden;
      background: radial-gradient(circle at 34% 18%, #ffffff 0%, #eef2f7 58%, #dde5ee 100%);
    }}
    #canvas {{ width: 100vw; height: 100vh; display: block; }}
    .panel {{
      position: fixed;
      left: 22px;
      top: 18px;
      width: min(520px, calc(100vw - 44px));
      padding: 15px 16px 14px;
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.84);
      backdrop-filter: blur(14px);
      box-shadow: 0 22px 60px rgba(15, 23, 42, 0.14);
    }}
    h1 {{ margin: 0 0 8px; font-size: 18px; line-height: 1.2; letter-spacing: 0; }}
    .meta {{ margin: 3px 0; font-size: 12px; line-height: 1.45; color: #475569; }}
    .intent {{
      margin-top: 9px;
      padding: 9px 10px;
      border-radius: 10px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      font-size: 12px;
      line-height: 1.45;
      color: #334155;
    }}
    .controls {{
      position: fixed;
      left: 22px;
      bottom: 20px;
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      border: 1px solid rgba(148, 163, 184, 0.42);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.86);
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 9px 13px;
      background: #0f172a;
      color: white;
      font-size: 12px;
      font-weight: 720;
      cursor: pointer;
    }}
    button.secondary {{ background: #e2e8f0; color: #0f172a; }}
    input[type="range"] {{ width: 220px; accent-color: #16a34a; }}
    .status {{
      min-width: 74px;
      font-size: 12px;
      font-weight: 760;
      color: #166534;
      text-align: center;
    }}
    .hint {{
      position: fixed;
      right: 22px;
      bottom: 20px;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.78);
      color: white;
      font-size: 12px;
      box-shadow: 0 14px 35px rgba(15, 23, 42, 0.22);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #dcfce7;
      color: #166534;
      font-size: 11px;
      font-weight: 760;
    }}
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <section class="panel">
    <h1>SpaceFit 3D Scenario Viewer <span class="badge">Before → After</span></h1>
    <p class="meta">Case: {case_id}</p>
    <p class="meta">Room: {room_type} · Target: {target_category}</p>
    <div class="intent">{intent}</div>
  </section>
  <div class="controls">
    <button id="beforeBtn" class="secondary">Before</button>
    <button id="afterBtn">After</button>
    <input id="blend" type="range" min="0" max="1" value="1" step="0.01" />
    <span id="status" class="status">After</span>
  </div>
  <div class="hint">drag: rotate · wheel: zoom · right-drag: pan</div>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
    }}
  }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ MTLLoader }} from "three/addons/loaders/MTLLoader.js";
    import {{ OBJLoader }} from "three/addons/loaders/OBJLoader.js";
    import {{ RoomEnvironment }} from "three/addons/environments/RoomEnvironment.js";

    const SCENE_DATA = {scene_json};
    const canvas = document.querySelector("#canvas");
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f8fb);
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

    const camera = new THREE.PerspectiveCamera(39, window.innerWidth / window.innerHeight, 0.01, 200);
    const center = new THREE.Vector3(SCENE_DATA.center[0], 0, SCENE_DATA.center[1]);
    const span = SCENE_DATA.span;
    camera.position.set(center.x + span * 0.68, span * 0.58, center.z + span * 0.82);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.target.set(center.x, 0.55, center.z);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c2cc, 1.9));
    const key = new THREE.DirectionalLight(0xffffff, 3.4);
    key.position.set(center.x + span * 0.35, span * 0.9, center.z + span * 0.25);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -span;
    key.shadow.camera.right = span;
    key.shadow.camera.top = span;
    key.shadow.camera.bottom = -span;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xdbeafe, 1.1);
    rim.position.set(center.x - span * 0.8, span * 0.5, center.z - span * 0.8);
    scene.add(rim);

    const root = new THREE.Group();
    scene.add(root);
    const targetGroup = new THREE.Group();
    scene.add(targetGroup);

    function makeMat(color, roughness = 0.68, opacity = 1.0) {{
      return new THREE.MeshStandardMaterial({{
        color,
        roughness,
        metalness: 0.02,
        transparent: opacity < 1.0,
        opacity,
        side: THREE.DoubleSide,
      }});
    }}

    const floorShape = new THREE.Shape();
    SCENE_DATA.floor.forEach((p, i) => {{
      if (i === 0) floorShape.moveTo(p[0], p[1]);
      else floorShape.lineTo(p[0], p[1]);
    }});
    floorShape.closePath();
    const floorGeom = new THREE.ShapeGeometry(floorShape);
    floorGeom.rotateX(Math.PI / 2);
    const floor = new THREE.Mesh(floorGeom, makeMat(0xe8edf3, 0.82));
    floor.receiveShadow = true;
    scene.add(floor);

    const wallMat = makeMat(0xf8fafc, 0.78, 0.34);
    const edgeMat = new THREE.LineBasicMaterial({{ color: 0x334155, linewidth: 2 }});
    const wallH = 2.6;
    for (let i = 0; i < SCENE_DATA.floor.length; i++) {{
      const a = SCENE_DATA.floor[i];
      const b = SCENE_DATA.floor[(i + 1) % SCENE_DATA.floor.length];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len < 0.08) continue;
      const geom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(a[0], 0, a[1]),
        new THREE.Vector3(b[0], 0, b[1]),
        new THREE.Vector3(b[0], wallH, b[1]),
        new THREE.Vector3(a[0], wallH, a[1]),
      ]);
      geom.setIndex([0, 1, 2, 0, 2, 3]);
      geom.computeVertexNormals();
      const wall = new THREE.Mesh(geom, wallMat);
      scene.add(wall);
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(a[0], 0.02, a[1]),
        new THREE.Vector3(b[0], 0.02, b[1]),
      ]);
      scene.add(new THREE.Line(lineGeom, edgeMat));
    }}

    const staging = new THREE.Mesh(
      new THREE.RingGeometry(0.32, 0.44, 48),
      new THREE.MeshBasicMaterial({{ color: 0x16a34a, transparent: true, opacity: 0.55, side: THREE.DoubleSide }})
    );
    staging.rotation.x = -Math.PI / 2;
    staging.position.set(SCENE_DATA.staging_pose.x, 0.015, SCENE_DATA.staging_pose.z);
    scene.add(staging);

    const targetHalo = new THREE.Mesh(
      new THREE.RingGeometry(0.42, 0.58, 64),
      new THREE.MeshBasicMaterial({{ color: 0x22c55e, transparent: true, opacity: 0.68, side: THREE.DoubleSide }})
    );
    targetHalo.rotation.x = -Math.PI / 2;
    scene.add(targetHalo);

    const labels = [];
    function addLabel(text, x, z, color = "#0f172a") {{
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 128;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.strokeStyle = "rgba(148,163,184,0.72)";
      ctx.lineWidth = 4;
      ctx.roundRect(16, 20, 480, 88, 24);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "700 38px Segoe UI, Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 256, 64);
      const texture = new THREE.CanvasTexture(canvas);
      const mat = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(x, 1.25, z);
      sprite.scale.set(1.25, 0.31, 1);
      scene.add(sprite);
      labels.push(sprite);
      return sprite;
    }}
    addLabel("Target staging", SCENE_DATA.staging_pose.x, SCENE_DATA.staging_pose.z, "#166534");
    addLabel("Final placement", SCENE_DATA.after_pose.x, SCENE_DATA.after_pose.z, "#166534");

    function enhanceMaterial(mat, isTarget=false) {{
      if (!mat) return;
      mat.side = THREE.DoubleSide;
      if ("roughness" in mat) mat.roughness = Math.min(0.78, Math.max(0.36, mat.roughness ?? 0.55));
      if ("metalness" in mat) mat.metalness = Math.min(0.12, mat.metalness ?? 0.0);
      if (isTarget && mat.color) {{
        mat.color.lerp(new THREE.Color(0x22c55e), 0.18);
      }}
      mat.needsUpdate = true;
    }}

    function normalizeLoadedObject(object, spec) {{
      const box = new THREE.Box3().setFromObject(object);
      const size = new THREE.Vector3();
      const centerObj = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(centerObj);
      object.position.sub(centerObj);
      object.scale.set(
        spec.size[0] / Math.max(size.x, 1e-6),
        spec.size[1] / Math.max(size.y, 1e-6),
        spec.size[2] / Math.max(size.z, 1e-6)
      );
      const box2 = new THREE.Box3().setFromObject(object);
      object.position.y -= box2.min.y;
    }}

    function applyPose(object, pose) {{
      object.position.x = pose.x;
      object.position.z = pose.z;
      object.rotation.y = -THREE.MathUtils.degToRad(pose.yaw || 0);
    }}

    function loadObj(spec, isTarget=false) {{
      return new Promise((resolve, reject) => {{
        new MTLLoader()
          .setPath(spec.asset_path + "/")
          .load("model.mtl", (materials) => {{
            materials.preload();
            for (const key of Object.keys(materials.materials)) enhanceMaterial(materials.materials[key], isTarget);
            new OBJLoader()
              .setMaterials(materials)
              .setPath(spec.asset_path + "/")
              .load(spec.obj_name, (object) => {{
                object.traverse((child) => {{
                  if (child.isMesh) {{
                    child.castShadow = true;
                    child.receiveShadow = true;
                    if (Array.isArray(child.material)) child.material.forEach(m => enhanceMaterial(m, isTarget));
                    else enhanceMaterial(child.material, isTarget);
                  }}
                }});
                normalizeLoadedObject(object, spec);
                resolve(object);
              }}, undefined, reject);
          }}, undefined, reject);
      }});
    }}

    for (const spec of SCENE_DATA.existing) {{
      loadObj(spec).then((object) => {{
        applyPose(object, spec.pose);
        root.add(object);
      }});
    }}

    let targetObject = null;
    loadObj(SCENE_DATA.target, true).then((object) => {{
      targetObject = object;
      targetGroup.add(object);
      updateBlend(Number(document.querySelector("#blend").value));
    }});

    function lerp(a, b, t) {{ return a + (b - a) * t; }}
    function updateBlend(t) {{
      const before = SCENE_DATA.staging_pose;
      const after = SCENE_DATA.after_pose;
      const x = lerp(before.x, after.x, t);
      const z = lerp(before.z, after.z, t);
      const yaw = lerp(before.yaw || 0, after.yaw || 0, t);
      if (targetObject) applyPose(targetObject, {{ x, z, yaw }});
      targetHalo.position.set(x, 0.018, z);
      document.querySelector("#status").textContent = t < 0.05 ? "Before" : (t > 0.95 ? "After" : "Blend");
    }}

    const slider = document.querySelector("#blend");
    slider.addEventListener("input", () => updateBlend(Number(slider.value)));
    document.querySelector("#beforeBtn").addEventListener("click", () => {{ slider.value = 0; updateBlend(0); }});
    document.querySelector("#afterBtn").addEventListener("click", () => {{ slider.value = 1; updateBlend(1); }});

    function resize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}
    window.addEventListener("resize", resize);

    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluated_intent_text(case: Dict[str, Any], case_id: str) -> str:
    """Return the user intent used by the GPT-intent benchmark when available."""
    if GPT_INTENT_CASES.exists():
        try:
            for row in load_json(GPT_INTENT_CASES):
                if str(row.get("id")) == case_id:
                    intent = row.get("intent") or {}
                    if isinstance(intent, dict) and intent.get("text"):
                        return str(intent["text"])
        except Exception:
            pass
    intent = case.get("intent") or {}
    return str(intent.get("text", "")) if isinstance(intent, dict) else str(intent)


def find_room(raw_scene: Dict[str, Any], room_id: str) -> Dict[str, Any]:
    for room in raw_scene.get("scene", {}).get("room", []):
        if str(room.get("instanceid")) == room_id:
            return room
    raise ValueError(f"Room not found: {room_id}")


def furniture_id_to_child(room: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(child.get("instanceid")): child
        for child in room.get("children", [])
        if str(child.get("instanceid", "")).startswith("furniture/")
    }


def furniture_uid_to_jid(raw_scene: Dict[str, Any]) -> Dict[str, str]:
    return {
        str(item.get("uid")): str(item.get("jid"))
        for item in raw_scene.get("furniture", [])
        if item.get("uid") and item.get("jid")
    }


def future_model_dir(jid: str) -> Path | None:
    for base in FUTURE_DIRS:
        path = base / jid
        if (path / "normalized_model.obj").exists() and (path / "model.mtl").exists():
            return path
    return None


def yaw_from_quat_y_up(rot: List[float] | None) -> float:
    if not rot or len(rot) < 4:
        return 0.0
    x, y, z, w = [float(v) for v in rot[:4]]
    siny = 2.0 * (w * y + x * z)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def prediction_for_case(predictions: Dict[str, Any], method: str, case_id: str) -> Dict[str, Any]:
    entries = (predictions.get(method) or {}).get(case_id) or []
    placed = [entry for entry in entries if entry.get("status", "placed") == "placed"]
    if not placed and not entries:
        raise ValueError(f"No prediction for {case_id}")
    return dict(placed[0] if placed else entries[0])


def copy_asset(asset_dir: Path, out_assets: Path, alias: str) -> str:
    dst = out_assets / alias
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["normalized_model.obj", "model.mtl", "texture.png", "image.jpg"]:
        src = asset_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)
    return f"assets/{alias}"


def pose(x: float, z: float, yaw: float) -> Dict[str, float]:
    return {"x": float(x), "z": float(z), "yaw": float(yaw)}


def opening_specs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = []
    for item in items:
        pos = item.get("position") or [0.0, 0.0, 0.0]
        seg = item.get("segment") or []
        if len(seg) >= 2:
            p0 = [float(seg[0][0]), float(seg[0][1])]
            p1 = [float(seg[1][0]), float(seg[1][1])]
        else:
            x = float(pos[0])
            z = float(pos[2] if len(pos) > 2 else pos[1])
            yaw = math.radians(float(item.get("yaw", 0.0)))
            half = float(item.get("width", 0.8)) * 0.5
            dx = math.cos(yaw) * half
            dz = math.sin(yaw) * half
            p0 = [x - dx, z - dz]
            p1 = [x + dx, z + dz]
        specs.append(
            {
                "id": str(item.get("id", "")),
                "position": [float(pos[0]), float(pos[2] if len(pos) > 2 else pos[1])],
                "segment": [p0, p1],
                "width": float(item.get("width", 0.8)),
                "height": float(item.get("height", 2.0)),
                "yaw": float(item.get("yaw", 0.0)),
            }
        )
    return specs


def bounds_from_floor(floor: List[List[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in floor]
    zs = [float(p[1]) for p in floor]
    return min(xs), max(xs), min(zs), max(zs)


def build_scene_data(case_id: str, method: str, predictions_path: Path, out_dir: Path) -> Dict[str, Any]:
    case = load_json(CASE_DIR / f"{case_id}.json")
    predictions = load_json(predictions_path)
    raw_scene = load_json(FRONT_DIR / f"{case['scene']['scene_id']}.json")
    room = find_room(raw_scene, case["scene"]["room_id"])
    child_by_id = furniture_id_to_child(room)
    jid_by_uid = furniture_uid_to_jid(raw_scene)
    pred = prediction_for_case(predictions, method, case_id)
    target_id = str(case["target_asset"]["id"])
    out_assets = out_dir / "assets"

    existing: List[Dict[str, Any]] = []
    asset_index = 0
    for obj in case["scene"].get("objects", []):
        instance_id = str(obj["id"])
        child = child_by_id.get(instance_id)
        if child is None:
            continue
        jid = child.get("replace_jid") or jid_by_uid.get(str(child.get("ref")))
        asset_dir = future_model_dir(str(jid)) if jid else None
        if not asset_dir:
            continue
        asset_path = copy_asset(asset_dir, out_assets, f"existing_{asset_index:03d}_{asset_dir.name}")
        asset_index += 1
        child_pos = child.get("pos") or obj["position"]
        existing.append(
            {
                "name": str(obj.get("category", "object")),
                "asset_path": asset_path,
                "obj_name": "normalized_model.obj",
                "size": [float(obj["size"][0]), float(obj["size"][1]), float(obj["size"][2])],
                "pose": pose(float(child_pos[0]), float(child_pos[2]), yaw_from_quat_y_up(child.get("rot"))),
            }
        )

    target_child = child_by_id.get(target_id)
    if target_child is None:
        raise ValueError(f"Target child not found in raw room: {target_id}")
    target_jid = target_child.get("replace_jid") or jid_by_uid.get(str(target_child.get("ref")))
    target_dir = future_model_dir(str(target_jid)) if target_jid else None
    if not target_dir:
        raise ValueError(f"Target 3D-FUTURE model not found: {target_jid}")
    target_asset_path = copy_asset(target_dir, out_assets, f"target_{target_dir.name}")

    floor = [[float(x), float(z)] for x, z in case["scene"]["floor"]["polygon"]]
    min_x, max_x, min_z, max_z = bounds_from_floor(floor)
    span = max(max_x - min_x, max_z - min_z)
    center = [(min_x + max_x) / 2.0, (min_z + max_z) / 2.0]
    staging_x = min_x - max(1.0, span * 0.16)
    staging_z = center[1]

    size_dict = pred.get("size") or case["target_asset"]["size"]
    after = pred.get("position") or case["reference_pose"]["position"]
    after_pose = pose(float(after["x"]), float(after["z"]), float(pred.get("rotation_y", case["reference_pose"].get("rotation_y", 0.0))))

    return {
        "case_id": case_id,
        "room_type": case["scene"].get("room_type", ""),
        "target_category": case["target_asset"]["category"],
        "intent": evaluated_intent_text(case, case_id),
        "floor": floor,
        "doors": opening_specs(case["scene"].get("doors", [])),
        "windows": opening_specs(case["scene"].get("windows", [])),
        "center": center,
        "span": span,
        "existing": existing,
        "target": {
            "name": case["target_asset"]["category"],
            "asset_path": target_asset_path,
            "obj_name": "normalized_model.obj",
            "size": [float(size_dict["width"]), float(size_dict["height"]), float(size_dict["depth"])],
        },
        "staging_pose": pose(staging_x, staging_z, 0.0),
        "after_pose": after_pose,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--method", default="constraint_solver")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = args.predictions if args.predictions.is_absolute() else ROOT / args.predictions
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_data = build_scene_data(args.case_id, args.method, predictions, out_dir)
    (out_dir / "scene_data.json").write_text(json.dumps(scene_data, ensure_ascii=False, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.format(
        case_id=scene_data["case_id"],
        room_type=scene_data["room_type"],
        target_category=scene_data["target_category"],
        intent=scene_data["intent"],
        scene_json=json.dumps(scene_data, ensure_ascii=False),
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(out_dir / "index.html")
    print("Serve with: python -m http.server 8898 --directory", out_dir)


if __name__ == "__main__":
    main()
