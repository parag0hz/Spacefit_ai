"""Create an actual-data Three.js pipeline viewer for one 3D-FRONT case."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.scripts.make_3dfront_threejs_scene_viewer import (  # noqa: E402
    DEFAULT_PREDICTIONS,
    build_scene_data,
)


DEFAULT_SOLVER = (
    ROOT
    / "spacefit_v2"
    / "results"
    / "final_constraint_solver_human_rerank"
    / "test_gpt_intent"
    / "raw_predictions.json"
)


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SpaceFit Actual 3D-FRONT Pipeline</title>
  <style>
    body { margin: 0; overflow: hidden; font-family: Inter, "Segoe UI", Arial, sans-serif; background: #f6f8fb; color: #111827; }
    #canvas { width: 100vw; height: 100vh; display: block; }
    .panel {
      position: fixed; left: 22px; top: 18px; width: min(560px, calc(100vw - 44px));
      padding: 15px 16px; border: 1px solid rgba(148,163,184,.45); border-radius: 14px;
      background: rgba(255,255,255,.86); backdrop-filter: blur(14px); box-shadow: 0 22px 60px rgba(15,23,42,.14);
    }
    h1 { margin: 0 0 8px; font-size: 18px; }
    .meta { margin: 3px 0; font-size: 12px; color: #475569; line-height: 1.45; }
    .intent { margin-top: 9px; padding: 9px 10px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; font-size: 12px; line-height: 1.45; color: #334155; }
    .stage {
      margin-top: 10px; padding: 10px; border-radius: 10px; background: #ecfdf5; border: 1px solid #bbf7d0;
      font-size: 13px; line-height: 1.45; color: #14532d; font-weight: 650;
    }
    .controls {
      position: fixed; left: 22px; bottom: 20px; right: 22px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
      padding: 12px 14px; border: 1px solid rgba(148,163,184,.42); border-radius: 18px;
      background: rgba(255,255,255,.88); backdrop-filter: blur(12px); box-shadow: 0 18px 45px rgba(15,23,42,.16);
    }
    button {
      border: 0; border-radius: 999px; padding: 9px 12px; background: #e2e8f0; color: #0f172a;
      font-size: 12px; font-weight: 720; cursor: pointer;
    }
    button.active { background: #16a34a; color: white; }
    .hint { position: fixed; right: 22px; top: 20px; padding: 10px 12px; border-radius: 999px; background: rgba(15,23,42,.78); color: white; font-size: 12px; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <section class="panel">
    <h1>Actual 3D-FRONT Processing Flow</h1>
    <p class="meta">Case: __CASE_ID__</p>
    <p class="meta">Room: __ROOM__ · Target: __TARGET__</p>
    <div class="intent">__INTENT__</div>
    <div id="stageText" class="stage"></div>
  </section>
  <div class="hint">drag: rotate · wheel: zoom · right-drag: pan</div>
  <div id="buttons" class="controls"></div>

  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
    }
  }
  </script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
    import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
    import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

    const DATA = __SCENE_JSON__;
    const STAGES = [
      ["Input", "기존 3D-FRONT 방 + 배치할 target 가구 mesh가 입력으로 들어온 상태입니다."],
      ["Target removed", "target은 고정 장면에서 분리되고, 기존 가구만 fixed context로 남습니다."],
      ["Free space", "방 경계와 기존 가구 점유 영역을 기준으로 후보를 만들 수 있는 빈 공간을 확인합니다."],
      ["Solver top-k", "실제 solver가 생성한 top-k (x,z,yaw) 후보를 target mesh 복사본으로 표시합니다."],
      ["Human rerank", "Random Forest human-aligned scorer가 top-k 후보를 다시 정렬한 결과입니다."],
      ["Final placement", "rerank 후 top-1 후보가 최종 배치 결과로 출력됩니다."]
    ];

    const renderer = new THREE.WebGLRenderer({ canvas: document.querySelector("#canvas"), antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.06;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f8fb);
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

    const center = new THREE.Vector3(DATA.center[0], 0, DATA.center[1]);
    const span = DATA.span;
    const camera = new THREE.PerspectiveCamera(39, window.innerWidth / window.innerHeight, 0.01, 200);
    camera.position.set(center.x + span * 0.68, span * 0.62, center.z + span * 0.82);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.target.set(center.x, 0.55, center.z);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c2cc, 1.9));
    const key = new THREE.DirectionalLight(0xffffff, 3.4);
    key.position.set(center.x + span * .35, span * .9, center.z + span * .25);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -span; key.shadow.camera.right = span; key.shadow.camera.top = span; key.shadow.camera.bottom = -span;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xdbeafe, 1.1);
    rim.position.set(center.x - span * .8, span * .5, center.z - span * .8);
    scene.add(rim);

    function makeMat(color, opacity = 1, roughness = .72) {
      return new THREE.MeshStandardMaterial({ color, roughness, metalness: .02, transparent: opacity < 1, opacity, side: THREE.DoubleSide });
    }

    const floorShape = new THREE.Shape();
    DATA.floor.forEach((p, i) => { if (i === 0) floorShape.moveTo(p[0], p[1]); else floorShape.lineTo(p[0], p[1]); });
    floorShape.closePath();
    const floorGeom = new THREE.ShapeGeometry(floorShape);
    floorGeom.rotateX(Math.PI / 2);
    const floor = new THREE.Mesh(floorGeom, makeMat(0xe8edf3));
    floor.receiveShadow = true;
    scene.add(floor);

    const wallMat = makeMat(0xf8fafc, .34);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x334155 });
    for (let i=0; i<DATA.floor.length; i++) {
      const a=DATA.floor[i], b=DATA.floor[(i+1)%DATA.floor.length], len=Math.hypot(b[0]-a[0], b[1]-a[1]);
      if (len < .08) continue;
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],0,a[1]), new THREE.Vector3(b[0],0,b[1]), new THREE.Vector3(b[0],2.6,b[1]), new THREE.Vector3(a[0],2.6,a[1])]);
      g.setIndex([0,1,2,0,2,3]); g.computeVertexNormals();
      scene.add(new THREE.Mesh(g, wallMat));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],.025,a[1]), new THREE.Vector3(b[0],.025,b[1])]), edgeMat));
    }

    function addOpeningLabel(text,x,z,color) {
      const c=document.createElement("canvas"); c.width=384; c.height=96; const ctx=c.getContext("2d");
      ctx.fillStyle="rgba(255,255,255,.9)"; ctx.strokeStyle="rgba(148,163,184,.7)"; ctx.lineWidth=3;
      ctx.roundRect(12,16,360,64,18); ctx.fill(); ctx.stroke();
      ctx.fillStyle=color; ctx.font="700 30px Segoe UI, Arial"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(text,192,48);
      const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));
      s.position.set(x,.85,z); s.scale.set(1.05,.27,1); scene.add(s);
    }
    function drawDoor(d) {
      const a=d.segment[0], b=d.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5, len=Math.hypot(b[0]-a[0],b[1]-a[1]);
      const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]);
      const panel=new THREE.Mesh(new THREE.BoxGeometry(len,.08,.055), makeMat(0xb45309,.92,.55));
      panel.position.set(mx,.05,mz); panel.rotation.y=-yaw; panel.castShadow=true; scene.add(panel);
      const clear=new THREE.Mesh(new THREE.CircleGeometry(Math.max(.55,len*.78),48,0,Math.PI*.62), new THREE.MeshBasicMaterial({color:0xf97316,transparent:true,opacity:.18,side:THREE.DoubleSide}));
      clear.rotation.x=-Math.PI/2; clear.rotation.z=-yaw; clear.position.set(a[0],.026,a[1]); scene.add(clear);
      addOpeningLabel("DOOR",mx,mz,"#92400e");
    }
    function drawWindow(w) {
      const a=w.segment[0], b=w.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5, len=Math.hypot(b[0]-a[0],b[1]-a[1]);
      const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]);
      const panel=new THREE.Mesh(new THREE.BoxGeometry(len,.12,.07), makeMat(0x38bdf8,.88,.25));
      panel.position.set(mx,1.15,mz); panel.rotation.y=-yaw; scene.add(panel);
      addOpeningLabel("WINDOW",mx,mz,"#0369a1");
    }
    (DATA.doors||[]).forEach(drawDoor); (DATA.windows||[]).forEach(drawWindow);

    const existingGroup = new THREE.Group();
    const stageGroup = new THREE.Group();
    const candidateGroup = new THREE.Group();
    const overlayGroup = new THREE.Group();
    scene.add(existingGroup, stageGroup, candidateGroup, overlayGroup);

    function addDisk(x, z, color, radius=.42, opacity=.55) {
      const mesh = new THREE.Mesh(new THREE.RingGeometry(radius*.72, radius, 48), new THREE.MeshBasicMaterial({ color, transparent:true, opacity, side: THREE.DoubleSide }));
      mesh.rotation.x = -Math.PI/2; mesh.position.set(x, .018, z); overlayGroup.add(mesh); return mesh;
    }

    function addLabel(text, x, z, color="#0f172a", parent=overlayGroup) {
      const c = document.createElement("canvas"); c.width=512; c.height=128;
      const ctx = c.getContext("2d");
      ctx.fillStyle="rgba(255,255,255,.88)"; ctx.strokeStyle="rgba(148,163,184,.75)"; ctx.lineWidth=4;
      ctx.roundRect(16,20,480,88,24); ctx.fill(); ctx.stroke();
      ctx.fillStyle=color; ctx.font="700 38px Segoe UI, Arial"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(text,256,64);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map:new THREE.CanvasTexture(c), transparent:true }));
      sprite.position.set(x,1.25,z); sprite.scale.set(1.25,.31,1); parent.add(sprite); return sprite;
    }

    function enhanceMaterial(mat, isTarget=false, tint=null, opacity=null) {
      if (!mat) return;
      mat.side = THREE.DoubleSide;
      if ("roughness" in mat) mat.roughness = Math.min(.78, Math.max(.36, mat.roughness ?? .55));
      if ("metalness" in mat) mat.metalness = Math.min(.12, mat.metalness ?? 0);
      if (tint && mat.color) mat.color.lerp(new THREE.Color(tint), isTarget ? .35 : .12);
      if (opacity !== null) { mat.transparent = true; mat.opacity = opacity; }
      mat.needsUpdate = true;
    }

    function normalizeLoadedObject(object, spec) {
      const box = new THREE.Box3().setFromObject(object);
      const size = new THREE.Vector3(); const ctr = new THREE.Vector3();
      box.getSize(size); box.getCenter(ctr);
      object.position.sub(ctr);
      object.scale.set(spec.size[0]/Math.max(size.x,1e-6), spec.size[1]/Math.max(size.y,1e-6), spec.size[2]/Math.max(size.z,1e-6));
      const box2 = new THREE.Box3().setFromObject(object);
      object.position.y -= box2.min.y;
    }

    function applyPose(object, pose) {
      object.position.x = pose.x; object.position.z = pose.z; object.rotation.y = -THREE.MathUtils.degToRad(pose.yaw || 0);
    }

    function loadObj(spec, opts={}) {
      return new Promise((resolve, reject) => {
        new MTLLoader().setPath(spec.asset_path + "/").load("model.mtl", (materials) => {
          materials.preload();
          for (const key of Object.keys(materials.materials)) enhanceMaterial(materials.materials[key], opts.target, opts.tint, opts.opacity);
          new OBJLoader().setMaterials(materials).setPath(spec.asset_path + "/").load(spec.obj_name, (object) => {
            object.traverse((child) => {
              if (child.isMesh) {
                child.castShadow = true; child.receiveShadow = true;
                if (Array.isArray(child.material)) child.material.forEach(m => enhanceMaterial(m, opts.target, opts.tint, opts.opacity));
                else enhanceMaterial(child.material, opts.target, opts.tint, opts.opacity);
              }
            });
            normalizeLoadedObject(object, spec);
            resolve(object);
          }, undefined, reject);
        }, undefined, reject);
      });
    }

    let targetTemplate = null;
    let targetFinal = null;
    const candidateObjects = [];

    for (const spec of DATA.existing) {
      loadObj(spec, { opacity: .95 }).then((obj) => { applyPose(obj, spec.pose); existingGroup.add(obj); });
    }

    loadObj(DATA.target, { target:true, tint:0x22c55e }).then((obj) => {
      targetTemplate = obj;
      targetFinal = obj.clone(true);
      stageGroup.add(targetFinal);
      for (let i=0; i<DATA.solver_candidates.length; i++) {
        const c = DATA.solver_candidates[i];
        const clone = obj.clone(true);
        applyPose(clone, c.pose);
        clone.traverse((child) => { if (child.isMesh) {
          child.material = child.material.clone(); enhanceMaterial(child.material, true, 0xf97316, .42);
        }});
        candidateGroup.add(clone);
        candidateObjects.push({ object: clone, data: c, kind: "solver" });
      }
      for (let i=0; i<DATA.rerank_candidates.length; i++) {
        const c = DATA.rerank_candidates[i];
        const clone = obj.clone(true);
        applyPose(clone, c.pose);
        clone.traverse((child) => { if (child.isMesh) {
          child.material = child.material.clone(); enhanceMaterial(child.material, true, 0x22c55e, .62);
        }});
        candidateGroup.add(clone);
        candidateObjects.push({ object: clone, data: c, kind: "rerank" });
      }
      setStage(0);
    });

    function clearOverlays() {
      while (overlayGroup.children.length) overlayGroup.remove(overlayGroup.children[0]);
    }

    function addFreeSpaceOverlay() {
      clearOverlays();
      const minX = Math.min(...DATA.floor.map(p=>p[0])), maxX = Math.max(...DATA.floor.map(p=>p[0]));
      const minZ = Math.min(...DATA.floor.map(p=>p[1])), maxZ = Math.max(...DATA.floor.map(p=>p[1]));
      for (let x=minX+.45; x<maxX-.3; x+=.55) for (let z=minZ+.45; z<maxZ-.3; z+=.55) addDisk(x,z,0x16a34a,.08,.32);
      for (const e of DATA.existing) addDisk(e.pose.x, e.pose.z, 0xef4444, .36, .28);
      addLabel("green dots = sampled free-space grid", center.x, center.z + span*.36, "#166534");
    }

    function addCandidateLabels(kind) {
      clearOverlays();
      const arr = kind === "solver" ? DATA.solver_candidates : DATA.rerank_candidates;
      arr.forEach((c, i) => {
        addDisk(c.pose.x, c.pose.z, kind === "solver" ? 0xf97316 : 0x22c55e, .36, .48);
        addLabel(`${i+1}`, c.pose.x, c.pose.z, kind === "solver" ? "#c2410c" : "#166534");
      });
    }

    function setVisibility({ target=false, solver=false, rerank=false }) {
      if (targetFinal) targetFinal.visible = target;
      for (const item of candidateObjects) item.object.visible = (item.kind === "solver" && solver) || (item.kind === "rerank" && rerank);
    }

    function setStage(idx) {
      document.querySelectorAll("button").forEach((b, i) => b.classList.toggle("active", i===idx));
      document.querySelector("#stageText").textContent = `${idx}. ${STAGES[idx][0]} — ${STAGES[idx][1]}`;
      clearOverlays();
      if (!targetFinal) return;
      if (idx === 0) { setVisibility({target:true}); applyPose(targetFinal, DATA.staging_pose); addDisk(DATA.staging_pose.x, DATA.staging_pose.z, 0x22c55e); addLabel("target asset", DATA.staging_pose.x, DATA.staging_pose.z, "#166534"); }
      if (idx === 1) { setVisibility({}); addLabel("fixed 3D-FRONT room after target removal", center.x, center.z, "#334155"); }
      if (idx === 2) { setVisibility({}); addFreeSpaceOverlay(); }
      if (idx === 3) { setVisibility({solver:true}); addCandidateLabels("solver"); }
      if (idx === 4) { setVisibility({rerank:true}); addCandidateLabels("rerank"); }
      if (idx === 5) { setVisibility({target:true}); applyPose(targetFinal, DATA.after_pose); addDisk(DATA.after_pose.x, DATA.after_pose.z, 0x22c55e); addLabel("final top-1", DATA.after_pose.x, DATA.after_pose.z, "#166534"); }
    }

    const buttons = document.querySelector("#buttons");
    STAGES.forEach((s, i) => {
      const b = document.createElement("button"); b.textContent = `${i}. ${s[0]}`;
      b.addEventListener("click", () => setStage(i)); buttons.appendChild(b);
    });

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); }
    animate();
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_rows(predictions: Dict[str, Any], method: str, case_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    for idx, item in enumerate((predictions.get(method) or {}).get(case_id, [])[:limit], 1):
        if item.get("status", "placed") != "placed":
            continue
        pos = item["position"]
        out.append(
            {
                "rank": idx,
                "score": item.get("score"),
                "human_aligned_score": item.get("human_aligned_score"),
                "human_aligned_original_rank": item.get("human_aligned_original_rank"),
                "pose": {"x": float(pos["x"]), "z": float(pos["z"]), "yaw": float(item.get("rotation_y", 0.0))},
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case_id", required=True)
    p.add_argument("--method", default="constraint_solver")
    p.add_argument("--solver_predictions", type=Path, default=DEFAULT_SOLVER)
    p.add_argument("--rerank_predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rerank_path = args.rerank_predictions if args.rerank_predictions.is_absolute() else ROOT / args.rerank_predictions
    solver_path = args.solver_predictions if args.solver_predictions.is_absolute() else ROOT / args.solver_predictions
    data = build_scene_data(args.case_id, args.method, rerank_path, out_dir)
    solver_preds = load_json(solver_path)
    rerank_preds = load_json(rerank_path)
    data["solver_candidates"] = candidate_rows(solver_preds, args.method, args.case_id)
    data["rerank_candidates"] = candidate_rows(rerank_preds, args.method, args.case_id)

    scene_json = json.dumps(data, ensure_ascii=False)
    html = (
        HTML.replace("__CASE_ID__", data["case_id"])
        .replace("__ROOM__", data["room_type"])
        .replace("__TARGET__", data["target_category"])
        .replace("__INTENT__", data["intent"])
        .replace("__SCENE_JSON__", scene_json)
    )
    (out_dir / "scene_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
