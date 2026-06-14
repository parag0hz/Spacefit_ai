"""Create a Three.js actual 3D-FRONT method comparison viewer."""
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


BASELINE_PREDICTIONS = ROOT / "spacefit_v2" / "results" / "experiment_gpt_intent" / "test_gpt_intent" / "raw_predictions.json"
SOLVER_PREDICTIONS = ROOT / "spacefit_v2" / "results" / "final_constraint_solver_human_rerank" / "test_gpt_intent" / "raw_predictions.json"
RERANK_PREDICTIONS = DEFAULT_PREDICTIONS


METHOD_SPECS = [
    {
        "label": "Direct Coordinate Prediction",
        "short": "Direct",
        "path": BASELINE_PREDICTIONS,
        "method": "layoutgpt_direct",
        "color": "#ef4444",
        "desc": "LLM directly predicts final x, z, yaw from text scene description.",
    },
    {
        "label": "LLM Region Selection",
        "short": "LLM Region",
        "path": BASELINE_PREDICTIONS,
        "method": "spacefit_gpt_text",
        "color": "#8b5cf6",
        "desc": "LLM selects/grounds a plausible placement region, then placement module computes pose.",
    },
    {
        "label": "Candidate Search & Scoring",
        "short": "Search+Score",
        "path": SOLVER_PREDICTIONS,
        "method": "constraint_solver",
        "color": "#f97316",
        "desc": "Grid-searches feasible x, z, yaw candidates and ranks them with code-based constraint scoring.",
    },
    {
        "label": "Loss-based Pose Optimization",
        "short": "Loss Opt",
        "path": BASELINE_PREDICTIONS,
        "method": "proposal_diffopt_constraint",
        "color": "#0ea5e9",
        "desc": "Starts from an initial pose and optimizes x, z, yaw with constraint losses.",
    },
    {
        "label": "Preference-Reranked Candidate Selection",
        "short": "Ours",
        "path": RERANK_PREDICTIONS,
        "method": "constraint_solver",
        "color": "#16a34a",
        "desc": "Reranks top-k candidates using a Random Forest scorer trained from human visual audit labels.",
    },
]


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3D-FRONT Method Comparison</title>
  <style>
    body { margin:0; overflow:hidden; font-family:Inter,"Segoe UI",Arial,sans-serif; background:#f6f8fb; color:#111827; }
    #canvas { width:100vw; height:100vh; display:block; }
    .panel { position:fixed; left:22px; top:18px; width:min(620px,calc(100vw - 44px)); padding:15px 16px; border:1px solid rgba(148,163,184,.45); border-radius:14px; background:rgba(255,255,255,.87); backdrop-filter:blur(14px); box-shadow:0 22px 60px rgba(15,23,42,.14); }
    h1 { margin:0 0 8px; font-size:18px; }
    .meta { margin:3px 0; font-size:12px; color:#475569; line-height:1.45; }
    .intent { margin-top:9px; padding:9px 10px; border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0; font-size:12px; line-height:1.45; color:#334155; }
    .methodBox { margin-top:10px; padding:10px; border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0; font-size:13px; line-height:1.45; color:#0f172a; }
    .methodTitle { font-weight:800; }
    .controls { position:fixed; left:22px; bottom:20px; right:22px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:12px 14px; border:1px solid rgba(148,163,184,.42); border-radius:18px; background:rgba(255,255,255,.88); backdrop-filter:blur(12px); box-shadow:0 18px 45px rgba(15,23,42,.16); }
    button { border:0; border-radius:999px; padding:9px 12px; background:#e2e8f0; color:#0f172a; font-size:12px; font-weight:720; cursor:pointer; }
    button.active { color:white; }
    .hint { position:fixed; right:22px; top:20px; padding:10px 12px; border-radius:999px; background:rgba(15,23,42,.78); color:white; font-size:12px; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <section class="panel">
    <h1>Actual 3D-FRONT Method Comparison</h1>
    <p class="meta">Case: __CASE_ID__</p>
    <p class="meta">Room: __ROOM__ · Target: __TARGET__</p>
    <div class="intent">__INTENT__</div>
    <div id="methodBox" class="methodBox"></div>
  </section>
  <div class="hint">drag: rotate · wheel: zoom · right-drag: pan</div>
  <div id="buttons" class="controls"></div>

  <script type="importmap">
  {"imports":{"three":"https://unpkg.com/three@0.164.1/build/three.module.js","three/addons/":"https://unpkg.com/three@0.164.1/examples/jsm/"}}
  </script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
    import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
    import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

    const DATA = __SCENE_JSON__;
    const renderer = new THREE.WebGLRenderer({canvas:document.querySelector("#canvas"), antialias:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2)); renderer.setSize(window.innerWidth,window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.06;
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf6f8fb);
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), .04).texture;
    const center = new THREE.Vector3(DATA.center[0],0,DATA.center[1]), span = DATA.span;
    const camera = new THREE.PerspectiveCamera(39, window.innerWidth/window.innerHeight, .01, 200);
    camera.position.set(center.x+span*.68, span*.62, center.z+span*.82);
    const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.dampingFactor=.055; controls.target.set(center.x,.55,center.z); controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff,0xb8c2cc,1.9));
    const key = new THREE.DirectionalLight(0xffffff,3.4); key.position.set(center.x+span*.35,span*.9,center.z+span*.25); key.castShadow=true; key.shadow.mapSize.set(2048,2048); key.shadow.camera.left=-span; key.shadow.camera.right=span; key.shadow.camera.top=span; key.shadow.camera.bottom=-span; scene.add(key);
    const rim = new THREE.DirectionalLight(0xdbeafe,1.1); rim.position.set(center.x-span*.8,span*.5,center.z-span*.8); scene.add(rim);

    function makeMat(color, opacity=1, roughness=.72){ return new THREE.MeshStandardMaterial({color,roughness,metalness:.02,transparent:opacity<1,opacity,side:THREE.DoubleSide}); }
    const floorShape = new THREE.Shape(); DATA.floor.forEach((p,i)=>{ if(i===0) floorShape.moveTo(p[0],p[1]); else floorShape.lineTo(p[0],p[1]); }); floorShape.closePath();
    const floorGeom = new THREE.ShapeGeometry(floorShape); floorGeom.rotateX(Math.PI/2); const floor = new THREE.Mesh(floorGeom, makeMat(0xe8edf3)); floor.receiveShadow=true; scene.add(floor);
    const wallMat = makeMat(0xf8fafc,.34), edgeMat = new THREE.LineBasicMaterial({color:0x334155});
    for(let i=0;i<DATA.floor.length;i++){ const a=DATA.floor[i], b=DATA.floor[(i+1)%DATA.floor.length], len=Math.hypot(b[0]-a[0],b[1]-a[1]); if(len<.08) continue; const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],0,a[1]),new THREE.Vector3(b[0],0,b[1]),new THREE.Vector3(b[0],2.6,b[1]),new THREE.Vector3(a[0],2.6,a[1])]); g.setIndex([0,1,2,0,2,3]); g.computeVertexNormals(); scene.add(new THREE.Mesh(g,wallMat)); scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],.025,a[1]),new THREE.Vector3(b[0],.025,b[1])]), edgeMat)); }

    function addOpeningLabel(text,x,z,color){ const c=document.createElement("canvas"); c.width=384; c.height=96; const ctx=c.getContext("2d"); ctx.fillStyle="rgba(255,255,255,.9)"; ctx.strokeStyle="rgba(148,163,184,.7)"; ctx.lineWidth=3; ctx.roundRect(12,16,360,64,18); ctx.fill(); ctx.stroke(); ctx.fillStyle=color; ctx.font="700 30px Segoe UI, Arial"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(text,192,48); const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true})); s.position.set(x,.85,z); s.scale.set(1.05,.27,1); scene.add(s); }
    function drawDoor(d){ const a=d.segment[0], b=d.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5, len=Math.hypot(b[0]-a[0],b[1]-a[1]); const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]); const mat=makeMat(0xb45309,.92,.55); const panel=new THREE.Mesh(new THREE.BoxGeometry(len,.08,.055),mat); panel.position.set(mx,.05,mz); panel.rotation.y=-yaw; panel.castShadow=true; scene.add(panel); const clear=new THREE.Mesh(new THREE.CircleGeometry(Math.max(.55,len*.78),48,0,Math.PI*.62), new THREE.MeshBasicMaterial({color:0xf97316,transparent:true,opacity:.18,side:THREE.DoubleSide})); clear.rotation.x=-Math.PI/2; clear.rotation.z=-yaw; clear.position.set(a[0],.026,a[1]); scene.add(clear); addOpeningLabel("DOOR",mx,mz,"#92400e"); }
    function drawWindow(w){ const a=w.segment[0], b=w.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5, len=Math.hypot(b[0]-a[0],b[1]-a[1]); const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]); const panel=new THREE.Mesh(new THREE.BoxGeometry(len,.12,.07), makeMat(0x38bdf8,.88,.25)); panel.position.set(mx,1.15,mz); panel.rotation.y=-yaw; scene.add(panel); addOpeningLabel("WINDOW",mx,mz,"#0369a1"); }
    (DATA.doors||[]).forEach(drawDoor); (DATA.windows||[]).forEach(drawWindow);

    const existingGroup = new THREE.Group(), methodGroup = new THREE.Group(), overlayGroup = new THREE.Group(); scene.add(existingGroup, methodGroup, overlayGroup);
    function addDisk(x,z,color,radius=.48,opacity=.55){ const m=new THREE.Mesh(new THREE.RingGeometry(radius*.72,radius,48), new THREE.MeshBasicMaterial({color,transparent:true,opacity,side:THREE.DoubleSide})); m.rotation.x=-Math.PI/2; m.position.set(x,.018,z); overlayGroup.add(m); return m; }
    function addLabel(text,x,z,color="#0f172a"){ const c=document.createElement("canvas"); c.width=640; c.height=128; const ctx=c.getContext("2d"); ctx.fillStyle="rgba(255,255,255,.88)"; ctx.strokeStyle="rgba(148,163,184,.75)"; ctx.lineWidth=4; ctx.roundRect(16,20,608,88,24); ctx.fill(); ctx.stroke(); ctx.fillStyle=color; ctx.font="700 34px Segoe UI, Arial"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(text,320,64); const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true})); s.position.set(x,1.25,z); s.scale.set(1.7,.34,1); overlayGroup.add(s); return s; }
    function enhanceMaterial(mat, tint=null, opacity=null){ if(!mat) return; mat.side=THREE.DoubleSide; if("roughness" in mat) mat.roughness=Math.min(.78,Math.max(.36,mat.roughness??.55)); if("metalness" in mat) mat.metalness=Math.min(.12,mat.metalness??0); if(tint && mat.color) mat.color.lerp(new THREE.Color(tint),.30); if(opacity!==null){ mat.transparent=true; mat.opacity=opacity; } mat.needsUpdate=true; }
    function normalizeLoadedObject(object,spec){ const box=new THREE.Box3().setFromObject(object), size=new THREE.Vector3(), ctr=new THREE.Vector3(); box.getSize(size); box.getCenter(ctr); object.position.sub(ctr); object.scale.set(spec.size[0]/Math.max(size.x,1e-6),spec.size[1]/Math.max(size.y,1e-6),spec.size[2]/Math.max(size.z,1e-6)); const box2=new THREE.Box3().setFromObject(object); object.position.y-=box2.min.y; }
    function applyPose(object,pose){ object.position.x=pose.x; object.position.z=pose.z; object.rotation.y=-THREE.MathUtils.degToRad(pose.yaw||0); }
    function loadObj(spec, opts={}){ return new Promise((resolve,reject)=>{ new MTLLoader().setPath(spec.asset_path+"/").load("model.mtl",(materials)=>{ materials.preload(); for(const k of Object.keys(materials.materials)) enhanceMaterial(materials.materials[k],opts.tint,opts.opacity); new OBJLoader().setMaterials(materials).setPath(spec.asset_path+"/").load(spec.obj_name,(object)=>{ object.traverse((child)=>{ if(child.isMesh){ child.castShadow=true; child.receiveShadow=true; if(Array.isArray(child.material)) child.material.forEach(m=>enhanceMaterial(m,opts.tint,opts.opacity)); else enhanceMaterial(child.material,opts.tint,opts.opacity); }}); normalizeLoadedObject(object,spec); resolve(object);},undefined,reject);},undefined,reject);});}
    for(const spec of DATA.existing){ loadObj(spec,{opacity:.94}).then((obj)=>{ applyPose(obj,spec.pose); existingGroup.add(obj); }); }

    let targetTemplate = null;
    loadObj(DATA.target,{tint:0x22c55e}).then((obj)=>{ targetTemplate=obj; setMethod(0); });
    function clearGroup(g){ while(g.children.length) g.remove(g.children[0]); }
    function setMethod(idx){
      document.querySelectorAll("button").forEach((b,i)=>{ b.classList.toggle("active",i===idx); if(i===idx) b.style.background=DATA.methods[i].color; });
      clearGroup(methodGroup); clearGroup(overlayGroup);
      const m=DATA.methods[idx];
      document.querySelector("#methodBox").innerHTML = `<div class="methodTitle" style="color:${m.color}">${m.label}</div><div>${m.desc}</div><div class="meta">x=${m.pose.x.toFixed(3)}, z=${m.pose.z.toFixed(3)}, yaw=${m.pose.yaw.toFixed(1)}° ${m.score_text || ""}</div>`;
      if(!targetTemplate) return;
      const obj=targetTemplate.clone(true);
      obj.traverse((child)=>{ if(child.isMesh){ child.material=child.material.clone(); enhanceMaterial(child.material, new THREE.Color(m.color).getHex(), .94); }});
      applyPose(obj,m.pose); methodGroup.add(obj); addDisk(m.pose.x,m.pose.z,new THREE.Color(m.color).getHex(),.54,.62); addLabel(m.short,m.pose.x,m.pose.z,m.color);
    }
    const buttons=document.querySelector("#buttons");
    DATA.methods.forEach((m,i)=>{ const b=document.createElement("button"); b.textContent=m.short; b.addEventListener("click",()=>setMethod(i)); buttons.appendChild(b); });
    window.addEventListener("resize",()=>{ camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth,window.innerHeight); });
    function animate(){ controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate); } animate();
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction(path: Path, method: str, case_id: str) -> Dict[str, Any]:
    data = load_json(path)
    entries = (data.get(method) or {}).get(case_id) or []
    placed = [e for e in entries if e.get("status", "placed") == "placed"]
    if not placed and not entries:
        raise ValueError(f"No prediction for {method} in {path}")
    return placed[0] if placed else entries[0]


def method_rows(case_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in METHOD_SPECS:
        pred = read_prediction(spec["path"], spec["method"], case_id)
        pos = pred.get("position") or {}
        score_bits = []
        if pred.get("score") is not None:
            score_bits.append(f"score={float(pred['score']):.3f}")
        if pred.get("human_aligned_score") is not None:
            score_bits.append(f"human={float(pred['human_aligned_score']):.3f}")
        rows.append(
            {
                "label": spec["label"],
                "short": spec["short"],
                "color": spec["color"],
                "desc": spec["desc"],
                "pose": {
                    "x": float(pos.get("x", 0.0)),
                    "z": float(pos.get("z", 0.0)),
                    "yaw": float(pred.get("rotation_y", 0.0)),
                },
                "score_text": " · ".join(score_bits),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case_id", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_scene_data(args.case_id, "constraint_solver", RERANK_PREDICTIONS, out_dir)
    data["methods"] = method_rows(args.case_id)
    html = (
        HTML.replace("__CASE_ID__", data["case_id"])
        .replace("__ROOM__", data["room_type"])
        .replace("__TARGET__", data["target_category"])
        .replace("__INTENT__", data["intent"])
        .replace("__SCENE_JSON__", json.dumps(data, ensure_ascii=False))
    )
    (out_dir / "scene_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
