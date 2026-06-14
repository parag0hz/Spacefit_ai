"""Create a Three.js viewer that visualizes the VLM judge decision process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.scripts.make_3dfront_threejs_scene_viewer import (  # noqa: E402
    DEFAULT_PREDICTIONS,
    build_scene_data,
)


DEFAULT_JUDGE = ROOT / "spacefit_v2" / "results" / "vlm_quality_judge_full_gpt4o" / "vlm_judge_predictions.json"


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLM Judge Process Viewer</title>
  <style>
    body { margin:0; overflow:hidden; font-family:Inter,"Segoe UI",Arial,sans-serif; background:#f6f8fb; color:#111827; }
    #canvas { width:100vw; height:100vh; display:block; }
    .panel { position:fixed; left:22px; top:18px; width:min(560px,calc(100vw - 44px)); padding:16px; border:1px solid rgba(148,163,184,.45); border-radius:16px; background:rgba(255,255,255,.88); backdrop-filter:blur(14px); box-shadow:0 22px 60px rgba(15,23,42,.14); }
    h1 { margin:0 0 8px; font-size:18px; line-height:1.25; }
    .meta { margin:3px 0; font-size:12px; color:#475569; line-height:1.45; }
    .intent { margin-top:10px; padding:10px 11px; border-radius:11px; background:#f8fafc; border:1px solid #e2e8f0; font-size:12px; line-height:1.45; color:#334155; }
    .judge { margin-top:11px; padding:11px; border-radius:12px; background:#fff7ed; border:1px solid #fed7aa; }
    .judge.good { background:#f0fdf4; border-color:#bbf7d0; }
    .stageTitle { margin:0 0 6px; font-size:14px; font-weight:850; }
    .reason { margin:0; font-size:12px; line-height:1.45; color:#334155; }
    .scoreGrid { margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .score { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:6px 8px; border-radius:9px; background:#f8fafc; border:1px solid #e2e8f0; font-size:11px; }
    .score b { font-size:12px; }
    .pass { color:#15803d; }
    .fail { color:#dc2626; }
    .controls { position:fixed; left:22px; bottom:20px; right:22px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:12px 14px; border:1px solid rgba(148,163,184,.42); border-radius:18px; background:rgba(255,255,255,.88); backdrop-filter:blur(12px); box-shadow:0 18px 45px rgba(15,23,42,.16); }
    button { border:0; border-radius:999px; padding:9px 12px; background:#e2e8f0; color:#0f172a; font-size:12px; font-weight:760; cursor:pointer; }
    button.active { color:white; background:#2563eb; }
    .hint { position:fixed; right:22px; top:20px; padding:10px 12px; border-radius:999px; background:rgba(15,23,42,.78); color:white; font-size:12px; }
    .legend { position:fixed; right:22px; bottom:20px; width:270px; padding:12px 13px; border-radius:14px; background:rgba(255,255,255,.88); border:1px solid rgba(148,163,184,.45); box-shadow:0 16px 40px rgba(15,23,42,.13); font-size:12px; line-height:1.45; color:#334155; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <section class="panel">
    <h1>VLM Judge 판단 과정 시각화</h1>
    <p class="meta">Case: __CASE_ID__</p>
    <p class="meta">Room: __ROOM__ · Target: __TARGET__</p>
    <div class="intent"><b>User intent</b><br>__INTENT__</div>
    <div id="judgeBox" class="judge"></div>
  </section>
  <div class="hint">drag: rotate · wheel: zoom · right-drag: pan</div>
  <div class="legend">
    <div><span class="dot" style="background:#22c55e"></span>목표 가구 배치 결과</div>
    <div><span class="dot" style="background:#f97316"></span>문/접근성 확인 영역</div>
    <div><span class="dot" style="background:#ef4444"></span>VLM이 문제로 본 관계/방향</div>
    <div><span class="dot" style="background:#3b82f6"></span>장면 전체 자연스러움 판단</div>
  </div>
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
    const J = DATA.vlm_judgment;
    const canvas = document.querySelector("#canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias:true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.06;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f8fb);
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), .04).texture;
    const center = new THREE.Vector3(DATA.center[0],0,DATA.center[1]), span = DATA.span;
    const camera = new THREE.PerspectiveCamera(39, window.innerWidth/window.innerHeight, .01, 200);
    camera.position.set(center.x+span*.68, span*.62, center.z+span*.82);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = .055; controls.target.set(center.x,.55,center.z); controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff,0xb8c2cc,1.9));
    const key = new THREE.DirectionalLight(0xffffff,3.4); key.position.set(center.x+span*.35,span*.9,center.z+span*.25); key.castShadow=true; key.shadow.mapSize.set(2048,2048); key.shadow.camera.left=-span; key.shadow.camera.right=span; key.shadow.camera.top=span; key.shadow.camera.bottom=-span; scene.add(key);
    const rim = new THREE.DirectionalLight(0xdbeafe,1.1); rim.position.set(center.x-span*.8,span*.5,center.z-span*.8); scene.add(rim);

    const existingGroup = new THREE.Group(), targetGroup = new THREE.Group(), overlayGroup = new THREE.Group();
    scene.add(existingGroup, targetGroup, overlayGroup);
    function makeMat(color, opacity=1, roughness=.72){ return new THREE.MeshStandardMaterial({color,roughness,metalness:.02,transparent:opacity<1,opacity,side:THREE.DoubleSide}); }
    const floorShape = new THREE.Shape(); DATA.floor.forEach((p,i)=>{ if(i===0) floorShape.moveTo(p[0],p[1]); else floorShape.lineTo(p[0],p[1]); }); floorShape.closePath();
    const floorGeom = new THREE.ShapeGeometry(floorShape); floorGeom.rotateX(Math.PI/2); const floor = new THREE.Mesh(floorGeom, makeMat(0xe8edf3)); floor.receiveShadow=true; scene.add(floor);
    const wallMat = makeMat(0xf8fafc,.34), edgeMat = new THREE.LineBasicMaterial({color:0x334155});
    for(let i=0;i<DATA.floor.length;i++){ const a=DATA.floor[i], b=DATA.floor[(i+1)%DATA.floor.length], len=Math.hypot(b[0]-a[0],b[1]-a[1]); if(len<.08) continue; const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],0,a[1]),new THREE.Vector3(b[0],0,b[1]),new THREE.Vector3(b[0],2.6,b[1]),new THREE.Vector3(a[0],2.6,a[1])]); g.setIndex([0,1,2,0,2,3]); g.computeVertexNormals(); scene.add(new THREE.Mesh(g,wallMat)); scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],.025,a[1]),new THREE.Vector3(b[0],.025,b[1])]), edgeMat)); }

    function addLabel(text,x,z,color="#0f172a",y=1.25){ const c=document.createElement("canvas"); c.width=760; c.height=150; const ctx=c.getContext("2d"); ctx.fillStyle="rgba(255,255,255,.91)"; ctx.strokeStyle="rgba(148,163,184,.75)"; ctx.lineWidth=4; ctx.roundRect(16,22,728,104,28); ctx.fill(); ctx.stroke(); ctx.fillStyle=color; ctx.font="800 38px Segoe UI, Arial"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(text,380,75); const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true})); s.position.set(x,y,z); s.scale.set(1.9,.38,1); overlayGroup.add(s); return s; }
    function addDisk(x,z,color,radius=.58,opacity=.42){ const m=new THREE.Mesh(new THREE.RingGeometry(radius*.7,radius,64), new THREE.MeshBasicMaterial({color,transparent:true,opacity,side:THREE.DoubleSide})); m.rotation.x=-Math.PI/2; m.position.set(x,.03,z); overlayGroup.add(m); return m; }
    function addArrow(x,z,yaw,color=0xef4444){ const dir=new THREE.Vector3(Math.sin(-THREE.MathUtils.degToRad(yaw)),0,Math.cos(-THREE.MathUtils.degToRad(yaw))); const ar=new THREE.ArrowHelper(dir,new THREE.Vector3(x,.42,z),1.05,color,.28,.18); overlayGroup.add(ar); return ar; }
    function addOpeningLabel(text,x,z,color){ addLabel(text,x,z,color,.9); }
    function drawDoor(d){ const a=d.segment[0], b=d.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5, len=Math.hypot(b[0]-a[0],b[1]-a[1]); const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]); const panel=new THREE.Mesh(new THREE.BoxGeometry(len,.08,.055), makeMat(0xb45309,.92)); panel.position.set(mx,.05,mz); panel.rotation.y=-yaw; panel.castShadow=true; scene.add(panel); const clear=new THREE.Mesh(new THREE.CircleGeometry(Math.max(.55,len*.78),48,0,Math.PI*.62), new THREE.MeshBasicMaterial({color:0xf97316,transparent:true,opacity:.18,side:THREE.DoubleSide})); clear.rotation.x=-Math.PI/2; clear.rotation.z=-yaw; clear.position.set(a[0],.028,a[1]); scene.add(clear); addOpeningLabel("DOOR",mx,mz,"#92400e"); }
    function drawWindow(w){ const a=w.segment[0], b=w.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5, len=Math.hypot(b[0]-a[0],b[1]-a[1]); const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]); const panel=new THREE.Mesh(new THREE.BoxGeometry(len,.12,.07), makeMat(0x38bdf8,.88)); panel.position.set(mx,1.15,mz); panel.rotation.y=-yaw; scene.add(panel); addOpeningLabel("WINDOW",mx,mz,"#0369a1"); }
    (DATA.doors||[]).forEach(drawDoor); (DATA.windows||[]).forEach(drawWindow);

    function enhanceMaterial(mat, tint=null, opacity=null){ if(!mat) return; mat.side=THREE.DoubleSide; if("roughness" in mat) mat.roughness=Math.min(.78,Math.max(.36,mat.roughness??.55)); if("metalness" in mat) mat.metalness=Math.min(.12,mat.metalness??0); if(tint && mat.color) mat.color.lerp(new THREE.Color(tint),.28); if(opacity!==null){ mat.transparent=true; mat.opacity=opacity; } mat.needsUpdate=true; }
    function normalizeLoadedObject(object,spec){ const box=new THREE.Box3().setFromObject(object), size=new THREE.Vector3(), ctr=new THREE.Vector3(); box.getSize(size); box.getCenter(ctr); object.position.sub(ctr); object.scale.set(spec.size[0]/Math.max(size.x,1e-6),spec.size[1]/Math.max(size.y,1e-6),spec.size[2]/Math.max(size.z,1e-6)); const box2=new THREE.Box3().setFromObject(object); object.position.y-=box2.min.y; }
    function applyPose(object,pose){ object.position.x=pose.x; object.position.z=pose.z; object.rotation.y=-THREE.MathUtils.degToRad(pose.yaw||0); }
    function loadObj(spec, opts={}){ return new Promise((resolve,reject)=>{ new MTLLoader().setPath(spec.asset_path+"/").load("model.mtl",(materials)=>{ materials.preload(); for(const k of Object.keys(materials.materials)) enhanceMaterial(materials.materials[k],opts.tint,opts.opacity); new OBJLoader().setMaterials(materials).setPath(spec.asset_path+"/").load(spec.obj_name,(object)=>{ object.traverse((child)=>{ if(child.isMesh){ child.castShadow=true; child.receiveShadow=true; if(Array.isArray(child.material)) child.material.forEach(m=>enhanceMaterial(m,opts.tint,opts.opacity)); else enhanceMaterial(child.material,opts.tint,opts.opacity); }}); normalizeLoadedObject(object,spec); resolve(object);},undefined,reject);},undefined,reject);});}
    for(const spec of DATA.existing){ loadObj(spec,{opacity:.94}).then((obj)=>{ applyPose(obj,spec.pose); existingGroup.add(obj); }); }
    let targetObject = null;
    loadObj(DATA.target,{tint:0x22c55e,opacity:.96}).then((obj)=>{ targetObject=obj; applyPose(obj,DATA.after_pose); targetGroup.add(obj); setStage(0); });

    const stages = [
      {label:"1. 장면 입력", title:"VLM 입력: 최종 배치가 포함된 3D 장면", color:"#2563eb", text:"VLM은 사용자 의도와 배치 이미지를 함께 보고 판단한다.", mode:"input"},
      {label:"2. 물리/접근성", title:"Physical validity / Accessibility", color:"#16a34a", text:"충돌, 경계, 문 주변 접근 가능성을 먼저 확인한다.", mode:"physical"},
      {label:"3. 관계/방향", title:"Relation satisfaction / Orientation", color:"#dc2626", text:"목표 가구가 의도한 기준 가구 근처에 있는지, 바라보는 방향이 맞는지 확인한다.", mode:"relation"},
      {label:"4. 자연스러움", title:"Grouping / Overall naturalness", color:"#3b82f6", text:"가구군과 어울리는지, 사람이 보기에도 자연스러운 배치인지 판단한다.", mode:"natural"},
      {label:"5. 최종 판정", title:"VLM verdict", color:"#ea580c", text:"세부 항목 점수를 종합해 quality score와 main issue를 기록한다.", mode:"verdict"},
    ];
    function scoreHtml(name,key){ const v=Number(J[key]||0); return `<div class="score"><span>${name}</span><b class="${v>=1?'pass':'fail'}">${v>=1?'PASS':'FAIL'}</b></div>`; }
    function clearOverlay(){ while(overlayGroup.children.length) overlayGroup.remove(overlayGroup.children[0]); }
    function setStage(i){
      document.querySelectorAll("button").forEach((b,idx)=>b.classList.toggle("active",idx===i));
      clearOverlay();
      const st=stages[i], p=DATA.after_pose;
      const good = Number(J.quality_score||0) >= 7;
      const box=document.querySelector("#judgeBox"); box.classList.toggle("good",good);
      box.innerHTML = `<div class="stageTitle" style="color:${st.color}">${st.title}</div><p class="reason">${st.text}</p><div class="scoreGrid">${scoreHtml("Physical", "physical_validity")}${scoreHtml("Accessibility", "accessibility")}${scoreHtml("Relation", "relation_satisfaction")}${scoreHtml("Orientation", "orientation_naturalness")}${scoreHtml("Grouping", "grouping_naturalness")}${scoreHtml("Overall", "overall_naturalness")}</div><p class="reason" style="margin-top:9px"><b>Quality ${J.quality_score}/10 · Issue: ${J.main_issue}</b><br>${J.reason}</p>`;
      addDisk(p.x,p.z,0x22c55e,.58,.55); addLabel("Target result",p.x,p.z,"#15803d"); addArrow(p.x,p.z,p.yaw,0x22c55e);
      if(st.mode==="physical"){ (DATA.doors||[]).forEach(d=>{ const a=d.segment[0], b=d.segment[1], mx=(a[0]+b[0])*.5, mz=(a[1]+b[1])*.5; addDisk(mx,mz,0xf97316,.65,.35); }); addLabel("collision-free / accessible",p.x,p.z+0.9,"#15803d"); }
      if(st.mode==="relation"){ addLabel("relation / facing check",p.x,p.z+0.95,"#dc2626"); addArrow(p.x,p.z,p.yaw,0xdc2626); }
      if(st.mode==="natural"){ addLabel("grouping naturalness",DATA.center[0],DATA.center[1],"#2563eb",1.55); addDisk(DATA.center[0],DATA.center[1],0x3b82f6,Math.max(.9,DATA.span*.18),.16); }
      if(st.mode==="verdict"){ addLabel(`score ${J.quality_score}/10`,p.x,p.z+1.05, good ? "#15803d" : "#dc2626",1.45); addLabel(J.main_issue,p.x,p.z-1.05, good ? "#15803d" : "#dc2626",1.15); }
    }
    const buttons=document.querySelector("#buttons");
    stages.forEach((s,i)=>{ const b=document.createElement("button"); b.textContent=s.label; b.addEventListener("click",()=>setStage(i)); buttons.appendChild(b); });
    window.addEventListener("resize",()=>{ camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth,window.innerHeight); });
    function animate(){ controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate); } animate();
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_case(judge_path: Path) -> str:
    rows = load_json(judge_path)
    for row in rows:
        judgment = row.get("judgment") or {}
        if judgment.get("main_issue") == "relation_mismatch" and float(judgment.get("quality_score", 10)) <= 3:
            return str(row["case_id"])
    return str(rows[0]["case_id"])


def judge_for_case(judge_path: Path, case_id: str, method: str) -> Dict[str, Any]:
    rows = load_json(judge_path)
    for row in rows:
        if row.get("case_id") == case_id and row.get("method") == method:
            return dict(row.get("judgment") or {})
    raise ValueError(f"No VLM judgment found for {case_id} / {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--method", default="constraint_solver")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--judge_predictions", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--out_dir", default="spacefit_v2/results/3d_qualitative_renders/threejs_vlm_judge_process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = args.predictions if args.predictions.is_absolute() else ROOT / args.predictions
    judge_path = args.judge_predictions if args.judge_predictions.is_absolute() else ROOT / args.judge_predictions
    case_id = args.case_id or pick_case(judge_path)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_scene_data(case_id, args.method, predictions, out_dir)
    data["vlm_judgment"] = judge_for_case(judge_path, case_id, args.method)
    (out_dir / "scene_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    html = (
        HTML.replace("__CASE_ID__", case_id)
        .replace("__ROOM__", data["room_type"])
        .replace("__TARGET__", data["target_category"])
        .replace("__INTENT__", data["intent"])
        .replace("__SCENE_JSON__", json.dumps(data, ensure_ascii=False))
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"case_id={case_id}")
    print(f"output={out_dir / 'index.html'}")
    print(f"quality={data['vlm_judgment'].get('quality_score')} issue={data['vlm_judgment'].get('main_issue')}")


if __name__ == "__main__":
    main()
