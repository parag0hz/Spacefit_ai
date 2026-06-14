"""Create a Three.js WebGL viewer for one 3D-FUTURE OBJ asset.

This is a lightweight high-quality alternative when Blender is unavailable.
It copies OBJ/MTL/texture files into an output folder and writes an HTML viewer.

Example:
    python spacefit_v2/scripts/make_3dfuture_threejs_viewer.py \
      --asset_dir dataset/3D-FUTURE-model-part1/0a1e2fb3-3345-4483-a6ae-049923d15934 \
      --out_dir spacefit_v2/results/3d_qualitative_renders/threejs_asset_0a1e2fb3
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3D-FUTURE Asset Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      background: #f5f7fb;
      color: #111827;
    }}
    body {{
      margin: 0;
      overflow: hidden;
      background: radial-gradient(circle at 30% 20%, #ffffff 0%, #eef2f7 55%, #e2e8f0 100%);
    }}
    #canvas {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    .panel {{
      position: fixed;
      left: 24px;
      top: 20px;
      max-width: 460px;
      padding: 14px 16px;
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
    }}
    .title {{
      margin: 0 0 6px;
      font-size: 16px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
      color: #475569;
      word-break: break-all;
    }}
    .hint {{
      position: fixed;
      right: 22px;
      bottom: 18px;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.78);
      color: white;
      font-size: 12px;
      box-shadow: 0 14px 35px rgba(15, 23, 42, 0.22);
    }}
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <section class="panel">
    <h1 class="title">3D-FUTURE Asset Viewer</h1>
    <p class="meta">Asset: {asset_name}</p>
    <p class="meta">OBJ: {obj_name} / MTL: {mtl_name}</p>
  </section>
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

    const canvas = document.querySelector("#canvas");
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: false }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f8fb);
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

    const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(3.2, 2.3, 3.7);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(0, 0.45, 0);

    const hemi = new THREE.HemisphereLight(0xffffff, 0xb7c1cc, 2.0);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(4, 5, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 18;
    key.shadow.camera.left = -5;
    key.shadow.camera.right = 5;
    key.shadow.camera.top = 5;
    key.shadow.camera.bottom = -5;
    scene.add(key);

    const rim = new THREE.DirectionalLight(0xc7ddff, 1.25);
    rim.position.set(-4, 3, -4);
    scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(3.4, 96),
      new THREE.MeshStandardMaterial({{ color: 0xe8edf3, roughness: 0.72, metalness: 0.0 }})
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.02;
    floor.receiveShadow = true;
    scene.add(floor);

    const contact = new THREE.Mesh(
      new THREE.CircleGeometry(1.25, 96),
      new THREE.MeshBasicMaterial({{ color: 0x0f172a, transparent: true, opacity: 0.08, depthWrite: false }})
    );
    contact.rotation.x = -Math.PI / 2;
    contact.position.y = 0.002;
    scene.add(contact);

    function normalizeObject(root) {{
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const maxDim = Math.max(size.x, size.y, size.z);
      root.position.sub(center);
      root.scale.multiplyScalar(2.1 / Math.max(maxDim, 1e-6));

      const newBox = new THREE.Box3().setFromObject(root);
      root.position.y -= newBox.min.y;
      controls.target.copy(newBox.getCenter(new THREE.Vector3()));
      controls.target.y *= 0.72;
      controls.update();
    }}

    function enhanceMaterial(mat) {{
      if (!mat) return;
      mat.side = THREE.DoubleSide;
      if ("roughness" in mat) mat.roughness = Math.min(0.78, Math.max(0.38, mat.roughness ?? 0.55));
      if ("metalness" in mat) mat.metalness = Math.min(0.2, mat.metalness ?? 0.0);
      mat.needsUpdate = true;
    }}

    const manager = new THREE.LoadingManager();
    manager.onError = (url) => console.warn("Failed to load", url);

    new MTLLoader(manager)
      .setPath("./")
      .load("{mtl_name}", (materials) => {{
        materials.preload();
        for (const key of Object.keys(materials.materials)) enhanceMaterial(materials.materials[key]);
        const loader = new OBJLoader(manager).setMaterials(materials).setPath("./");
        loader.load("{obj_name}", (object) => {{
          object.traverse((child) => {{
            if (child.isMesh) {{
              child.castShadow = true;
              child.receiveShadow = true;
              if (Array.isArray(child.material)) child.material.forEach(enhanceMaterial);
              else enhanceMaterial(child.material);
            }}
          }});
          normalizeObject(object);
          scene.add(object);
        }});
      }});

    function resize() {{
      const w = window.innerWidth;
      const h = window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--use_raw", action="store_true")
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    asset_dir = (ROOT / args.asset_dir).resolve() if not Path(args.asset_dir).is_absolute() else Path(args.asset_dir)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_name = "raw_model.obj" if args.use_raw else "normalized_model.obj"
    obj_path = asset_dir / obj_name
    mtl_path = asset_dir / "model.mtl"
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)
    if not mtl_path.exists():
        raise FileNotFoundError(mtl_path)

    for name in [obj_name, "model.mtl", "texture.png", "image.jpg"]:
        copy_if_exists(asset_dir / name, out_dir / name)

    html = HTML_TEMPLATE.format(asset_name=asset_dir.name, obj_name=obj_name, mtl_name="model.mtl")
    out_html = out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(out_html)
    print("Serve this folder with: python -m http.server 8899 --directory", out_dir)


if __name__ == "__main__":
    main()
