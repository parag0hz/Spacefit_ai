from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_dataset_eda")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
REPORT_PATH = ROOT / "DATASET_EDA_SUMMARY.md"
EXT_CHART_PATH = ROOT / "dataset_eda_extension_distribution.png"
SUBFOLDER_CHART_PATH = ROOT / "dataset_eda_subfolder_distribution.png"


def format_int(value: int) -> str:
    return f"{value:,}"


def short_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def detect_folder_meaning(name: str) -> str:
    mapping = {
        "3D-FRONT": "Room-level indoor scene JSONs; likely the main furnished room geometry/source scenes used for benchmark construction.",
        "3D-FRONT-texture": "Texture/material assets plus texture metadata for 3D-FRONT scenes.",
        "3D-FUTURE-model": "Furniture asset library with per-asset meshes, textures, preview images, and metadata.",
        "3D-FUTURE-model-part1": "Appears to be a partial shard/copy of 3D-FUTURE assets. Exact role is unclear from folder name alone.",
        "3D-FUTURE-scene": "Rendered scene images, id maps, and GT annotation JSONs for train/test splits.",
        "roomplan": "Small custom/example room JSON files.",
        "scannetpp": "ScanNet++ scene scans, semantic meshes, split files, and metadata.",
        "test_asset_dir": "Currently empty; likely a temporary/test asset folder.",
    }
    return mapping.get(name, "Folder meaning uncertain from lightweight scan.")


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def collect_scan() -> dict[str, Any]:
    ext_counts: Counter[str] = Counter()
    top_folder_counts: Counter[str] = Counter()
    top_folder_examples: dict[str, list[str]] = defaultdict(list)
    ext_examples: dict[str, list[str]] = defaultdict(list)
    total_files = 0

    for path in DATASET_ROOT.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        rel = path.relative_to(DATASET_ROOT)
        top = rel.parts[0] if rel.parts else "."
        suffix = "".join(path.suffixes).lower() if path.suffixes else "[no_ext]"

        top_folder_counts[top] += 1
        ext_counts[suffix] += 1

        if len(top_folder_examples[top]) < 3:
            top_folder_examples[top].append(str(rel))
        if len(ext_examples[suffix]) < 3:
            ext_examples[suffix].append(str(rel))

    top_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    folder_details: list[dict[str, Any]] = []
    for folder in top_dirs:
        file_count = 0
        dir_count = 0
        for child in folder.rglob("*"):
            if child.is_file():
                file_count += 1
            elif child.is_dir():
                dir_count += 1
        child_dirs = sorted([p.name for p in folder.iterdir() if p.is_dir()])[:8]
        folder_details.append(
            {
                "name": folder.name,
                "file_count": file_count,
                "dir_count": dir_count,
                "child_dirs": child_dirs,
                "meaning": detect_folder_meaning(folder.name),
                "examples": top_folder_examples.get(folder.name, []),
            }
        )

    benchmark: dict[str, Any] = {}

    front_dir = DATASET_ROOT / "3D-FRONT"
    benchmark["front_scene_json_count"] = sum(1 for _ in front_dir.glob("*.json"))
    sample_front = next(front_dir.glob("*.json"))
    front_sample = load_json(sample_front)
    benchmark["front_sample_keys"] = list(front_sample.keys())[:14]

    future_dir = DATASET_ROOT / "3D-FUTURE-model"
    future_assets = [p for p in future_dir.iterdir() if p.is_dir()]
    benchmark["future_asset_count"] = len(future_assets)
    benchmark["future_avg_files_per_asset"] = (
        top_folder_counts["3D-FUTURE-model"] / len(future_assets) if future_assets else 0.0
    )
    model_info = load_json(future_dir / "model_info.json")
    benchmark["future_model_info_count"] = len(model_info)
    future_cat_counts = Counter()
    for row in model_info:
        category = row.get("category")
        if category:
            future_cat_counts[str(category)] += 1
    benchmark["future_top_categories"] = future_cat_counts.most_common(10)

    future_part1_dir = DATASET_ROOT / "3D-FUTURE-model-part1"
    benchmark["future_part1_asset_count"] = sum(1 for p in future_part1_dir.iterdir() if p.is_dir())

    future_scene_dir = DATASET_ROOT / "3D-FUTURE-scene"
    train_gt = load_json(future_scene_dir / "GT" / "train_set.json")
    test_gt = load_json(future_scene_dir / "GT" / "test_set.json")
    benchmark["future_scene_train_images"] = len(train_gt.get("images", []))
    benchmark["future_scene_train_annotations"] = len(train_gt.get("annotations", []))
    benchmark["future_scene_train_categories"] = len(train_gt.get("categories", []))
    benchmark["future_scene_test_images"] = len(test_gt.get("images", []))
    benchmark["future_scene_test_annotations"] = len(test_gt.get("annotations", []))

    texture_info = load_json(DATASET_ROOT / "3D-FRONT-texture" / "texture_info.json")
    benchmark["front_texture_count"] = len(texture_info)
    texture_cat_counts = Counter()
    for row in texture_info:
        category = row.get("category")
        if category:
            texture_cat_counts[str(category)] += 1
    benchmark["front_texture_top_categories"] = texture_cat_counts.most_common(8)

    scannet_data_dir = DATASET_ROOT / "scannetpp" / "data"
    benchmark["scannet_scene_count"] = sum(1 for p in scannet_data_dir.iterdir() if p.is_dir())
    benchmark["scannet_avg_files_per_scene"] = (
        top_folder_counts["scannetpp"] / benchmark["scannet_scene_count"] if benchmark["scannet_scene_count"] else 0.0
    )

    return {
        "total_files": total_files,
        "ext_counts": ext_counts,
        "ext_examples": ext_examples,
        "top_folder_counts": top_folder_counts,
        "folder_details": folder_details,
        "benchmark": benchmark,
    }


def make_extension_chart(ext_counts: Counter[str]) -> None:
    top = ext_counts.most_common(10)
    labels = [k for k, _ in top]
    values = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title("Dataset File Extension Distribution")
    ax.set_ylabel("File Count")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_int(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(EXT_CHART_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_subfolder_chart(folder_details: list[dict[str, Any]]) -> None:
    top = sorted(folder_details, key=lambda item: item["file_count"], reverse=True)[:8]
    labels = [item["name"] for item in top]
    values = [item["file_count"] for item in top]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(labels, values, color="#F58518")
    ax.set_title("Dataset Top-Level Folder File Counts")
    ax.set_ylabel("File Count")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_int(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(SUBFOLDER_CHART_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(data: dict[str, Any]) -> str:
    ext_counts: Counter[str] = data["ext_counts"]
    folder_details: list[dict[str, Any]] = data["folder_details"]
    benchmark = data["benchmark"]

    important_exts = [".json", ".jpg", ".png", ".obj", ".mtl", ".ply", ".npy"]
    important_rows = []
    for ext in important_exts:
        if ext == ".ply":
            matching = sum(v for k, v in ext_counts.items() if k.endswith(".ply"))
            examples = []
            for k, vals in data["ext_examples"].items():
                if k.endswith(".ply"):
                    examples.extend(vals)
            important_rows.append((ext, matching, examples[:3]))
        else:
            important_rows.append((ext, ext_counts.get(ext, 0), data["ext_examples"].get(ext, [])))

    lines: list[str] = []
    lines.append("# Dataset EDA Summary")
    lines.append("")
    lines.append(f"Dataset root: `{DATASET_ROOT}`")
    lines.append("")
    lines.append("This is a lightweight, PPT-oriented scan of the local dataset folder. It summarizes folder structure, file types, and a few benchmark-relevant counts without doing any heavy per-file analysis.")
    lines.append("")
    lines.append("## 1. Top-Level Folder Structure")
    lines.append("")
    lines.append("| Folder | Files | Subdirs | What It Seems To Contain | Example Contents |")
    lines.append("|---|---:|---:|---|---|")
    for item in folder_details:
        examples = "<br>".join(item["examples"][:2]) if item["examples"] else "-"
        lines.append(
            f"| `{item['name']}` | {format_int(item['file_count'])} | {format_int(item['dir_count'])} | {item['meaning']} | {examples} |"
        )

    lines.append("")
    lines.append("## 2. File Format / Extension Summary")
    lines.append("")
    lines.append(f"Total files scanned: **{format_int(data['total_files'])}**")
    lines.append("")
    lines.append("| Extension | Count | Example Paths |")
    lines.append("|---|---:|---|")
    for ext, count in ext_counts.most_common(12):
        examples = "<br>".join(data["ext_examples"].get(ext, [])[:3])
        lines.append(f"| `{ext}` | {format_int(count)} | {examples} |")

    lines.append("")
    lines.append("## 3. Basic File Count Summary")
    lines.append("")
    lines.append("### Top-level folders by file count")
    lines.append("")
    for item in sorted(folder_details, key=lambda x: x["file_count"], reverse=True):
        lines.append(f"- `{item['name']}`: {format_int(item['file_count'])} files")

    lines.append("")
    lines.append("### Major file types")
    lines.append("")
    for ext, count in ext_counts.most_common(8):
        lines.append(f"- `{ext}`: {format_int(count)} files")

    lines.append("")
    lines.append("## 4. Representative Examples by Role")
    lines.append("")
    lines.append("- Room geometry / room scenes:")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FRONT' / '0003d406-5f27-4bbf-94cd-1cff7c310ba1.json')}`")
    lines.append(f"  `{short_rel(DATASET_ROOT / 'roomplan' / 'Livingroom.json')}`")
    lines.append("- Furniture assets:")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FUTURE-model' / '0fee7a5a-5478-4c0d-af5f-e6dd0159bf6d' / 'raw_model.obj')}`")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FUTURE-model' / '0fee7a5a-5478-4c0d-af5f-e6dd0159bf6d' / 'image.jpg')}`")
    lines.append("- Images / renders:")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FUTURE-scene' / 'train' / 'image' / '0000403.jpg')}`")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FUTURE-scene' / 'train' / 'idmap' / '0002865.png')}`")
    lines.append("- Metadata / annotations:")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FUTURE-model' / 'model_info.json')}`")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FUTURE-scene' / 'GT' / 'train_set.json')}`")
    lines.append(f"  `{short_rel(DATASET_ROOT / '3D-FRONT-texture' / 'texture_info.json')}`")
    lines.append(f"  `{short_rel(DATASET_ROOT / 'scannetpp' / 'metadata' / 'scene_types.json')}`")

    lines.append("")
    lines.append("## 5. Simple Benchmark-Relevant Summary")
    lines.append("")
    lines.append("| Item | Count / Value | Note |")
    lines.append("|---|---:|---|")
    lines.append(f"| 3D-FRONT room scene JSONs | {format_int(benchmark['front_scene_json_count'])} | Strong candidate for the main furnished room source used in the placement benchmark |")
    lines.append(f"| 3D-FUTURE furniture assets | {format_int(benchmark['future_asset_count'])} | Counted as per-asset directories in `3D-FUTURE-model/` |")
    lines.append(f"| Avg. files per 3D-FUTURE asset | {benchmark['future_avg_files_per_asset']:.1f} | Roughly mesh + texture + preview bundle per asset |")
    lines.append(f"| 3D-FUTURE scene train renders | {format_int(benchmark['future_scene_train_images'])} | RGB images in `3D-FUTURE-scene/train/image` |")
    lines.append(f"| 3D-FUTURE scene test renders | {format_int(benchmark['future_scene_test_images'])} | RGB images in `3D-FUTURE-scene/test/image` |")
    lines.append(f"| 3D-FUTURE train annotations | {format_int(benchmark['future_scene_train_annotations'])} | From `GT/train_set.json` |")
    lines.append(f"| 3D-FUTURE test annotations | {format_int(benchmark['future_scene_test_annotations'])} | From `GT/test_set.json` |")
    lines.append(f"| 3D-FRONT texture entries | {format_int(benchmark['front_texture_count'])} | From `3D-FRONT-texture/texture_info.json` |")
    lines.append(f"| ScanNet++ scene folders | {format_int(benchmark['scannet_scene_count'])} | Separate scanned-scene dataset, likely auxiliary rather than core benchmark data |")
    lines.append(f"| Avg. files per ScanNet++ scene | {benchmark['scannet_avg_files_per_scene']:.1f} | Very lightweight scan-level estimate |")

    lines.append("")
    lines.append("### Easy category metadata")
    lines.append("")
    lines.append("Top 10 3D-FUTURE categories from `model_info.json`:")
    lines.append("")
    for category, count in benchmark["future_top_categories"]:
        lines.append(f"- {category}: {format_int(count)}")

    lines.append("")
    lines.append("Top 8 texture categories from `3D-FRONT-texture/texture_info.json`:")
    lines.append("")
    for category, count in benchmark["front_texture_top_categories"]:
        lines.append(f"- {category}: {format_int(count)}")

    lines.append("")
    lines.append("## 6. PPT-Friendly Summary Table")
    lines.append("")
    lines.append("| Source / Folder | Main Formats | Rough Count | Intended Use |")
    lines.append("|---|---|---:|---|")
    lines.append(f"| `3D-FRONT` | `.json` | {format_int(benchmark['front_scene_json_count'])} scenes | Furnished room geometry / furniture layout source scenes |")
    lines.append(f"| `3D-FUTURE-model` | `.obj`, `.mtl`, `.png`, `.jpg`, `.json` | {format_int(benchmark['future_asset_count'])} assets | Furniture mesh/texture asset library |")
    lines.append(f"| `3D-FUTURE-scene` | `.jpg`, `.png`, `.json` | {format_int(data['top_folder_counts']['3D-FUTURE-scene'])} files | Rendered scene images + id maps + annotations |")
    lines.append(f"| `3D-FRONT-texture` | `.json`, texture assets | {format_int(benchmark['front_texture_count'])} texture entries | Material / texture metadata |")
    lines.append(f"| `scannetpp` | `.ply`, `.json`, `.txt`, `.csv` | {format_int(benchmark['scannet_scene_count'])} scene dirs | Real scanned 3D scenes and semantic metadata |")
    lines.append(f"| `roomplan` | `.json` | {format_int(data['top_folder_counts']['roomplan'])} files | Small custom room examples |")

    lines.append("")
    lines.append("## 7. Charts")
    lines.append("")
    lines.append(f"- Extension distribution: `{short_rel(EXT_CHART_PATH)}`")
    lines.append(f"- Top-level folder file counts: `{short_rel(SUBFOLDER_CHART_PATH)}`")
    lines.append("")
    lines.append("## Notes / Caveats")
    lines.append("")
    lines.append("- `3D-FUTURE-model-part1` looks like a partial shard or duplicated subset of the 3D-FUTURE asset library; its exact role was not inferred deeply.")
    lines.append("- `3D-FRONT` seems to be the most directly benchmark-relevant room-scene source for the current indoor furniture placement project.")
    lines.append("- This scan is intentionally lightweight and does not validate every annotation schema or inspect every large JSON deeply.")

    return "\n".join(lines) + "\n"


def main() -> None:
    data = collect_scan()
    make_extension_chart(data["ext_counts"])
    make_subfolder_chart(data["folder_details"])
    report = build_report(data)
    REPORT_PATH.write_text(report)
    print(json.dumps(
        {
            "report": str(REPORT_PATH),
            "charts": [str(EXT_CHART_PATH), str(SUBFOLDER_CHART_PATH)],
            "total_files": data["total_files"],
            "top_folders": data["top_folder_counts"].most_common(8),
            "top_extensions": data["ext_counts"].most_common(10),
            "benchmark": data["benchmark"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
