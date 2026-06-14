"""논문용 시각화 2종 생성.

표 1: 주요 성능 지표 비교 (테이블 이미지)
그림 3: 실제 배치 결과 예시 (방 top-down)
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
def _setup_korean_font():
    for fname in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic",
                  "Noto Sans CJK KR"]:
        try:
            found = fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
            if "DejaVu" not in found:
                return fname
        except Exception:
            pass
    ttc = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(ttc):
        try:
            from fontTools.ttLib import TTCollection
            tmp = "/tmp/_NotoSansCJK-KR.otf"
            TTCollection(ttc).fonts[1].save(tmp)
            fm.fontManager.addfont(tmp)
            return "Noto Sans CJK KR"
        except Exception:
            pass
    return None

_kr = _setup_korean_font()
if _kr:
    plt.rcParams["font.family"] = _kr
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path("spacefit_v2/results/visualizations/paper_kr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 표 1: 지표 비교 테이블
# ══════════════════════════════════════════════════════════════════════════════
COL_LABELS = ["방법", "CF ↑", "IB ↑", "조건 만족도 ↑", "Success@1 ↑"]
ROW_DATA = [
    ["Heuristic",   "0.956", "0.939", "0.502", "0.188"],
    ["DiffOpt",     "0.928", "0.812", "0.568", "0.199"],
    ["LayoutGPT",   "0.398", "0.635", "0.657", "0.094"],
    ["GPT-Text",    "0.950", "0.884", "0.646", "0.376"],
    ["제안 방법",    "0.961", "0.961", "0.725", "0.503"],
]
BEST = {"CF ↑": "0.961", "IB ↑": "0.961", "조건 만족도 ↑": "0.725", "Success@1 ↑": "0.503"}

fig, ax = plt.subplots(figsize=(9.5, 2.6))
ax.axis("off")
fig.patch.set_facecolor("white")

tbl = ax.table(
    cellText=ROW_DATA,
    colLabels=COL_LABELS,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11.5)
tbl.scale(1.0, 2.1)

HEADER_BG  = "#2471A3"
OURS_BG    = "#EAF4FB"
BEST_BG    = "#D6EAF8"
ALT_BG     = "#F8F9FA"

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#C8C8C8")
    cell.set_linewidth(0.6)
    if row == 0:
        cell.set_facecolor(HEADER_BG)
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        is_ours = (row == len(ROW_DATA))
        metric_col = col >= 1
        is_best = (metric_col and ROW_DATA[row - 1][col] == BEST.get(COL_LABELS[col], ""))

        if is_best:
            cell.set_facecolor(BEST_BG)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#1A5276")
        elif is_ours:
            cell.set_facecolor(OURS_BG)
        elif row % 2 == 0:
            cell.set_facecolor(ALT_BG)
        else:
            cell.set_facecolor("white")

        if is_ours and col == 0:
            cell.get_text().set_fontweight("bold")

# 제안 방법 행 왼쪽 강조선 (첫 번째 셀 border)
tbl[len(ROW_DATA), 0].set_edgecolor("#1A5276")
tbl[len(ROW_DATA), 0].visible_edges = "LBT"

fig.suptitle("표 1. 방법별 주요 성능 지표 비교 (3D-FRONT, 181 케이스)",
             fontsize=12, fontweight="bold", y=0.97)
out1 = OUT_DIR / "표1_지표비교.png"
fig.savefig(out1, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"저장: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# 그림 3: 실제 배치 결과 예시
# ══════════════════════════════════════════════════════════════════════════════
with open("spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json") as f:
    preds_all = json.load(f)
with open("spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json") as f:
    cases_all = json.load(f)
cases_by_id = {c["id"]: c for c in cases_all}

VIS_METHODS = ["heuristic_baseline", "proposal_diffopt_constraint",
               "spacefit_gpt_text",  "constraint_solver"]
VIS_LABELS  = ["Heuristic", "DiffOpt", "GPT", "제안 방법"]

C_OK   = "#27AE60"
C_CF   = "#E74C3C"
C_IB   = "#E67E22"
C_FAIL = "#BBBBBB"


def rotated_corners(cx, cz, w, d, yaw_deg):
    hw, hd = max(w, 0.05) / 2, max(d, 0.05) / 2
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return [(cx + dx*c - dz*s, cz + dx*s + dz*c)
            for dx, dz in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]]


def check_pred(pred, case):
    if pred.get("status") != "placed":
        return False, False
    try:
        from shapely.geometry import Polygon as SP
        pos, sz = pred["position"], pred["size"]
        px, pz = float(pos["x"]), float(pos["z"])
        pw, pd = max(float(sz["width"]), 0.05), max(float(sz["depth"]), 0.05)
        yaw = float(pred.get("rotation_y", 0))
        fp = SP(rotated_corners(px, pz, pw, pd, yaw))
        floor = SP([(float(x), float(z)) for x, z in case["scene"]["floor"]["polygon"]])
        ib = fp.difference(floor).area <= 1e-4
        cf = True
        for obj in case["scene"].get("objects", []):
            ox, oz = float(obj["position"][0]), float(obj["position"][2])
            ow, od = max(float(obj["size"][0]), 0.05), max(float(obj["size"][2]), 0.05)
            if fp.intersection(SP(rotated_corners(ox, oz, ow, od, float(obj.get("yaw", 0))))).area > 1e-4:
                cf = False
                break
        return cf, ib
    except Exception:
        return False, False


def draw_room(ax, case, pred, method_label):
    scene = case["scene"]
    floor_pts = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    ax.add_patch(mpatches.Polygon(floor_pts, closed=True,
                                   facecolor="#F5F5F0", edgecolor="#444", lw=1.8, zorder=0))

    # 기존 가구
    for obj in scene.get("objects", []):
        px, _, pz = obj["position"]
        w, _, d = obj["size"]
        pts = rotated_corners(float(px), float(pz), float(w), float(d), float(obj.get("yaw", 0)))
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor="#CCD1D9",
                                       edgecolor="#888", lw=0.8, alpha=0.9, zorder=1))
        cat = str(obj.get("category", "")).replace("_", " ").split()
        short = cat[0][:7] if cat else ""
        ax.text(float(px), float(pz), short, fontsize=3.8, ha="center",
                va="center", color="#555", zorder=4)

    # 문 / 창문
    for door in scene.get("doors", []):
        p = door.get("position", [0, 0, 0])
        ax.plot(float(p[0]), float(p[2]), "s", color="#8B4513", ms=6, zorder=3)
    for win in scene.get("windows", []):
        p = win.get("position", [0, 0, 0])
        ax.plot(float(p[0]), float(p[2]), "D", color="#5DADE2", ms=5, zorder=3)

    # 예측 가구
    if pred.get("status") == "placed":
        cf, ib = check_pred(pred, case)
        pos, sz = pred["position"], pred["size"]
        px, pz = float(pos["x"]), float(pos["z"])
        pw, pd = float(sz["width"]), float(sz["depth"])
        yaw = float(pred.get("rotation_y", 0))
        fill_c = C_OK if (cf and ib) else C_CF if not cf else C_IB
        pts = rotated_corners(px, pz, pw, pd, yaw)
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=fill_c,
                                       edgecolor="white", lw=2.0, alpha=0.92, zorder=6))
        rad = math.radians(yaw)
        ln = min(pw, pd) * 0.38
        ax.annotate("", xy=(px + math.cos(rad)*ln, pz + math.sin(rad)*ln),
                    xytext=(px, pz),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5), zorder=7)
        status = "성공" if (cf and ib) else ("충돌" if not cf else "경계 이탈")
        ok = cf and ib
        sc = "#1B6B3A" if ok else "#922B21"
        marker = "O" if ok else "X"
    else:
        status = "미배치"
        sc = "#777"
        marker = "X"

    xs = [p[0] for p in floor_pts]
    zs = [p[1] for p in floor_pts]
    m = 0.4
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(zs) - m, max(zs) + m)
    ax.set_aspect("equal")
    ax.axis("off")

    is_ours = "제안" in method_label
    title_color = sc
    fw = "bold" if is_ours else "normal"
    ax.set_title(f"{method_label}\n{marker}  {status}",
                 fontsize=9.5, fontweight=fw, color=title_color, pad=5)

    if is_ours:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#1A5276")
            spine.set_linewidth(2.8)


# ── 케이스 선택 (hardcoded best examples) ──────────────────────────────────
# Row 1: armchair — DiffOpt & GPT-Text 모두 경계 이탈, ours 성공
# Row 2: bed      — DiffOpt 명확한 경계 이탈(0.78m²), ours 1.36m 여유
SELECTED_IDS = [
    # Row 1: 3개 baseline 모두 실패, ours만 성공 (armchair beside bed)
    "0ab19ed6-debc-4c6e-857b-29290fb111d9__secondbedroom_11769__furniture_421",
    # Row 2: 2개 IB 실패, ours 성공 (armchair beside sofa facing TV)
    "09cbb7d2-3005-4bbe-bf52-b05214df88ce__livingdiningroom_13172__furniture_140",
]
sel = [cases_by_id[cid] for cid in SELECTED_IDS if cid in cases_by_id]

fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
fig.patch.set_facecolor("white")

for row_idx, case in enumerate(sel):
    cid = case["id"]
    target = str(case["target_asset"]["category"]).replace("_", " ")
    intent = case.get("intent", {}).get("text", "")[:60]

    axes[row_idx][0].set_ylabel(
        f"[{target}]\n\"{intent}...\"",
        fontsize=7.5, rotation=0, labelpad=6,
        ha="right", va="center",
    )

    for col_idx, (method, label) in enumerate(zip(VIS_METHODS, VIS_LABELS)):
        ax = axes[row_idx][col_idx]
        pred = (preds_all.get(method, {}).get(cid) or [{}])[0]
        draw_room(ax, case, pred, label)

# 범례
patches = [
    mpatches.Patch(facecolor=C_OK,   edgecolor="white", label="성공 (CF+IB 통과)"),
    mpatches.Patch(facecolor=C_CF,   edgecolor="white", label="충돌 발생"),
    mpatches.Patch(facecolor=C_IB,   edgecolor="white", label="경계 이탈"),
    mpatches.Patch(facecolor=C_FAIL, edgecolor="white", label="배치 실패"),
    mpatches.Patch(facecolor="#CCD1D9", edgecolor="#888", label="기존 가구"),
]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out2 = OUT_DIR / "그림2_배치예시.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"저장: {out2}")
print("\n완료.")
