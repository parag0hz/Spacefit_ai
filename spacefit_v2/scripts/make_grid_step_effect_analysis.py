"""Build a presentation-ready analysis for grid-step resolution ablation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "spacefit_v2" / "results" / "grid_step_ablation"
SUMMARY_CSV = RESULT_DIR / "grid_step_ablation_summary.csv"
OUT_PNG = RESULT_DIR / "grid_step_effect_analysis.png"
OUT_SVG = RESULT_DIR / "grid_step_effect_analysis.svg"
OUT_MD = RESULT_DIR / "GRID_STEP_EFFECT_ANALYSIS.md"


def setup_korean_font() -> None:
    candidates = ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "AppleGothic"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in candidates:
        if family in available:
            matplotlib.rcParams["font.family"] = family
            break
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if key != "cases"} | {"cases": float(row["cases"])})
    rows.sort(key=lambda r: r["grid_step_m"])
    return rows


def save_plot(rows: list[dict[str, float]]) -> None:
    setup_korean_font()
    steps = [r["grid_step_m"] for r in rows]
    runtime = [r["runtime_s"] for r in rows]
    cf = [r["CF"] for r in rows]
    ib = [r["IB"] for r in rows]
    cps = [r["CPS"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(steps, cf, marker="o", linewidth=2.4, markersize=8, color="#2563eb", linestyle="--", label="CF")
    ax.plot(steps, ib, marker="s", linewidth=2.6, markersize=8, color="#16a34a", label="IB")
    ax.plot(steps, cps, marker="^", linewidth=2.6, markersize=8, color="#f97316", label="CPS")
    ax.set_title("후보 생성 해상도에 따른 성능 변화", fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("x/z grid step (m)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(steps)
    ax.set_ylim(0.45, 0.99)
    ax.grid(axis="y", color="#e5e7eb", linewidth=1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    ax.text(
        0.12,
        0.94,
        "CF와 IB는 본 실험에서 같은 값으로 측정되어 선이 겹침",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.92),
    )

    for x, y in zip(steps, cps):
        ax.text(x, y - 0.035, f"{y:.3f}", ha="center", va="top", fontsize=10, color="#9a3412")

    ax2 = axes[1]
    bars = ax2.bar([str(s) for s in steps], runtime, color=["#93c5fd", "#86efac", "#fdba74"], edgecolor="#334155")
    ax2.set_title("실행시간 변화", fontsize=15, weight="bold", pad=12)
    ax2.set_xlabel("x/z grid step (m)", fontsize=12)
    ax2.set_ylabel("Runtime (seconds)", fontsize=12)
    ax2.grid(axis="y", color="#e5e7eb", linewidth=1.0)
    for bar, value in zip(bars, runtime):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(runtime) * 0.025,
            f"{value:.1f}s\n({value / 60:.1f}m)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0f172a",
        )

    fig.suptitle("Grid Step Ablation: Candidate Generation Resolution", fontsize=19, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "0.10m는 후보를 더 촘촘히 만들지만 실행시간이 크게 증가하고 top-1 CPS는 개선되지 않음. 0.18m가 현재 정확도-효율 균형점.",
        ha="center",
        fontsize=12,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt(v: float) -> str:
    return f"{v:.3f}"


def save_markdown(rows: list[dict[str, float]]) -> None:
    by_step = {r["grid_step_m"]: r for r in rows}
    best_cf = max(rows, key=lambda r: r["CF"])
    best_ib = max(rows, key=lambda r: r["IB"])
    best_cps = max(rows, key=lambda r: r["CPS"])
    fastest = min(rows, key=lambda r: r["runtime_s"])
    densest = min(rows, key=lambda r: r["grid_step_m"])
    current = by_step.get(0.18, rows[1])

    lines = [
        "# Grid 간격별 후보 생성 해상도 분석",
        "",
        "## 목적",
        "",
        "후보 생성 단계에서 사용하는 x/z grid 간격이 물리적 유효성(CF, IB), 최종 성공률(CPS), 실행시간에 어떤 영향을 주는지 분석한다.",
        "grid 간격은 후보 위치를 샘플링하는 해상도이므로, 간격이 작을수록 더 촘촘하게 탐색하지만 계산량이 증가한다.",
        "",
        "## 실험 설정",
        "",
        "- 방법론: `Candidate Search & Scoring` (`constraint_solver`)",
        "- 데이터: GPT-intent single-target test split",
        f"- 케이스 수: `{int(rows[0]['cases'])}`",
        "- 공통 조건: `top-k=5`, yaw 후보는 동일하게 고정",
        "- 변경한 변수: x/z 후보 생성 `grid_step_m`",
        "",
        "## 결과",
        "",
        "| Grid step | Runtime | CF | IB | CPS | Success@5 | Constraint Acc. |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['grid_step_m']:.2f} m` | `{r['runtime_s']:.1f} s` | "
            f"{fmt(r['CF'])} | {fmt(r['IB'])} | {fmt(r['CPS'])} | "
            f"{fmt(r['Success@5'])} | {fmt(r['Constraint Accuracy'])} |"
        )

    runtime_ratio = densest["runtime_s"] / current["runtime_s"]
    cps_delta_dense = densest["CPS"] - current["CPS"]
    cps_delta_coarse = by_step[0.25]["CPS"] - current["CPS"] if 0.25 in by_step else 0.0

    lines += [
        "",
        "## 해석",
        "",
        f"- 가장 촘촘한 `0.10 m`는 `Success@5={fmt(densest['Success@5'])}`로 top-k 후보 안에 성공 후보가 들어올 가능성은 가장 높다.",
        f"- 하지만 `0.10 m`의 top-1 CPS는 `{fmt(densest['CPS'])}`로, 현재 기본값 `0.18 m`의 `{fmt(current['CPS'])}`보다 낮다. 차이: `{cps_delta_dense:+.3f}`.",
        f"- `0.10 m` 실행시간은 `{densest['runtime_s']:.1f}s`로 `0.18 m` 대비 약 `{runtime_ratio:.1f}배` 길다.",
        f"- `0.25 m`는 `{fastest['runtime_s']:.1f}s`로 가장 빠르지만, CF/IB/Constraint Accuracy가 모두 `0.18 m`보다 낮다.",
        f"- `0.25 m`의 CPS는 반올림상 `0.18 m`와 같지만, Success@5와 물리/제약 관련 지표가 약해져 후보 품질은 더 불안정하다. CPS 차이: `{cps_delta_coarse:+.3f}`.",
        "",
        "## 왜 grid 간격이 성능에 영향을 주는가?",
        "",
        "후보 생성은 연속적인 방 공간을 일정 간격의 이산 후보점으로 바꾸는 과정이다.",
        "grid가 너무 작으면 가능한 후보는 많이 생기지만, 비슷한 후보가 과도하게 늘어나 top-1 scorer가 더 어려운 선택을 해야 하고 실행시간이 증가한다.",
        "반대로 grid가 너무 크면 실제로는 가능한 좁은 배치 영역을 건너뛸 수 있다.",
        "특히 큰 가구나 벽/문/창문/기존 가구 clearance가 동시에 걸리는 경우, 성공 가능한 영역이 수십 cm 수준으로 좁아질 수 있어 `0.25 m`에서는 좋은 후보가 아예 생성되지 않을 수 있다.",
        "",
        "## 결론",
        "",
        f"현재 실험에서는 `0.18 m`가 가장 합리적인 기본값이다. `0.18 m`는 CF `{fmt(best_cf['CF'])}`, IB `{fmt(best_ib['IB'])}`, CPS `{fmt(best_cps['CPS'])}` 수준을 유지하면서 `0.10 m`보다 훨씬 빠르다.",
        "따라서 후보 생성 해상도는 단순히 촘촘할수록 좋은 것이 아니라, 후보 품질과 실행시간 사이의 균형점으로 설정해야 한다.",
        "",
        "## 발표용 한 문장",
        "",
        "> Grid 간격을 줄이면 후보 탐색은 촘촘해지지만 계산량이 크게 증가하고 top-1 성능이 반드시 좋아지지는 않는다. 반대로 grid가 너무 크면 가능한 좁은 배치 영역을 놓칠 수 있다. 본 실험에서는 `0.18 m`가 CF, IB, CPS와 실행시간의 균형이 가장 좋은 후보 생성 해상도였다.",
        "",
        "## 생성 파일",
        "",
        f"- `{OUT_PNG.relative_to(ROOT)}`",
        f"- `{OUT_SVG.relative_to(ROOT)}`",
        f"- `{SUMMARY_CSV.relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    save_plot(rows)
    save_markdown(rows)
    print(OUT_MD)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
