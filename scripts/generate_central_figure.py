"""
generate_central_figure.py — Physio-PPT central comparison figure.

Compares three order-aware SSL pretexts on a 3-beat synthetic ECG:
  (A) Original signal with P/QRS/T annotations
  (B) PPT — globally shuffled patches (physiology-agnostic)
  (C) ECGWavePuzzle — segment-level P/QRS/T jigsaw inside each beat
  (D) Physio-PPT — patch permutations constrained within each segment

Output: fig_physio_ppt_3beats_en.{pdf,png}   (in --outdir, default: paper_notes/)

Style is standardised to match fig_preprocessing.pdf:
  – same rcParams (DejaVu Sans, 9 pt)
  – same P/QRS/T colour palette (#4E9AF1 / #E84040 / #4EBF6C)
  – same spine removal (top+right+left)
  – separate per-panel axes with shared x-axis
  – panel labels (A)–(D) in IEEEtran style
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — identical to fig_preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
C_P   = "#4E9AF1"   # blue   — P segment
C_QRS = "#E84040"   # red    — QRS segment
C_T   = "#4EBF6C"   # green  — T segment
C_R   = "#C0392B"   # dark red — R-peak marker
C_PPT = "#8E44AD"   # purple — PPT row
C_WAV = "#E67E22"   # orange — WavePuzzle row
C_PHY = "#1A5276"   # navy   — Physio-PPT row
C_SIG = "0.15"      # near-black for original signal

# ─────────────────────────────────────────────────────────────────────────────
# rcParams — same as fig_preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic ECG helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gauss(tt, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((tt - mu) / sigma) ** 2)


def synth_3beat_ecg(fs=500, rr_s=0.88, noise=0.012, seed=0):
    dur    = 3 * rr_s + 0.30
    t      = np.arange(0, dur, 1.0 / fs)
    rpeaks = [rr_s * (i + 0.5) for i in range(3)]
    x      = np.zeros_like(t)
    for r in rpeaks:
        x += (
            _gauss(t, r - 0.195, 0.030,  0.13) +
            _gauss(t, r - 0.018, 0.010, -0.14) +
            _gauss(t, r,         0.006,  1.05) +
            _gauss(t, r + 0.022, 0.012, -0.25) +
            _gauss(t, r + 0.285, 0.070,  0.33)
        )
    rng = np.random.default_rng(seed)
    x  += 0.03 * np.sin(2 * np.pi * 0.25 * t) + noise * rng.standard_normal(t.shape)
    return x.astype(np.float32), rpeaks, t

# ─────────────────────────────────────────────────────────────────────────────
# Segment boundaries (relative to R-peak, in seconds)
# ─────────────────────────────────────────────────────────────────────────────
P_REL   = (-0.28, -0.09)
QRS_REL = (-0.06,  0.09)
T_REL   = ( 0.12,  0.48)

def _si(t_arr, t_sec):
    return int(np.clip(np.searchsorted(t_arr, t_sec), 0, len(t_arr) - 1))

def seg_bounds(t, r):
    return {
        "P":   (_si(t, r + P_REL[0]),   _si(t, r + P_REL[1])),
        "QRS": (_si(t, r + QRS_REL[0]), _si(t, r + QRS_REL[1])),
        "T":   (_si(t, r + T_REL[0]),   _si(t, r + T_REL[1])),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Perturbation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shuffle_blocks(seg, blk, rng):
    if len(seg) < 2 or blk >= len(seg):
        return seg
    blocks = [seg[i: i + blk] for i in range(0, len(seg) - blk + 1, blk)]
    tail   = seg[len(blocks) * blk:]
    rng.shuffle(blocks)
    return np.concatenate(blocks + ([tail] if len(tail) else []))


def make_ppt(x, t, seed=42):
    blk = int(0.10 * 500)
    rng = np.random.default_rng(seed)
    n   = len(x) // blk
    idx = np.arange(n)
    rng.shuffle(idx)
    out = np.empty_like(x)
    for new_i, old_i in enumerate(idx):
        out[new_i * blk:(new_i + 1) * blk] = x[old_i * blk:(old_i + 1) * blk]
    out[n * blk:] = x[n * blk:]
    return out


def make_wavepuzzle(x, t, rpeaks, order=("T", "QRS", "P")):
    xw = x.copy()
    for r in rpeaks:
        sb = seg_bounds(t, r)
        all_i = [v for seg in sb.values() for v in seg]
        core0, core1 = min(all_i), max(all_i)
        segs = {name: xw[s:e].copy() for name, (s, e) in sb.items()}
        new_core = np.concatenate([segs[name] for name in order])
        core_len = core1 - core0
        if len(new_core) < core_len:
            new_core = np.pad(new_core, (0, core_len - len(new_core)))
        xw[core0:core1] = new_core[:core_len]
    return xw


def make_physio_ppt(x, t, rpeaks, seed=7):
    xp  = x.copy()
    blk = {"P": int(0.030 * 500), "QRS": int(0.015 * 500), "T": int(0.040 * 500)}
    for bi, r in enumerate(rpeaks):
        sb = seg_bounds(t, r)
        for sname, (s, e) in sb.items():
            rng = np.random.default_rng(seed + bi * 10 + {"P": 0, "QRS": 1, "T": 2}[sname])
            xp[s:e] = _shuffle_blocks(xp[s:e], blk[sname], rng)
    return xp

# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    x, rpeaks, t = synth_3beat_ecg(fs=500, rr_s=0.88, noise=0.012, seed=3)
    x_ppt  = make_ppt(x, t, seed=42)
    x_wave = make_wavepuzzle(x, t, rpeaks, order=("T", "QRS", "P"))
    x_phys = make_physio_ppt(x, t, rpeaks, seed=7)

    panels = [
        ("(A)", "Original — P / QRS / T structure preserved",
         x,      C_SIG, True),
        ("(B)", r"PPT (traditional): globally shuffled 100-ms patches — physiology-agnostic",
         x_ppt,  C_PPT, False),
        ("(C)", r"ECGWavePuzzle: segment-level jigsaw (T$\to$QRS$\to$P inside each beat)",
         x_wave, C_WAV, False),
        ("(D)", r"Physio-PPT (ours): patch shuffle $within$ each segment — macro-order preserved",
         x_phys, C_PHY, False),
    ]

    fig = plt.figure(figsize=(13.5, 8.2))
    gs  = gridspec.GridSpec(
        4, 1, figure=fig,
        hspace=0.52,
        left=0.06, right=0.97, top=0.94, bottom=0.07,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    seg_colors = {"P": C_P, "QRS": C_QRS, "T": C_T}

    for idx_panel, (ax, (lbl, title, sig, row_color, annotate_segs)) in \
            enumerate(zip(axes, panels)):

        # shaded segment regions on every row
        for r in rpeaks:
            sb = seg_bounds(t, r)
            for sname, (s, e) in sb.items():
                ax.axvspan(t[s], t[e], alpha=0.15,
                           color=seg_colors[sname], lw=0, zorder=1)
            ax.axvline(t[_si(t, r)], color=C_R, lw=0.75, ls="--", alpha=0.45, zorder=2)

        ax.plot(t, sig, color=row_color, lw=0.95, zorder=3)

        # R-peak + segment name annotations on the original row only
        if annotate_segs:
            for r in rpeaks:
                ri_idx = _si(t, r)
                ax.text(t[ri_idx], sig[ri_idx] + 0.07, "R",
                        ha="center", va="bottom", fontsize=7.5,
                        color=C_R, fontweight="bold")
            r0  = rpeaks[0]
            sb0 = seg_bounds(t, r0)
            for sname, (s, e) in sb0.items():
                ax.text((t[s] + t[e]) / 2, 1.15, sname,
                        ha="center", va="bottom", fontsize=10,
                        color=seg_colors[sname], fontweight="bold")

        ax.set_title(f"$\\mathbf{{{lbl}}}$  {title}",
                     fontsize=9.2, loc="left", pad=4)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-0.85, 1.40)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

        if idx_panel < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(
                r"Time (s)  —  $f_s{=}500\,\mathrm{Hz}$, synthetic ECG for illustration",
                fontsize=8.5)

    # shared segment legend
    handles = [
        mpatches.Patch(color=C_P,   alpha=0.55, label="P region"),
        mpatches.Patch(color=C_QRS, alpha=0.55, label="QRS region"),
        mpatches.Patch(color=C_T,   alpha=0.55, label="T region"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="upper right",
                   framealpha=0.80, edgecolor="0.7", ncol=3)

    # callout annotation boxes for (B), (C), (D)
    callouts = [
        (axes[1], C_PPT, "#F5EEF8",
         "Patch boundaries cross P/QRS/T\n"
         "\u2192 physiologically implausible reorderings"),
        (axes[2], C_WAV, "#FEF5E7",
         "Macro-order P\u2192QRS\u2192T violated\n"
         "\u2192 requires beat delineation\n"
         "\u2192 no weak/strong view invariance"),
        (axes[3], C_PHY, "#EAF2FF",
         "Macro-order P\u2192QRS\u2192T preserved\n"
         "Micro-order disturbed within segments\n"
         "\u2192 physiologically plausible perturbation"),
    ]
    for cax, ec, fc, txt in callouts:
        cax.text(0.987, 0.95, txt,
                 transform=cax.transAxes,
                 ha="right", va="top", fontsize=7.5, color=ec,
                 bbox=dict(boxstyle="round,pad=0.38", fc=fc, ec=ec, lw=0.8))

    fig.suptitle(
        "Visual comparison of PPT, ECGWavePuzzle and Physio-PPT "
        "on a 3-beat synthetic ECG (single lead shown)",
        fontsize=10, y=0.975,
    )

    for ext in ("pdf", "png"):
        out = outdir / f"fig_physio_ppt_3beats_en.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=180)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="paper_notes", type=Path,
                        help="Output directory (default: paper_notes/)")
    args = parser.parse_args()
    make_figure(args.outdir)

