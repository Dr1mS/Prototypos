"""Publication figures for the paper, EN + FR.

Usage:  python make_figs.py        -> English labels  -> fig*.pdf
        python make_figs.py --fr   -> French labels   -> fr_fig*.pdf

Needs the G0 simulator on the path (G0.py in this repo, or g0_harness.py).
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import G0 as g
except ImportError:
    import g0_harness as g

FR = "--fr" in sys.argv
OUT = "fr_" if FR else ""

L = {
    "history":      ("history", "historique"),
    "coupled":      ("state-coupled perception", "perception couplée à l'état"),
    "ablated":      ("ablation (no coupling)", "ablation (sans couplage)"),
    "adverse":      ("adverse", "phase adverse"),
    "corrective":   ("identical corrective input", "correction identique"),
    "interaction":  ("interaction", "interaction"),
    "latent":       (r"latent state $x_A$", r"état latent $x_A$"),
    "recency":      ("recency\n(R1)", "récence\n(R1)"),
    "rewrite":      ("rewrite\n(R2)", "réécriture\n(R2)"),
    "content":      ("content\n(R3)", "contenu\n(R3)"),
    "pct_null":     ("percentile of own null", "percentile du modèle nul"),
    "pct95":        ("95th pct", "95e pct"),
    "a_order":      ("(a) order transmission", "(a) transmission de l'ordre"),
    "addressing":   ("retrieval addressing", "adressage de la récupération"),
    "spread":       (r"spread of endpoint, $\sigma$", r"dispersion finale, $\sigma$"),
    "noise":        ("noise band", "bande de bruit"),
    "b_attr":       ("(b) attractor structure", "(b) structure d'attracteurs"),
    "mem_arch":     ("memory architecture", "architecture mémoire"),
    "early_ex":     ("early (induction) exemplars", "exemplaires précoces"),
    "corr_ex":      ("corrective exemplars", "exemplaires correctifs"),
    "base_rate":    ("base rate", "taux de base"),
    "corr_phase":   ("corrective phase", "phase corrective"),
    "turn":         ("turn", "tour"),
    "slots":        ("fraction of retrieved slots", "part des slots récupérés"),
    "never":        ("never\npressured", "jamais\npressé"),
    "p_to_c":       ("perm.\n$\\rightarrow$ caut.", "perm.\n$\\rightarrow$ prud."),
    "c_to_p":       ("caut.\n$\\rightarrow$ perm.", "prud.\n$\\rightarrow$ perm."),
    "adherence":    ("safeguard adherence", "respect des garde-fous"),
    "a_hist":       ("(a) interaction history", "(a) historique d'interaction"),
    "b_mem":        ("(b) memory contents", "(b) contenu mémoire"),
    "prepared":     ("prepared adherence", "niveau préparé"),
    "lv":           (["4 neut.", "4 perm.", "8 perm.", "4 caut.", "8 caut."],
                     ["4 neut.", "4 perm.", "8 perm.", "4 prud.", "8 prud."]),
}


def T(k):
    return L[k][1 if FR else 0]


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito colorblind-safe palette
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
GREY = "#666666"


# ---------------------------------------------------------------- FIG 1
def fig1_engineered():
    """The engineered signature: irreversibility under identical kindness."""
    LR = 0.13
    warmup = ["nurture", "play"] * 14
    harm = ["harm", "scold"] * 37
    cap = g.run(warmup + harm, g.P, bias_on=True, lr_const=LR)
    s_dmg = cap[-1].copy()
    heal = ["nurture", "play"] * 70
    rec_on = g.run(heal, g.P, s0=s_dmg, bias_on=True, lr_const=LR)
    rec_off = g.run(heal, g.P, s0=s_dmg, bias_on=False, lr_const=LR)

    t_cap = np.arange(len(cap))
    t_rec = np.arange(len(cap), len(cap) + len(heal))

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.axvspan(len(warmup), len(cap), color=VERM, alpha=0.10, lw=0)
    ax.axvspan(len(cap), len(cap) + len(heal), color=GREEN, alpha=0.10, lw=0)
    ax.plot(t_cap, cap[:, 2], color=GREY, lw=1.4, label=T("history"))
    ax.plot(t_rec, rec_on[:, 2], color=VERM, lw=1.8, label=T("coupled"))
    ax.plot(t_rec, rec_off[:, 2], color=BLUE, lw=1.6, ls="--", label=T("ablated"))
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.text(len(warmup) + 20, 1.05, T("adverse"), color=VERM, fontsize=8)
    ax.text(len(cap) + 30, 1.05, T("corrective"), color=GREEN, fontsize=8)
    ax.set_xlabel(T("interaction"))
    ax.set_ylabel(T("latent"))
    ax.set_ylim(-1.35, 1.35)
    ax.legend(loc="center right", frameon=False, fontsize=7.5)
    fig.savefig(OUT + "fig1_engineered.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 2
def fig2_ladder():
    """The ladder: order transmission vs attractor structure."""
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.5),
                           gridspec_kw={"width_ratios": [1, 1.15]})

    lab = [T("recency"), T("rewrite"), T("content")]
    pct = [3.7, 78.2, 99.5]
    cols = [BLUE, BLUE, VERM]
    ax[0].bar(range(3), pct, color=cols, width=0.6)
    ax[0].axhline(95, color="#333333", ls=":", lw=1.0)
    ax[0].text(2.45, 91, T("pct95"), fontsize=7, ha="right", color="#333333")
    for i, v in enumerate(pct):
        ax[0].text(i, v + 3, f"{v:.1f}", ha="center", fontsize=7.5, color=cols[i])
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(lab)
    ax[0].set_ylabel(T("pct_null"))
    ax[0].set_ylim(0, 118)
    ax[0].set_title(T("a_order"), fontsize=9)
    ax[0].set_xlabel(T("addressing"))

    lab2 = ["R0", "R1", "R2", "R3", "R4"]
    std = [0.071, 0.046, 0.051, 0.062, 0.764]
    cols2 = [GREY, BLUE, BLUE, VERM, PURPLE]
    ax[1].axhspan(0.03, 0.09, color="#dddddd", zorder=0)
    ax[1].bar(range(5), std, color=cols2, width=0.6, zorder=2)
    for i, v in enumerate(std):
        ax[1].text(i, v + 0.028, f"{v:.3f}", ha="center", fontsize=7)
    ax[1].text(1.6, 0.135, T("noise"), fontsize=7, ha="center", color="#555555")
    ax[1].set_xticks(range(5))
    ax[1].set_xticklabels(lab2)
    ax[1].set_ylabel(T("spread"))
    ax[1].set_ylim(0, 0.92)
    ax[1].set_title(T("b_attr"), fontsize=9)
    ax[1].set_xlabel(T("mem_arch"))

    fig.savefig(OUT + "fig2_ladder.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 3
def fig3_provenance():
    """Retrieval provenance: early exemplars persist through correction."""
    t = [15, 20, 25, 30, 35]
    soft = [1.000, 0.847, 0.708, 0.583, 0.583]
    firm = [0.000, 0.153, 0.292, 0.417, 0.417]
    base = 0.429

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.axvspan(15, 30, color=GREEN, alpha=0.10, lw=0)
    ax.plot(t, soft, "o-", color=VERM, lw=1.6, ms=4, label=T("early_ex"))
    ax.plot(t, firm, "s--", color=BLUE, lw=1.4, ms=3.5, label=T("corr_ex"))
    ax.axhline(base, color="#333333", ls=":", lw=1.0)
    ax.text(35.2, base, T("base_rate"), fontsize=7, va="center", color="#333333")
    ax.annotate(r"$1.36\times$", xy=(35, 0.583), xytext=(30.5, 0.72),
                fontsize=8, color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=0.9))
    ax.text(22, 0.05, T("corr_phase"), fontsize=7, color="#3a7a5a", ha="center")
    ax.set_xlabel(T("turn"))
    ax.set_ylabel(T("slots"))
    ax.set_xticks(t)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(13.5, 39)
    ax.legend(loc="upper center", frameon=False, fontsize=7,
              bbox_to_anchor=(0.52, 1.02))
    fig.savefig(OUT + "fig3_provenance.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 4
def fig4_mundane():
    """The mundane state: unpressured memory is the least cautious."""
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.5),
                           gridspec_kw={"width_ratios": [1, 1.1]})

    arms = [T("never"), T("p_to_c"), T("c_to_p")]
    seeds = [[0.667, 0.667, 0.792], [0.875, 1.000, 0.833],
             [0.917, 0.958, 0.917]]
    means = [np.mean(s) for s in seeds]
    cols = [VERM, BLUE, BLUE]
    for i, (s, m) in enumerate(zip(seeds, means)):
        ax[0].bar(i, m, color=cols[i], width=0.55, alpha=0.9)
        ax[0].scatter([i] * 3, s, color="white", edgecolor="#222222",
                      s=16, zorder=3, linewidth=0.7)
        ax[0].text(i, m + 0.045, f"{m:.2f}", ha="center", fontsize=7.5,
                   color=cols[i])
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(arms)
    ax[0].set_ylabel(T("adherence"))
    ax[0].set_ylim(0, 1.18)
    ax[0].set_title(T("a_hist"), fontsize=9)

    lab = T("lv")
    before = [0.625, 0.792, 0.875, 1.000, 1.000]
    cols2 = [VERM, BLUE, BLUE, BLUE, BLUE]
    ax[1].bar(range(5), before, color=cols2, width=0.6, alpha=0.9)
    for i, v in enumerate(before):
        ax[1].text(i, v + 0.032, f"{v:.3f}", ha="center", fontsize=7)
    ax[1].set_xticks(range(5))
    ax[1].set_xticklabels(lab, rotation=30, ha="right")
    ax[1].set_ylabel(T("prepared"))
    ax[1].set_ylim(0, 1.16)
    ax[1].set_title(T("b_mem"), fontsize=9)

    fig.savefig(OUT + "fig4_mundane.pdf")
    plt.close(fig)


fig1_engineered()
fig2_ladder()
fig3_provenance()
fig4_mundane()
print("wrote 4 vector PDFs" + (" (FR)" if FR else " (EN)"))
