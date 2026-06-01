"""
blimp_contrast_analysis_multigpu.py
══════════════════════════════════════════════════════════════════════════════
Contrastive SAE feature analysis: strong vs weak BLiMP tasks.
Multi-GPU version — fully utilises all 8 H100s.

Strategy: task-parallel sharding.
  • Each GPU process loads its own copy of the model + SAE.
  • The 67 BLiMP tasks are split evenly across ranks.
  • Rank 0 collects results from all ranks, aggregates, and generates figures.
  • Zero inter-GPU communication during inference → near-linear 8× speedup.

Launch (all 8 GPUs):
    torchrun --standalone --nproc_per_node=8 blimp_contrast_analysis_multigpu.py \
             --layer 12 --n-samples 1000

    # Or all trained layers:
    torchrun --standalone --nproc_per_node=8 blimp_contrast_analysis_multigpu.py \
             --all-layers --n-samples 1000

    # Fewer GPUs (e.g. 4):
    torchrun --standalone --nproc_per_node=4 blimp_contrast_analysis_multigpu.py \
             --layer 12

Original single-GPU command (still works unchanged):
    python blimp_contrast_analysis_multigpu.py --layer 12 --n-samples 1000
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sae_lens import SAE
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths & constants ──────────────────────────────────────────────────────────
GPT2_REPO   = "IParraMartin/gpt2-medium-bLM100M"
SAE_REPO    = "whitepenguin/gpt2-medium-bLM100M-SAE"
SAE_CACHE   = os.path.expanduser("~/.cache/babylm_saes")
RESULTS_DIR = "./blimp_contrast_results"
FIGURES_DIR = "./blimp_contrast_figures"

STRONG_THRESHOLD = 0.80
WEAK_THRESHOLD   = 0.40

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── BLiMP benchmark accuracies ─────────────────────────────────────────────────
BENCHMARK_ACC = {
    "adjunct_island":                                             0.804,
    "anaphor_gender_agreement":                                   0.913,
    "anaphor_number_agreement":                                   0.960,
    "animate_subject_passive":                                    0.753,
    "animate_subject_trans":                                      0.889,
    "causative":                                                  0.621,
    "complex_NP_island":                                          0.497,
    "coordinate_structure_constraint_complex_left_branch":        0.128,
    "coordinate_structure_constraint_object_extraction":          0.621,
    "determiner_noun_agreement_1":                                0.933,
    "determiner_noun_agreement_2":                                0.941,
    "determiner_noun_agreement_irregular_1":                      0.779,
    "determiner_noun_agreement_irregular_2":                      0.856,
    "determiner_noun_agreement_with_adj_2":                       0.885,
    "determiner_noun_agreement_with_adj_irregular_1":             0.741,
    "determiner_noun_agreement_with_adj_irregular_2":             0.817,
    "determiner_noun_agreement_with_adjective_1":                 0.873,
    "distractor_agreement_relational_noun":                       0.437,
    "distractor_agreement_relative_clause":                       0.258,
    "drop_argument":                                              0.742,
    "ellipsis_n_bar_1":                                           0.625,
    "ellipsis_n_bar_2":                                           0.693,
    "existential_there_object_raising":                           0.820,
    "existential_there_quantifiers_1":                            0.989,
    "existential_there_quantifiers_2":                            0.199,
    "existential_there_subject_raising":                          0.801,
    "expletive_it_object_raising":                                0.731,
    "inchoative":                                                 0.481,
    "intransitive":                                               0.673,
    "irregular_past_participle_adjectives":                       0.948,
    "irregular_past_participle_verbs":                            0.787,
    "irregular_plural_subject_verb_agreement_1":                  0.777,
    "irregular_plural_subject_verb_agreement_2":                  0.762,
    "left_branch_island_echo_question":                           0.339,
    "left_branch_island_simple_question":                         0.235,
    "matrix_question_npi_licensor_present":                       0.109,
    "npi_present_1":                                              0.321,
    "npi_present_2":                                              0.404,
    "only_npi_licensor_present":                                  0.875,
    "only_npi_scope":                                             0.552,
    "passive_1":                                                  0.899,
    "passive_2":                                                  0.869,
    "principle_A_c_command":                                      0.460,
    "principle_A_case_1":                                         1.000,
    "principle_A_case_2":                                         0.829,
    "principle_A_domain_1":                                       0.972,
    "principle_A_domain_2":                                       0.648,
    "principle_A_domain_3":                                       0.569,
    "principle_A_reconstruction":                                 0.428,
    "regular_plural_subject_verb_agreement_1":                    0.844,
    "regular_plural_subject_verb_agreement_2":                    0.766,
    "sentential_negation_npi_licensor_present":                   0.998,
    "sentential_negation_npi_scope":                              0.191,
    "sentential_subject_island":                                  0.349,
    "superlative_quantifiers_1":                                  0.931,
    "superlative_quantifiers_2":                                  0.797,
    "tough_vs_raising_1":                                         0.330,
    "tough_vs_raising_2":                                         0.752,
    "transitive":                                                 0.792,
    "wh_island":                                                  0.528,
    "wh_questions_object_gap":                                    0.670,
    "wh_questions_subject_gap":                                   0.905,
    "wh_questions_subject_gap_long_distance":                     0.983,
    "wh_vs_that_no_gap":                                          0.978,
    "wh_vs_that_no_gap_long_distance":                            0.994,
    "wh_vs_that_with_gap":                                        0.241,
    "wh_vs_that_with_gap_long_distance":                          0.069,
}

CATEGORY_MAP = {
    "Anaphor":     ["anaphor_gender_agreement", "anaphor_number_agreement"],
    "Determiner":  ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                    "determiner_noun_agreement_with_adj_2",
                    "determiner_noun_agreement_with_adjective_1",
                    "determiner_noun_agreement_irregular_1",
                    "determiner_noun_agreement_irregular_2"],
    "Island":      ["coordinate_structure_constraint_complex_left_branch",
                    "left_branch_island_simple_question",
                    "left_branch_island_echo_question",
                    "sentential_subject_island", "wh_island", "complex_NP_island",
                    "adjunct_island"],
    "NPI":         ["matrix_question_npi_licensor_present", "npi_present_1",
                    "npi_present_2", "sentential_negation_npi_scope",
                    "sentential_negation_npi_licensor_present",
                    "only_npi_licensor_present", "only_npi_scope"],
    "Wh":          ["wh_questions_object_gap", "wh_questions_subject_gap",
                    "wh_questions_subject_gap_long_distance",
                    "wh_vs_that_with_gap", "wh_vs_that_no_gap",
                    "wh_vs_that_with_gap_long_distance",
                    "wh_vs_that_no_gap_long_distance"],
    "Principle A": ["principle_A_c_command", "principle_A_case_1",
                    "principle_A_case_2", "principle_A_domain_1",
                    "principle_A_domain_2", "principle_A_domain_3",
                    "principle_A_reconstruction"],
    "Agreement":   ["regular_plural_subject_verb_agreement_1",
                    "regular_plural_subject_verb_agreement_2",
                    "irregular_plural_subject_verb_agreement_1",
                    "irregular_plural_subject_verb_agreement_2",
                    "distractor_agreement_relational_noun",
                    "distractor_agreement_relative_clause"],
}

TASK_TO_CATEGORY = {
    task: cat for cat, tasks in CATEGORY_MAP.items() for task in tasks
}

CATEGORY_COLORS = {
    "Anaphor":     "#2196F3",
    "Determiner":  "#4CAF50",
    "Island":      "#F44336",
    "NPI":         "#FF9800",
    "Wh":          "#9C27B0",
    "Principle A": "#00BCD4",
    "Agreement":   "#795548",
    "Other":       "#9E9E9E",
}


# =============================================================================
# DISTRIBUTED HELPERS
# =============================================================================

def setup_dist(rank: int, world_size: int):
    """Initialise process group (NCCL for H100s, Gloo fallback for CPU)."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_dist():
    dist.destroy_process_group()


def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available() else 1


def barrier():
    if is_dist_available():
        dist.barrier()


# =============================================================================
# LOADING
# =============================================================================

def load_sae(layer: int, device: str) -> SAE:
    subfolder = f"layer_{layer:02d}"
    local_dir = os.path.join(SAE_CACHE, subfolder)
    # Only rank 0 downloads; others wait at the barrier.
    if get_rank() == 0:
        snapshot_download(
            repo_id=SAE_REPO,
            repo_type="model",
            allow_patterns=[f"{subfolder}/*"],
            local_dir=local_dir,
        )
    barrier()  # all ranks wait until download is done

    sae = SAE.load_from_disk(os.path.join(local_dir, subfolder), device=device)
    sae.eval()
    scaler_path = os.path.join(local_dir, subfolder, "activation_scaler.json")
    if os.path.exists(scaler_path):
        sf = json.load(open(scaler_path)).get("scaling_factor")
        if sf is not None:
            sae.fold_activation_norm_scaling_factor(sf)
    return sae


# =============================================================================
# INFERENCE HELPERS
# =============================================================================

def get_features(model, tokenizer, sae, text: str, layer: int, device: str):
    """Returns (last_token_features [d_sae], mean_features [d_sae])."""
    cache = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cache["r"] = h.detach().squeeze(0)

    hook = model.transformer.h[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**tokenizer(text, return_tensors="pt").to(device))
    hook.remove()

    resid = cache["r"]          # [seq_len, 1024]
    acts  = sae.encode(resid)   # [seq_len, d_sae]
    return acts[-1].cpu(), acts.mean(0).cpu()


def get_ce(model, tokenizer, text: str, device: str) -> float:
    inp = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inp, labels=inp["input_ids"]).loss.item()


# =============================================================================
# PER-RANK DATA COLLECTION
# =============================================================================

def collect_task_data(
    task_name: str,
    model,
    tokenizer,
    sae,
    layer: int,
    n_samples: int,
    device: str,
) -> dict | None:
    try:
        ds = load_dataset(
            "nyu-mll/blimp", task_name,
            split="train",
        )
    except Exception:
        return None

    pairs = list(ds)[:n_samples]
    gram_last, gram_mean       = [], []
    ungram_last, ungram_mean   = [], []
    correct_list               = []

    for item in pairs:
        good = item["sentence_good"]
        bad  = item["sentence_bad"]

        ce_g = get_ce(model, tokenizer, good, device)
        ce_b = get_ce(model, tokenizer, bad,  device)
        correct = ce_g < ce_b

        g_last, g_mean = get_features(model, tokenizer, sae, good, layer, device)
        b_last, b_mean = get_features(model, tokenizer, sae, bad,  layer, device)

        gram_last.append(g_last);   gram_mean.append(g_mean)
        ungram_last.append(b_last); ungram_mean.append(b_mean)
        correct_list.append(correct)

    correct_t = torch.tensor(correct_list)
    return {
        "task":          task_name,
        "benchmark_acc": BENCHMARK_ACC.get(task_name, 0.0),
        "empirical_acc": correct_t.float().mean().item(),
        "gram_last":     torch.stack(gram_last),
        "gram_mean":     torch.stack(gram_mean),
        "ungram_last":   torch.stack(ungram_last),
        "ungram_mean":   torch.stack(ungram_mean),
        "correct":       correct_t,
    }


# =============================================================================
# ANALYSIS HELPER
# =============================================================================

def compute_delta(res: dict) -> dict:
    g, u, c = res["gram_last"], res["ungram_last"], res["correct"]
    delta_all     = (g - u).mean(0)
    delta_correct = (g[c]  - u[c] ).mean(0) if c.sum()    > 0 else torch.zeros_like(delta_all)
    delta_wrong   = (g[~c] - u[~c]).mean(0) if (~c).sum() > 0 else torch.zeros_like(delta_all)
    return {
        "delta_all":     delta_all,
        "delta_correct": delta_correct,
        "delta_wrong":   delta_wrong,
        "l1_all":        delta_all.abs().sum().item(),
        "l1_correct":    delta_correct.abs().sum().item(),
        "l1_wrong":      delta_wrong.abs().sum().item(),
    }


def detect_outliers(task_data: dict) -> set:
    """IQR-based outlier detection on L1 delta values (3×IQR rule)."""
    l1_vals = {k: compute_delta(v)["l1_all"] for k, v in task_data.items()}
    arr = np.array(list(l1_vals.values()))
    Q1, Q3 = np.percentile(arr, [25, 75])
    cap = Q3 + 3.0 * (Q3 - Q1)
    return {k for k, v in l1_vals.items() if v > cap}


# =============================================================================
# FIGURE 1 — Accuracy vs L1 Feature Delta
# =============================================================================

def fig1_scatter(task_data: dict, layer: int, out_dir: str):
    outliers = detect_outliers(task_data)

    task_info = {}
    for task_name, res in task_data.items():
        d   = compute_delta(res)
        cat = TASK_TO_CATEGORY.get(task_name, "Other")
        task_info[task_name] = {
            "acc":        res["empirical_acc"],
            "l1":         d["l1_all"],
            "color":      CATEGORY_COLORS.get(cat, "#9E9E9E"),
            "is_outlier": task_name in outliers,
        }

    accs_all   = np.array([v["acc"] for v in task_info.values()])
    deltas_all = np.array([v["l1"]  for v in task_info.values()])
    mask_clean = np.array([not v["is_outlier"] for v in task_info.values()])
    r, p             = stats.pearsonr(accs_all, deltas_all)
    r_clean, p_clean = stats.pearsonr(accs_all[mask_clean], deltas_all[mask_clean])

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, info in task_info.items():
        if info["is_outlier"]:
            continue
        ax.scatter(info["acc"], info["l1"], c=info["color"], s=60, alpha=0.8,
                   edgecolors="white", linewidth=0.5, zorder=3)

    for name, info in task_info.items():
        if not info["is_outlier"]:
            continue
        ax.scatter(info["acc"], info["l1"], c=info["color"], s=220, alpha=0.9,
                   marker="*", edgecolors="black", linewidth=0.8, zorder=4)
        ax.annotate(name.replace("_", " ")[:30], (info["acc"], info["l1"]),
                    xytext=(6, -4), textcoords="offset points", fontsize=7, style="italic")

    m, b   = np.polyfit(accs_all, deltas_all, 1)
    x_line = np.linspace(accs_all.min() - 0.02, accs_all.max() + 0.02, 100)
    ax.plot(x_line, m * x_line + b, color="#333333", linewidth=1.5,
            linestyle="--", alpha=0.7, zorder=2)
    n      = len(accs_all)
    stderr = np.sqrt(np.sum((deltas_all - (m * accs_all + b))**2) / (n - 2))
    conf   = 1.96 * stderr * np.sqrt(
        1/n + (x_line - accs_all.mean())**2 / np.sum((accs_all - accs_all.mean())**2)
    )
    ax.fill_between(x_line, m*x_line+b-conf, m*x_line+b+conf,
                    alpha=0.12, color="#333333", zorder=1)

    ax.axvline(STRONG_THRESHOLD, color="#4CAF50", linewidth=1, linestyle=":", alpha=0.6)
    ax.axvline(WEAK_THRESHOLD,   color="#F44336", linewidth=1, linestyle=":", alpha=0.6)

    ax.text(0.97, 0.97,
            f"Pearson $r = {r:.3f}$, $p = {p:.4f}$\n"
            f"excl. outliers: $r = {r_clean:.3f}$, $p = {p_clean:.4f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3", alpha=0.9))

    if outliers:
        ax.text(0.5, -0.12,
                "★ outlier (3×IQR): " + ", ".join(sorted(outliers)),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, style="italic", color="#666666")

    legend_patches = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in CATEGORY_COLORS.items()
        if any(TASK_TO_CATEGORY.get(t) == cat for t in task_data)
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              framealpha=0.9, fontsize=9, ncol=2)

    ax.set_xlabel("BLiMP Task Accuracy", fontweight="bold")
    ax.set_ylabel("L1 Norm of SAE Feature Delta\n(grammatical − ungrammatical)",
                  fontweight="bold")
    ax.set_title(f"SAE Feature Separation vs. Grammatical Accuracy\n"
                 f"GPT-2 Medium (BabyLM) · Layer {layer}", fontweight="bold")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, deltas_all.max() * 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, f"fig1_l1_delta_scatter_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 1 saved: {path}  (r={r:.3f}; excl. outliers: r={r_clean:.3f}, p={p_clean:.4f})")
    return r, p


# =============================================================================
# FIGURE 2 — Strong vs Weak Bar Chart
# =============================================================================

def fig2_strong_weak_bar(task_data: dict, layer: int, out_dir: str):
    outliers = detect_outliers(task_data)
    strong_l1, weak_l1, mid_l1 = [], [], []
    strong_l1c, weak_l1c, mid_l1c = [], [], []  # clean = outliers excluded

    for task_name, res in task_data.items():
        d      = compute_delta(res)
        acc    = res["empirical_acc"]
        l1     = d["l1_all"]
        is_out = task_name in outliers
        if acc >= STRONG_THRESHOLD:
            strong_l1.append(l1)
            if not is_out: strong_l1c.append(l1)
        elif acc <= WEAK_THRESHOLD:
            weak_l1.append(l1)
            if not is_out: weak_l1c.append(l1)
        else:
            mid_l1.append(l1)
            if not is_out: mid_l1c.append(l1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 20))

    groups     = ["Strong\n(acc ≥ 0.80)", "Mid\n(0.40–0.80)", "Weak\n(acc ≤ 0.40)"]
    clean_grps = [strong_l1c, mid_l1c, weak_l1c]
    all_ns     = [len(strong_l1), len(mid_l1), len(weak_l1)]
    means      = [np.mean(g) if g else 0 for g in clean_grps]
    sems       = [np.std(g) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in clean_grps]
    bar_colors = ["#4CAF50", "#FFC107", "#F44336"]

    bars = axes[0].bar(groups, means, color=bar_colors, alpha=0.85,
                       edgecolor="white", linewidth=1.2,
                       yerr=sems, capsize=5, error_kw={"linewidth": 1.5})

    for bar, n_all, n_cl in zip(bars, all_ns, [len(g) for g in clean_grps]):
        label = f"n={n_cl}" if n_all == n_cl else f"n={n_cl} (of {n_all})"
        y_top = bar.get_height() + (max(sems) * 0.5 if max(sems) > 0 else bar.get_height() * 0.05)
        axes[0].text(bar.get_x() + bar.get_width()/2, y_top,
                     label, ha="center", va="bottom", fontsize=10)

    axes[0].set_ylabel("Mean L1 Norm of SAE Feature Delta", fontweight="bold")
    subtitle = f"outliers excl: {', '.join(sorted(outliers))}" if outliers else "no outliers"
    axes[0].set_title(f"SAE Grammaticality Signal by Task Difficulty\nLayer {layer} ({subtitle})",
                      fontweight="bold", fontsize=10)

    if strong_l1c and weak_l1c:
        t_stat, p_val = stats.ttest_ind(strong_l1c, weak_l1c)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        y_max = max(means) + max(sems) * 2
        axes[0].annotate("", xy=(2, y_max * 1.02), xytext=(0, y_max * 1.02),
                         arrowprops=dict(arrowstyle="-", lw=1.2))
        axes[0].text(1, y_max * 1.04, sig, ha="center", fontsize=13, fontweight="bold")

    sorted_tasks = sorted(task_data.items(), key=lambda x: x[1]["empirical_acc"])
    task_l1s     = [compute_delta(r)["l1_all"] for _, r in sorted_tasks]
    task_cols    = []
    tick_labels  = []
    for t, res in sorted_tasks:
        a = res["empirical_acc"]
        if a >= STRONG_THRESHOLD:   task_cols.append("#4CAF50")
        elif a <= WEAK_THRESHOLD:   task_cols.append("#F44336")
        else:                        task_cols.append("#FFC107")
        tick_labels.append(("★ " if t in outliers else "") + t[:22])

    y_pos = np.arange(len(tick_labels))
    axes[1].barh(y_pos, task_l1s, color=task_cols, alpha=0.8, edgecolor="white")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(tick_labels, fontsize=5)
    axes[1].tick_params(axis="y", pad=2)
    axes[1].set_xlabel("L1 Feature Delta", fontweight="bold")
    axes[1].set_title("Per-Task L1 Delta\n(sorted by accuracy)", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label=f"Strong (≥{STRONG_THRESHOLD})"),
        mpatches.Patch(color="#FFC107", label="Mid"),
        mpatches.Patch(color="#F44336", label=f"Weak (≤{WEAK_THRESHOLD})"),
    ]
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f"fig2_strong_weak_bar_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 2 saved: {path}")


# =============================================================================
# FIGURE 3 — Category Cosine Similarity Heatmap
# =============================================================================

def fig3_category_heatmap(task_data: dict, layer: int, out_dir: str):
    cat_delta = {}
    for cat_name, task_list in CATEGORY_MAP.items():
        deltas = [
            compute_delta(task_data[t])["delta_all"]
            for t in task_list if t in task_data
        ]
        if deltas:
            cat_delta[cat_name] = torch.stack(deltas).mean(0)

    cats  = list(cat_delta.keys())
    n_cat = len(cats)
    sim_matrix = np.zeros((n_cat, n_cat))
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            sim_matrix[i, j] = F.cosine_similarity(
                cat_delta[c1].unsqueeze(0),
                cat_delta[c2].unsqueeze(0),
            ).item()

    cat_accs = {
        cat: np.mean([BENCHMARK_ACC.get(t, 0) for t in tasks if t in BENCHMARK_ACC])
        for cat, tasks in CATEGORY_MAP.items()
    }
    cat_labels = [f"{c}\n(acc={cat_accs.get(c, 0):.2f})" for c in cats]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im   = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=11)

    ax.set_xticks(range(n_cat)); ax.set_yticks(range(n_cat))
    ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(cat_labels, fontsize=9)

    for i in range(n_cat):
        for j in range(n_cat):
            val   = sim_matrix[i, j]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title(f"Cosine Similarity of SAE Grammaticality Signals\n"
                 f"by Linguistic Category · Layer {layer}",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(out_dir, f"fig3_category_heatmap_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 3 saved: {path}")
    return sim_matrix, cats


# =============================================================================
# FIGURE 4 — Strong vs Weak Task Feature Contrast (main analysis)
# =============================================================================

def fig4_strong_vs_weak(task_data: dict, layer: int, out_dir: str, top_k: int = 20):
    """Which SAE features encode grammaticality in strong tasks vs weak tasks?"""
    outliers = detect_outliers(task_data)
    strong_deltas, weak_deltas = [], []

    for task_name, res in task_data.items():
        if task_name in outliers:
            continue
        d   = compute_delta(res)
        acc = res["empirical_acc"]
        if acc >= STRONG_THRESHOLD:
            strong_deltas.append(d["delta_all"])
        elif acc <= WEAK_THRESHOLD:
            weak_deltas.append(d["delta_all"])

    if not strong_deltas or not weak_deltas:
        print("  ⚠  Not enough data for Fig 4 (strong vs weak)")
        return

    strong_mean = torch.stack(strong_deltas).mean(0)  # [d_sae]
    weak_mean   = torch.stack(weak_deltas).mean(0)    # [d_sae]

    top_strong = set(strong_mean.topk(top_k).indices.tolist())
    top_weak   = set(weak_mean.topk(top_k).indices.tolist())
    shared     = top_strong & top_weak

    strong_list = sorted(top_strong, key=lambda i: -strong_mean[i].item())
    weak_list   = sorted(top_weak,   key=lambda i: -weak_mean[i].item())
    shared_list = sorted(shared,     key=lambda i: -(strong_mean[i] + weak_mean[i]).item() / 2)
    if not shared_list:
        shared_list = strong_list[:min(5, len(strong_list))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    width = 0.38

    for ax, feat_ids, title in [
        (axes[0], strong_list, f"Top {top_k} Features: Strong Tasks\n(ranked by strong gram−ungram delta)"),
        (axes[1], weak_list,   f"Top {top_k} Features: Weak Tasks\n(ranked by weak gram−ungram delta)"),
        (axes[2], shared_list, f"Shared Features (in both top-{top_k})\n({len(shared)} found)"),
    ]:
        n     = len(feat_ids)
        y     = np.arange(n)
        svals = [strong_mean[i].item() for i in feat_ids]
        wvals = [weak_mean[i].item()   for i in feat_ids]

        ax.barh(y + width/2, svals, width, color="#4CAF50", alpha=0.85, label="Strong tasks")
        ax.barh(y - width/2, wvals, width, color="#F44336", alpha=0.85, label="Weak tasks")
        ax.set_yticks(y)
        ax.set_yticklabels([f"Feat {i}" for i in feat_ids], fontsize=8)
        ax.set_xlabel("Mean SAE Feature Delta\n(grammatical − ungrammatical)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.invert_yaxis()

    fig.suptitle(f"SAE Feature Contrast: Strong vs Weak BLiMP Tasks · Layer {layer}\n"
                 f"GPT-2 Medium (BabyLM)  —  {len(shared)} shared features in top-{top_k}",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig4_strong_vs_weak_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 4 saved: {path}  ({len(shared)} shared features in top-{top_k})")


# =============================================================================
# FIGURE 4b — Top Features Predicting Correctness (within-weak-task analysis)
# =============================================================================

def fig4b_correctness_features(task_data: dict, layer: int, out_dir: str,
                                top_k: int = 25):
    weak_correct, weak_wrong, strong_correct = [], [], []

    for task_name, res in task_data.items():
        c   = res["correct"]
        g   = res["gram_last"]
        acc = res["empirical_acc"]
        if acc <= WEAK_THRESHOLD:
            if c.sum()    > 0: weak_correct.append(g[c].mean(0))
            if (~c).sum() > 0: weak_wrong.append(g[~c].mean(0))
        elif acc >= STRONG_THRESHOLD:
            if c.sum() > 0:    strong_correct.append(g[c].mean(0))

    if not weak_correct or not weak_wrong:
        print("  ⚠  Not enough data for Fig 4")
        return

    wc    = torch.stack(weak_correct).mean(0)
    ww    = torch.stack(weak_wrong).mean(0)
    sc    = torch.stack(strong_correct).mean(0) if strong_correct else torch.zeros_like(wc)
    delta = wc - ww

    top_pos = delta.topk(top_k)
    top_neg = (-delta).topk(top_k)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = np.arange(top_k); width = 0.28

    for ax_idx, (feat_ids, title) in enumerate([
        (top_pos.indices.tolist(), f"Features Predicting Correct\nJudgment on Weak Tasks (Layer {layer})"),
        (top_neg.indices.tolist(), f"Features Active on Incorrect\nJudgments (Misleading) · Layer {layer}"),
    ]):
        ax = axes[ax_idx]
        wc_vals = [wc[i].item() for i in feat_ids]
        ww_vals = [ww[i].item() for i in feat_ids]
        sc_vals = [sc[i].item() for i in feat_ids]

        ax.barh(y + width, wc_vals, width, color="#4CAF50", alpha=0.85, label="Weak task correct")
        ax.barh(y,         ww_vals, width, color="#F44336", alpha=0.85, label="Weak task wrong")
        ax.barh(y - width, sc_vals, width, color="#2196F3", alpha=0.85, label="Strong task correct")

        ax.set_yticks(y)
        ax.set_yticklabels([f"Feat {i}" for i in feat_ids], fontsize=8)
        ax.set_xlabel("Mean Feature Activation", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.invert_yaxis()

    fig.suptitle("SAE Features and Model Correctness on Hard BLiMP Phenomena\n"
                 f"GPT-2 Medium (BabyLM)", fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig4b_correctness_features_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 4b saved: {path}")


# =============================================================================
# FIGURE 5 — PCA of task delta vectors (feature-space geometry)
# =============================================================================

def fig5_task_pca(task_data: dict, layer: int, out_dir: str):
    """
    Project per-task grammaticality delta vectors onto their first two principal
    components.  Normalising to unit length before PCA captures the *direction*
    of each task's grammaticality signal rather than its magnitude, so the plot
    reveals whether linguistic categories occupy distinct regions of SAE space
    (cf. Anthropic "Geometry of Concepts", 2024).
    """
    outliers = detect_outliers(task_data)

    task_names, delta_vecs, accs, cats = [], [], [], []
    for task_name, res in task_data.items():
        d = compute_delta(res)
        task_names.append(task_name)
        delta_vecs.append(d["delta_all"].detach().numpy())
        accs.append(res["empirical_acc"])
        cats.append(TASK_TO_CATEGORY.get(task_name, "Other"))

    X = np.stack(delta_vecs)  # [n_tasks, d_sae]
    # Normalise rows to unit length → PCA captures direction, not magnitude
    norms  = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms > 1e-8, norms, 1.0)

    # PCA via truncated SVD (no sklearn dependency)
    X_c = X_norm - X_norm.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_2d    = X_c @ Vt[:2].T                   # [n_tasks, 2]
    var_exp = S[:2]**2 / (S**2).sum()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Soft confidence ellipses per linguistic category
    for cat, color in CATEGORY_COLORS.items():
        idx = [i for i, c in enumerate(cats) if c == cat]
        if len(idx) < 3:
            continue
        pts = X_2d[idx]
        mu  = pts.mean(0)
        cov = np.cov(pts.T)
        try:
            evals, evecs = np.linalg.eigh(cov)
            evals = np.abs(evals)
            angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
            ell   = mpatches.Ellipse(mu, 2 * np.sqrt(evals[1]) * 1.5,
                                     2 * np.sqrt(evals[0]) * 1.5,
                                     angle=angle, color=color, alpha=0.08, zorder=1)
            ax.add_patch(ell)
        except np.linalg.LinAlgError:
            pass

    # Scatter points (size ∝ empirical accuracy)
    for i, (task_name, xy, acc, cat) in enumerate(zip(task_names, X_2d, accs, cats)):
        color  = CATEGORY_COLORS.get(cat, "#9E9E9E")
        is_out = task_name in outliers
        ax.scatter(xy[0], xy[1],
                   c=color, s=(300 if is_out else 50 + acc * 150),
                   alpha=0.9,
                   edgecolors="black" if is_out else "white",
                   linewidth=0.8, marker=("*" if is_out else "o"), zorder=3)

    # Label only outliers (to keep the plot readable)
    for task_name, xy in zip(task_names, X_2d):
        if task_name in outliers:
            ax.annotate(task_name.replace("_", " ")[:30], xy,
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7, style="italic",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.axhline(0, color="#cccccc", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="#cccccc", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"PC 1  ({var_exp[0]*100:.1f}% variance explained)", fontweight="bold")
    ax.set_ylabel(f"PC 2  ({var_exp[1]*100:.1f}% variance explained)", fontweight="bold")
    ax.set_title(f"Geometry of Task Grammaticality Signals in SAE Space\n"
                 f"GPT-2 Medium (BabyLM) · Layer {layer}  "
                 f"(unit-normalised delta vectors · size ∝ accuracy)",
                 fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in CATEGORY_COLORS.items()
        if any(TASK_TO_CATEGORY.get(t) == cat for t in task_data)
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"fig5_task_pca_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 5 saved: {path}  (PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%)")
    return X_2d, task_names, var_exp


# =============================================================================
# FIGURE 6 — Per-feature paired t-statistics (selectivity, Cunningham et al.)
# =============================================================================

def fig6_feature_tstat(task_data: dict, layer: int, out_dir: str, top_k: int = 20):
    """
    For each SAE feature, compute the paired t-statistic across all
    (grammatical − ungrammatical) pairs in strong / weak tasks separately.
    This is the selectivity metric from Cunningham et al. (2023) and is more
    statistically principled than raw mean delta because it accounts for
    within-feature variance across samples.
    """
    outliers = detect_outliers(task_data)
    strong_diffs, weak_diffs = [], []

    for task_name, res in task_data.items():
        if task_name in outliers:
            continue
        diff = res["gram_last"] - res["ungram_last"]  # [n, d_sae]
        acc  = res["empirical_acc"]
        if acc >= STRONG_THRESHOLD:
            strong_diffs.append(diff)
        elif acc <= WEAK_THRESHOLD:
            weak_diffs.append(diff)

    if not strong_diffs or not weak_diffs:
        print("  ⚠  Not enough data for Fig 6")
        return

    strong_cat = torch.cat(strong_diffs, dim=0)  # [N_strong_samples, d_sae]
    weak_cat   = torch.cat(weak_diffs,   dim=0)  # [N_weak_samples,   d_sae]

    def paired_tstat(diffs: torch.Tensor) -> torch.Tensor:
        """Paired t = mean(diff) / (std(diff) / √n).  Positive → gram > ungram."""
        mu  = diffs.mean(0)
        std = diffs.std(0).clamp(min=1e-8)
        return mu / (std / diffs.shape[0] ** 0.5)

    t_strong = paired_tstat(strong_cat)  # [d_sae]
    t_weak   = paired_tstat(weak_cat)    # [d_sae]

    top_strong_idx = t_strong.topk(top_k).indices.tolist()
    top_weak_idx   = t_weak.topk(top_k).indices.tolist()
    shared         = set(top_strong_idx) & set(top_weak_idx)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    width = 0.38

    for ax, feat_ids, t_primary, t_other, label, pcolor in [
        (axes[0], top_strong_idx, t_strong, t_weak,
         f"Strong tasks  (acc ≥ {STRONG_THRESHOLD})", "#4CAF50"),
        (axes[1], top_weak_idx,   t_weak,   t_strong,
         f"Weak tasks  (acc ≤ {WEAK_THRESHOLD})",   "#F44336"),
    ]:
        n = len(feat_ids)
        y = np.arange(n)
        pvals = [t_primary[i].item() for i in feat_ids]
        ovals = [t_other[i].item()   for i in feat_ids]

        ax.barh(y + width/2, pvals, width, color=pcolor,   alpha=0.85, label=label)
        ax.barh(y - width/2, ovals, width, color="#9E9E9E", alpha=0.65, label="Other group")

        for j, fi in enumerate(feat_ids):
            if fi in shared:
                ax.text(max(0, pvals[j]) + 0.05, j, "★",
                        fontsize=10, va="center", ha="left", color="navy", zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels([f"Feat {i}" for i in feat_ids], fontsize=8)
        ax.set_xlabel("Paired t-statistic  (gram − ungram)\n"
                      "↑ positive = feature more active on grammatical sentences",
                      fontweight="bold")
        ax.set_title(f"Top {top_k} by t-stat\n{label}", fontweight="bold")
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc="lower right")
        ax.invert_yaxis()

    fig.suptitle(f"Feature Grammaticality Selectivity (Cunningham et al.) · Layer {layer}\n"
                 f"GPT-2 Medium (BabyLM)  —  "
                 f"★ = shared in both top-{top_k}  (n={len(shared)})",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig6_feature_tstat_layer{layer}.pdf")
    plt.savefig(path); plt.savefig(path.replace(".pdf", ".png")); plt.close()
    print(f"  Fig 6 saved: {path}  ({len(shared)} shared features in top-{top_k})")


# =============================================================================
# SAVE JSON
# =============================================================================

def save_json(task_data: dict, layer: int, out_dir: str) -> dict:
    outliers = detect_outliers(task_data)
    summary  = {}
    for task_name, res in task_data.items():
        d = compute_delta(res)
        summary[task_name] = {
            "benchmark_acc":    res["benchmark_acc"],
            "empirical_acc":    res["empirical_acc"],
            "n_correct":        int(res["correct"].sum().item()),
            "n_wrong":          int((~res["correct"]).sum().item()),
            "l1_delta_all":     d["l1_all"],
            "l1_delta_correct": d["l1_correct"],
            "l1_delta_wrong":   d["l1_wrong"],
            "category":         TASK_TO_CATEGORY.get(task_name, "other"),
            "is_outlier":       task_name in outliers,
        }

    # Exclude outliers from summary statistics so they don't inflate means
    strong = {k: v for k, v in summary.items()
              if v["empirical_acc"] >= STRONG_THRESHOLD and not v["is_outlier"]}
    weak   = {k: v for k, v in summary.items()
              if v["empirical_acc"] <= WEAK_THRESHOLD   and not v["is_outlier"]}

    out = {
        "layer":           layer,
        "n_tasks":         len(summary),
        "n_outliers":      len(outliers),
        "outlier_tasks":   sorted(outliers),
        "mean_strong_l1":  np.mean([v["l1_delta_all"] for v in strong.values()]) if strong else 0,
        "mean_weak_l1":    np.mean([v["l1_delta_all"] for v in weak.values()])   if weak   else 0,
        "tasks":           summary,
    }
    path = os.path.join(out_dir, f"layer{layer:02d}_contrast_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON saved: {path}")
    return out


# =============================================================================
# GATHER HELPER — uses gather_object for NCCL compatibility
# =============================================================================

def _gather_dicts(local_dict: dict, rank: int, world_size: int) -> list[dict]:
    """
    Uses torch.distributed.gather_object to collect dicts from all ranks onto rank 0.
    This avoids the NCCL CPU-tensor problem of the previous send/recv approach.
    """
    if rank == 0:
        gathered = [None] * world_size
        dist.gather_object(local_dict, object_gather_list=gathered, dst=0)
        return gathered
    else:
        dist.gather_object(local_dict, object_gather_list=None, dst=0)
        return [local_dict]


# =============================================================================
# RANK WORKER — runs on each GPU
# =============================================================================

def worker(rank: int, world_size: int, args: argparse.Namespace):
    """
    Each rank:
      1. Initialises its own process group slot.
      2. Loads model + SAE onto its dedicated GPU.
      3. Processes its shard of tasks.
      4. Rank 0 collects all shards, generates figures, saves JSON.
    """
    # ── Setup ──────────────────────────────────────────────────────────────────
    setup_dist(rank, world_size)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    layers = [2, 4, 6, 8, 10, 12, 16, 22] if args.all_layers else [args.layer]

    # ── Load model (each rank loads independently — no shared memory needed) ──
    if rank == 0:
        print(f"\n{'═'*62}")
        print(f"  BLiMP Contrastive SAE Analysis  [multi-GPU  ×{world_size}]")
        print(f"  Tasks  : {len(BENCHMARK_ACC)}   n/task : {args.n_samples}")
        print(f"  Device : {device.upper()}")
        print(f"{'═'*62}")

    tokenizer = AutoTokenizer.from_pretrained(GPT2_REPO)
    model     = AutoModelForCausalLM.from_pretrained(GPT2_REPO).to(device).eval()

    if rank == 0:
        print(f"  ✓  GPT-2 loaded: {sum(p.numel() for p in model.parameters()):,} params  [rank 0]")

    # ── Per-layer loop ─────────────────────────────────────────────────────────
    for layer in layers:
        barrier()  # all ranks start each layer together

        sae = load_sae(layer, device)

        if rank == 0:
            print(f"\n{'─'*62}")
            print(f"  Layer {layer:02d}  |  d_sae={sae.cfg.d_sae}  |  world_size={world_size}")

        # ── Shard task list across ranks ───────────────────────────────────────
        all_tasks   = list(BENCHMARK_ACC.keys())
        local_tasks = all_tasks[rank::world_size]   # round-robin sharding

        if rank == 0:
            print(f"  Tasks per rank ≈ {len(all_tasks)//world_size}  "
                  f"(rank 0 handles {len(local_tasks)})")

        # ── Process local shard ────────────────────────────────────────────────
        local_data: dict[str, dict] = {}
        desc = f"[rank {rank}] layer {layer:02d}"
        for task_name in tqdm(local_tasks, desc=desc, position=rank, leave=False):
            res = collect_task_data(
                task_name, model, tokenizer, sae, layer, args.n_samples, device
            )
            if res is not None:
                # Move tensors to CPU before gathering
                res["gram_last"]   = res["gram_last"].cpu()
                res["gram_mean"]   = res["gram_mean"].cpu()
                res["ungram_last"] = res["ungram_last"].cpu()
                res["ungram_mean"] = res["ungram_mean"].cpu()
                res["correct"]     = res["correct"].cpu()
                local_data[task_name] = res

        barrier()

        # ── Gather all shards onto rank 0 via gather_object ────────────────────
        gathered = _gather_dicts(local_data, rank, world_size)

        # ── Rank 0: merge, analyse, plot ───────────────────────────────────────
        if rank == 0:
            task_data = {}
            for shard in gathered:
                task_data.update(shard)

            print(f"\n  Collected: {len(task_data)}/{len(BENCHMARK_ACC)} tasks")

            strong_l1 = [compute_delta(r)["l1_all"] for r in task_data.values()
                         if r["empirical_acc"] >= STRONG_THRESHOLD]
            weak_l1   = [compute_delta(r)["l1_all"] for r in task_data.values()
                         if r["empirical_acc"] <= WEAK_THRESHOLD]

            print(f"  Strong tasks (n={len(strong_l1)})  mean L1Δ = "
                  f"{np.mean(strong_l1):.2f}" if strong_l1 else "  No strong tasks")
            print(f"  Weak tasks   (n={len(weak_l1)})  mean L1Δ = "
                  f"{np.mean(weak_l1):.2f}" if weak_l1 else "  No weak tasks")

            if strong_l1 and weak_l1:
                t_stat, p_val = stats.ttest_ind(strong_l1, weak_l1)
                print(f"  t-test: t={t_stat:.2f}  p={p_val:.4f}")

            os.makedirs(FIGURES_DIR, exist_ok=True)
            os.makedirs(RESULTS_DIR, exist_ok=True)

            print(f"\n  Generating figures → {FIGURES_DIR}/")
            fig1_scatter(task_data, layer, FIGURES_DIR)
            fig2_strong_weak_bar(task_data, layer, FIGURES_DIR)
            fig3_category_heatmap(task_data, layer, FIGURES_DIR)
            fig4_strong_vs_weak(task_data, layer, FIGURES_DIR)
            fig4b_correctness_features(task_data, layer, FIGURES_DIR)
            fig5_task_pca(task_data, layer, FIGURES_DIR)
            fig6_feature_tstat(task_data, layer, FIGURES_DIR)
            save_json(task_data, layer, RESULTS_DIR)

        barrier()  # all ranks wait before next layer / cleanup

    # ── Teardown ───────────────────────────────────────────────────────────────
    cleanup_dist()

    if rank == 0:
        print(f"\n{'═'*62}")
        print(f"  All done.")
        print(f"  Results : {RESULTS_DIR}/")
        print(f"  Figures : {FIGURES_DIR}/  (PDF + PNG)")
        print(f"{'═'*62}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BLiMP contrastive SAE analysis (multi-GPU aware)"
    )
    parser.add_argument("--layer",      type=int,  default=12,
                        help="SAE layer to analyse (default 12)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Run layers 2,4,6,8,10,12,16,22")
    parser.add_argument("--n-samples",  type=int,  default=200,
                        help="Sentence pairs per task (default 200; max 1000)")
    args = parser.parse_args()

    # ── torchrun sets RANK / WORLD_SIZE / LOCAL_RANK automatically ─────────────
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # LOCAL_RANK matches the GPU index on a single-node job.
    effective_rank = local_rank if "LOCAL_RANK" in os.environ else rank

    if world_size > 1:
        worker(effective_rank, world_size, args)
    else:
        # Single-process fallback
        print("  [single-GPU mode]  use `torchrun --nproc_per_node=8` for multi-GPU")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(GPT2_REPO)
        model     = AutoModelForCausalLM.from_pretrained(GPT2_REPO).to(device).eval()
        layers    = [2, 4, 6, 8, 10, 12, 16, 22] if args.all_layers else [args.layer]

        for layer in layers:
            sae = load_sae(layer, device)
            task_data: dict[str, dict] = {}
            for task_name in tqdm(BENCHMARK_ACC.keys(), desc=f"Layer {layer:02d}"):
                res = collect_task_data(
                    task_name, model, tokenizer, sae, layer, args.n_samples, device
                )
                if res is not None:
                    task_data[task_name] = res

            fig1_scatter(task_data, layer, FIGURES_DIR)
            fig2_strong_weak_bar(task_data, layer, FIGURES_DIR)
            fig3_category_heatmap(task_data, layer, FIGURES_DIR)
            fig4_strong_vs_weak(task_data, layer, FIGURES_DIR)
            fig4b_correctness_features(task_data, layer, FIGURES_DIR)
            fig5_task_pca(task_data, layer, FIGURES_DIR)
            fig6_feature_tstat(task_data, layer, FIGURES_DIR)
            save_json(task_data, layer, RESULTS_DIR)


if __name__ == "__main__":
    main()