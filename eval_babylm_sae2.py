"""
eval_babylm_sae.py
══════════════════════════════════════════════════════════════════════════════
Complete evaluation & inference toolkit for BabyLM GPT-2 Medium SAEs.
Uses real BLiMP dataset from HuggingFace (nyu-mll/blimp) for evaluation.

Usage:
    pip install datasets tqdm matplotlib seaborn

    python eval_babylm_sae.py --task basic     --layer 12
    python eval_babylm_sae.py --task quality   --layer 12
    python eval_babylm_sae.py --task features  --layer 12
    python eval_babylm_sae.py --task blimp     --layer 12
    python eval_babylm_sae.py --task blimp     --layer 12 --all-blimp-tasks
    python eval_babylm_sae.py --task patching  --layer 12
    python eval_babylm_sae.py --task compare   --layers 2 6 8 10 12 16 22
    python eval_babylm_sae.py --task all       --layer 12
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

GPT2_REPO = "IParraMartin/gpt2-medium-bLM100M"
SAE_REPO  = "whitepenguin/gpt2-medium-bLM100M-SAE"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SAE_CACHE = os.path.expanduser("~/.cache/babylm_saes")

# All 67 BLiMP tasks on HuggingFace nyu-mll/blimp
ALL_BLIMP_TASKS = [
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
    "animate_subject_passive", "animate_subject_trans", "causative",
    "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1", "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause", "drop_argument",
    "ellipsis_n_bar_1", "ellipsis_n_bar_2", "existential_there_object_raising",
    "existential_there_quantifiers_1", "existential_there_quantifiers_2",
    "existential_there_subject_raising", "expletive_it_object_raising",
    "inchoative", "intransitive", "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs", "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2", "left_branch_island_echo_question",
    "left_branch_island_simple_question", "matrix_question_npi_licensor_present",
    "npi_present_1", "npi_present_2", "only_npi_licensor_present", "only_npi_scope",
    "passive_1", "passive_2", "principle_A_c_command", "principle_A_case_1",
    "principle_A_case_2", "principle_A_domain_1", "principle_A_domain_2",
    "principle_A_domain_3", "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope",
    "sentential_subject_island", "superlative_quantifiers_1", "superlative_quantifiers_2",
    "tough_vs_raising_1", "tough_vs_raising_2", "transitive", "wh_island",
    "wh_questions_object_gap", "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
]

TEST_SENTENCES = [
    "The child looked at the dog and smiled.",
    "Can you pass me the ball please?",
    "She went to the shop with her mother.",
    "The cat sat on the mat.",
    "Look at the big red balloon!",
    "The boys are playing in the garden.",
    "Mommy said we can have ice cream after dinner.",
    "He put the book on the table carefully.",
]


# =============================================================================
# UTILITIES
# =============================================================================
def safe_topk(acts: torch.Tensor, k: int = 5):
    n_active = int((acts > 0).sum().item())
    k_safe   = min(k, n_active)
    if k_safe == 0:
        return None, None
    top_v, top_i = acts.topk(k_safe)
    v_list = top_v.tolist()
    i_list = top_i.tolist()
    if not isinstance(v_list, list): v_list = [v_list]
    if not isinstance(i_list, list): i_list = [i_list]
    return v_list, i_list


def format_topk(acts: torch.Tensor, k: int = 5) -> str:
    top_v, top_i = safe_topk(acts, k)
    if top_v is None:
        return "(no active features)"
    return "  ".join(f"{idx}:{val:.2f}" for idx, val in zip(top_i, top_v))


# =============================================================================
# SAE LOADING
# =============================================================================
def load_sae(layer: int) -> SAE:
    subfolder = f"layer_{layer:02d}"
    local_dir = os.path.join(SAE_CACHE, subfolder)
    print(f"Loading SAE layer {layer:02d}...")
    snapshot_download(
        repo_id=SAE_REPO, repo_type="model",
        allow_patterns=[f"{subfolder}/*"],
        local_dir=local_dir, local_dir_use_symlinks=False,
    )
    sae_dir = os.path.join(local_dir, subfolder)
    sae     = SAE.load_from_disk(sae_dir, device=DEVICE)
    sae.eval()
    scaler_path = os.path.join(sae_dir, "activation_scaler.json")
    if os.path.exists(scaler_path):
        sf = json.load(open(scaler_path)).get("scaling_factor")
        if sf is not None:
            sae.fold_activation_norm_scaling_factor(sf)
            print(f"  ✓  Applied scaling factor: {sf:.6f}")
    print(f"  ✓  d_in={sae.cfg.d_in}  d_sae={sae.cfg.d_sae}")
    return sae


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model_and_tokenizer():
    print(f"Loading GPT-2: {GPT2_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(GPT2_REPO)
    model     = AutoModelForCausalLM.from_pretrained(GPT2_REPO).to(DEVICE).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓  {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")
    return model, tokenizer


# =============================================================================
# HOOK UTILITIES
# =============================================================================
def get_residual_stream(model, tokenizer, text: str, layer: int):
    cache = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cache["resid"] = h.detach().squeeze(0)   # [seq_len, 1024]
    hook   = model.transformer.h[layer].register_forward_hook(hook_fn)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    return cache["resid"], inputs["input_ids"]


def get_ce_loss(model, tokenizer, text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs, labels=inputs["input_ids"]).loss.item()


def get_ce_loss_sae_patched(model, tokenizer, sae: SAE, text: str, layer: int) -> float:
    def patch_hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        hidden   = out[0] if is_tuple else out
        recon    = sae.decode(sae.encode(hidden))
        return (recon,) + out[1:] if is_tuple else recon
    hook   = model.transformer.h[layer].register_forward_hook(patch_hook)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
    hook.remove()
    return loss


# =============================================================================
# TASK 1: BASIC INFERENCE
# =============================================================================
def task_basic(model, tokenizer, sae: SAE, layer: int):
    print(f"\n{'═'*62}")
    print(f"  TASK 1: Basic Inference  |  Layer {layer:02d}")
    print(f"{'═'*62}")
    text = "The child looked at the dog and smiled."
    print(f"\nInput: '{text}'\n")
    resid, input_ids = get_residual_stream(model, tokenizer, text, layer)
    tokens           = tokenizer.convert_ids_to_tokens(input_ids[0])
    feature_acts     = sae.encode(resid)
    recon            = sae.decode(feature_acts)
    print(f"  {'Token':<22} {'Active':>6}  Top-5 Features (idx: val)")
    print(f"  {'─'*22} {'─'*6}  {'─'*44}")
    for i, tok in enumerate(tokens):
        acts     = feature_acts[i]
        n_active = int((acts > 0).sum().item())
        print(f"  {tok:<22} {n_active:>6}  {format_topk(acts)}")
    mean_l0 = (feature_acts > 0).float().sum(-1).mean().item()
    mse     = F.mse_loss(recon, resid).item()
    print(f"\n  Mean L0: {mean_l0:.1f}  MSE: {mse:.4f}")


# =============================================================================
# TASK 2: RECONSTRUCTION QUALITY
# =============================================================================
def task_quality(model, tokenizer, sae: SAE, layer: int):
    print(f"\n{'═'*62}")
    print(f"  TASK 2: Reconstruction Quality  |  Layer {layer:02d}")
    print(f"{'═'*62}\n")
    all_l0, all_mse, ce_orig_list, ce_patch_list = [], [], [], []
    for text in TEST_SENTENCES:
        resid, _     = get_residual_stream(model, tokenizer, text, layer)
        feature_acts = sae.encode(resid)
        recon        = sae.decode(feature_acts)
        l0   = (feature_acts > 0).float().sum(-1).mean().item()
        mse  = F.mse_loss(recon, resid).item()
        ce_o = get_ce_loss(model, tokenizer, text)
        ce_p = get_ce_loss_sae_patched(model, tokenizer, sae, text, layer)
        rec  = max(0.0, (1 - (ce_p - ce_o) / max(ce_o, 1e-8)) * 100)
        all_l0.append(l0); all_mse.append(mse)
        ce_orig_list.append(ce_o); ce_patch_list.append(ce_p)
        print(f"  L0={l0:5.1f}  MSE={mse:.4f}  CE {ce_o:.3f}→{ce_p:.3f}  Rec={rec:5.1f}%  '{text[:38]}'")
    mn_l0  = sum(all_l0)  / len(all_l0)
    mn_mse = sum(all_mse) / len(all_mse)
    mn_o   = sum(ce_orig_list)  / len(ce_orig_list)
    mn_p   = sum(ce_patch_list) / len(ce_patch_list)
    mn_rec = max(0.0, (1 - (mn_p - mn_o) / mn_o) * 100)
    print(f"\n  Mean L0: {mn_l0:.1f}  MSE: {mn_mse:.4f}  CE Recovered: {mn_rec:.1f}%")
    return {"mean_l0": mn_l0, "mean_mse": mn_mse, "ce_recovery": mn_rec}


# =============================================================================
# TASK 3: FEATURE ANALYSIS
# =============================================================================
def task_features(model, tokenizer, sae: SAE, layer: int):
    print(f"\n{'═'*62}")
    print(f"  TASK 3: Feature Analysis  |  Layer {layer:02d}")
    print(f"{'═'*62}\n")
    feature_freq     = defaultdict(int)
    feature_maxval   = defaultdict(float)
    feature_examples = defaultdict(list)
    total_tokens     = 0
    for text in TEST_SENTENCES:
        resid, input_ids = get_residual_stream(model, tokenizer, text, layer)
        tokens           = tokenizer.convert_ids_to_tokens(input_ids[0])
        feature_acts     = sae.encode(resid)
        for i, tok in enumerate(tokens):
            acts       = feature_acts[i]
            active_idx = (acts > 0).nonzero(as_tuple=True)[0].tolist()
            for idx in active_idx:
                val = acts[idx].item()
                feature_freq[idx] += 1
                if val > feature_maxval[idx]: feature_maxval[idx] = val
                if len(feature_examples[idx]) < 3:
                    feature_examples[idx].append((tok, val))
            total_tokens += 1
    total_used = len(feature_freq)
    dead_count = sae.cfg.d_sae - total_used
    print(f"  Tokens: {total_tokens}  Features used: {total_used:,}/{sae.cfg.d_sae:,}  "
          f"Dead: {dead_count:,} ({dead_count/sae.cfg.d_sae*100:.1f}%)")
    top20 = sorted(feature_freq.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top-20 features:")
    print(f"  {'Feat':>8}  {'Freq':>6}  {'Max':>7}  Example tokens")
    for feat_idx, freq in top20:
        examples = ", ".join(f"{t}({v:.2f})" for t, v in feature_examples[feat_idx])
        print(f"  {feat_idx:>8}  {freq:>6}  {feature_maxval[feat_idx]:>7.3f}  {examples}")


# =============================================================================
# TASK 4: BLIMP EVALUATION — uses real HuggingFace BLiMP dataset
# =============================================================================
def task_blimp(model, tokenizer, sae: SAE, layer: int, all_tasks: bool = False):
    print(f"\n{'═'*62}")
    print(f"  TASK 4: BLiMP Evaluation  |  Layer {layer:02d}")
    print(f"  Dataset: nyu-mll/blimp (HuggingFace)")
    print(f"{'═'*62}")

    tasks_to_run = ALL_BLIMP_TASKS if all_tasks else [
        # representative subset covering key phenomena
        "anaphor_gender_agreement", "anaphor_number_agreement",
        "determiner_noun_agreement_1", "determiner_noun_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "distractor_agreement_relative_clause",
        "npi_present_1", "sentential_negation_npi_scope",
        "left_branch_island_simple_question",
        "coordinate_structure_constraint_complex_left_branch",
        "wh_questions_subject_gap", "wh_vs_that_with_gap_long_distance",
        "principle_A_case_1", "principle_A_c_command",
        "passive_1", "tough_vs_raising_1",
    ]

    results = {}
    print(f"\n  {'Task':<52}  {'Acc':>6}  {'SAE Acc':>8}  {'Drop':>6}  {'n':>5}")
    print(f"  {'─'*52}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*5}")

    for task_name in tqdm(tasks_to_run, desc="  BLiMP tasks"):
        try:
            ds = load_dataset("nyu-mll/blimp", task_name,
                              split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  ⚠  {task_name}: {e}")
            continue

        correct_orig = 0
        correct_sae  = 0
        n = len(ds)

        for item in ds:
            good = item["sentence_good"]
            bad  = item["sentence_bad"]

            ce_g = get_ce_loss(model, tokenizer, good)
            ce_b = get_ce_loss(model, tokenizer, bad)
            correct_orig += int(ce_g < ce_b)

            ce_gs = get_ce_loss_sae_patched(model, tokenizer, sae, good, layer)
            ce_bs = get_ce_loss_sae_patched(model, tokenizer, sae, bad,  layer)
            correct_sae += int(ce_gs < ce_bs)

        acc_orig = correct_orig / n
        acc_sae  = correct_sae  / n
        drop     = acc_orig - acc_sae
        results[task_name] = {"acc": acc_orig, "sae_acc": acc_sae, "drop": drop, "n": n}

        print(f"  {task_name:<52}  {acc_orig:>6.3f}  {acc_sae:>8.3f}  "
              f"{drop:>+6.3f}  {n:>5}")

    if results:
        mean_acc     = sum(v["acc"]     for v in results.values()) / len(results)
        mean_sae_acc = sum(v["sae_acc"] for v in results.values()) / len(results)
        mean_drop    = sum(v["drop"]    for v in results.values()) / len(results)
        print(f"\n  {'MEAN':<52}  {mean_acc:>6.3f}  {mean_sae_acc:>8.3f}  "
              f"{mean_drop:>+6.3f}")
        print(f"\n  {'✓' if mean_drop < 0.05 else '⚠'} SAE {'preserves' if mean_drop < 0.05 else 'partially degrades'} "
              f"grammatical knowledge at layer {layer:02d} (mean drop={mean_drop:+.3f})")

    # Save results
    os.makedirs("./results", exist_ok=True)
    out_path = f"./results/blimp_eval_layer{layer:02d}.json"
    with open(out_path, "w") as f:
        json.dump({"layer": layer, "tasks": results}, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    return results


# =============================================================================
# TASK 5: ACTIVATION PATCHING
# =============================================================================
def task_patching(model, tokenizer, sae: SAE, layer: int):
    print(f"\n{'═'*62}")
    print(f"  TASK 5: Activation Patching  |  Layer {layer:02d}")
    print(f"{'═'*62}\n")
    experiments = [
        ("SVA simple",     "The dogs run in the park.",  "The dogs runs in the park."),
        ("Anaphor gender", "The boy hurt himself.",       "The boy hurt herself."),
        ("NPI",            "Nobody has ever been here.", "Somebody has ever been here."),
    ]
    for name, clean, corrupt in experiments:
        resid_clean, _ = get_residual_stream(model, tokenizer, clean, layer)
        clean_recon    = sae.decode(sae.encode(resid_clean))
        ce_clean   = get_ce_loss(model, tokenizer, clean)
        ce_corrupt = get_ce_loss(model, tokenizer, corrupt)
        def make_patch_hook(recon):
            def hook_fn(module, inp, out):
                is_tuple = isinstance(out, tuple)
                hidden   = out[0] if is_tuple else out
                patched  = hidden.clone()
                n = min(patched.shape[1], recon.shape[0])
                patched[0, :n] = recon[:n]
                return (patched,) + out[1:] if is_tuple else patched
            return hook_fn
        hook = model.transformer.h[layer].register_forward_hook(make_patch_hook(clean_recon))
        inp  = tokenizer(corrupt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ce_patched = model(**inp, labels=inp["input_ids"]).loss.item()
        hook.remove()
        effect = ce_corrupt - ce_patched
        print(f"  {name}")
        print(f"    CE  clean={ce_clean:.4f}  corrupt={ce_corrupt:.4f}  patched={ce_patched:.4f}")
        print(f"    Effect: {effect:+.4f}  "
              f"{'✓ layer carries this distinction' if effect > 0.01 else '– minimal effect'}\n")


# =============================================================================
# TASK 6: CROSS-LAYER COMPARISON
# =============================================================================
def task_compare(model, tokenizer, layers: list[int]):
    print(f"\n{'═'*62}")
    print(f"  TASK 6: Cross-Layer Comparison  |  Layers {layers}")
    print(f"{'═'*62}\n")
    text = "The dogs run in the park."
    print(f"  Text: '{text}'\n")
    print(f"  {'Layer':>6}  {'L0':>6}  {'MSE':>10}  {'CE Rec':>8}  {'Dead%':>7}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*7}")
    results = []
    for layer in layers:
        sae = load_sae(layer)
        resid, _     = get_residual_stream(model, tokenizer, text, layer)
        feature_acts = sae.encode(resid)
        recon        = sae.decode(feature_acts)
        l0       = (feature_acts > 0).float().sum(-1).mean().item()
        mse      = F.mse_loss(recon, resid).item()
        ce_orig  = get_ce_loss(model, tokenizer, text)
        ce_patch = get_ce_loss_sae_patched(model, tokenizer, sae, text, layer)
        recovery = max(0.0, (1 - (ce_patch - ce_orig) / max(ce_orig, 1e-8)) * 100)
        dead_pct = ((feature_acts.sum(-1) == 0).float().mean() * 100).item()
        print(f"  {layer:>6}  {l0:>6.1f}  {mse:>10.4f}  {recovery:>7.1f}%  {dead_pct:>6.1f}%")
        results.append({"layer": layer, "l0": l0, "mse": mse,
                        "ce_recovery": recovery, "dead_pct": dead_pct})
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["basic","quality","features","blimp",
                                           "patching","compare","all"], default="quality")
    parser.add_argument("--layer",          type=int, default=12)
    parser.add_argument("--layers",         type=int, nargs="+", default=[2,6,8,10,12,16,22])
    parser.add_argument("--all-blimp-tasks",action="store_true",
                        help="Run all 67 BLiMP tasks (slow ~2hr/layer)")
    args = parser.parse_args()

    print(f"\n{'═'*62}")
    print(f"  BabyLM SAE Evaluation  |  Real BLiMP Dataset")
    print(f"  GPT-2 : {GPT2_REPO}")
    print(f"  SAEs  : {SAE_REPO}   Device: {DEVICE}")
    print(f"{'═'*62}")

    model, tokenizer = load_model_and_tokenizer()

    if args.task == "compare":
        task_compare(model, tokenizer, args.layers)
    else:
        sae = load_sae(args.layer)
        if args.task in ("basic",    "all"): task_basic(model, tokenizer, sae, args.layer)
        if args.task in ("quality",  "all"): task_quality(model, tokenizer, sae, args.layer)
        if args.task in ("features", "all"): task_features(model, tokenizer, sae, args.layer)
        if args.task in ("blimp",    "all"):
            task_blimp(model, tokenizer, sae, args.layer, args.all_blimp_tasks)
        if args.task in ("patching", "all"): task_patching(model, tokenizer, sae, args.layer)

    print(f"\n{'═'*62}  Done.\n")


if __name__ == "__main__":
    main()