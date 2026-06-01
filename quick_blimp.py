"""
quick_blimp.py
Minimal BLiMP benchmark on layers 8, 12, 16.
Also prints per-token feature counts to diagnose L0.

Run: python3 quick_blimp.py
"""
import json, os
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SAE_CACHE = os.path.expanduser("~/.cache/babylm_saes")
GPT2_REPO = "IParraMartin/gpt2-medium-bLM100M"
SAE_REPO  = "whitepenguin/gpt2-medium-bLM100M-SAE"

BLIMP_PAIRS = [
    ("The dogs run in the park.",         "The dogs runs in the park.",         "SVA_simple"),
    ("The girl who likes cats is happy.", "The girl who likes cats are happy.", "SVA_relative"),
    ("The keys to the cabinet are here.", "The keys to the cabinet is here.",   "SVA_PP"),
    ("The boy hurt himself.",             "The boy hurt herself.",              "anaphor_gender"),
    ("The children helped themselves.",   "The children helped himself.",       "anaphor_number"),
    ("A dog is in the garden.",           "An dog is in the garden.",           "det_noun"),
    ("Nobody has ever been there.",       "Somebody has ever been there.",      "NPI"),
    ("That is the book that I read.",     "That is the book that I read it.",   "filler_gap"),
]

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained(GPT2_REPO)
model     = AutoModelForCausalLM.from_pretrained(GPT2_REPO).to(DEVICE).eval()


def load_sae(layer: int) -> SAE:
    subfolder = f"layer_{layer:02d}"
    local_dir = os.path.join(SAE_CACHE, subfolder)
    snapshot_download(repo_id=SAE_REPO, repo_type="model",
                      allow_patterns=[f"{subfolder}/*"],
                      local_dir=local_dir, local_dir_use_symlinks=False)
    sae = SAE.load_from_disk(os.path.join(local_dir, subfolder), device=DEVICE)
    sae.eval()
    scaler_path = os.path.join(local_dir, subfolder, "activation_scaler.json")
    if os.path.exists(scaler_path):
        sf = json.load(open(scaler_path)).get("scaling_factor")
        if sf is not None:
            sae.fold_activation_norm_scaling_factor(sf)
    return sae


def get_resid(text: str, layer: int) -> torch.Tensor:
    """Returns [seq_len, 1024] — no batch dim."""
    cache = {}
    hook  = model.transformer.h[layer].register_forward_hook(
        lambda m, i, o: cache.update({"r": o[0].detach()})
    )
    inp = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**inp)
    hook.remove()
    # o[0] has shape [1, seq_len, 1024] → squeeze batch
    resid = cache["r"]
    if resid.dim() == 3:
        resid = resid.squeeze(0)   # → [seq_len, 1024]
    return resid


def ce_loss(text: str) -> float:
    inp = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inp, labels=inp["input_ids"]).loss.item()


def ce_loss_sae_patched(text: str, sae: SAE, layer: int) -> float:
    """
    Patch SAE reconstruction into GPT-2 at given layer.
    GPT-2 block output is a tuple: (hidden_state_tensor, *other).
    We replace the hidden_state with the SAE reconstruction.
    """
    def patch_hook(module, inp, out):
        # out is a tuple; first element is the hidden state [batch, seq, 1024]
        hidden  = out[0]
        recon   = sae.decode(sae.encode(hidden))
        # Rebuild tuple with patched hidden state
        return (recon,) + out[1:]

    hook = model.transformer.h[layer].register_forward_hook(patch_hook)
    inp  = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**inp, labels=inp["input_ids"]).loss.item()
    hook.remove()
    return loss


# ── Diagnose per-token activations ───────────────────────────────────────────
print("\n=== Per-token feature activation diagnosis (layer 16) ===")
sae16 = load_sae(16)
text  = "The child looked at the dog and smiled."
inp   = tokenizer(text, return_tensors="pt").to(DEVICE)
tokens = tokenizer.convert_ids_to_tokens(inp["input_ids"][0])

print(f"Input shape  : {inp['input_ids'].shape}")
print(f"Tokens       : {tokens}")

# Get resid WITH batch dim to check what encode sees
cache = {}
hook  = model.transformer.h[16].register_forward_hook(
    lambda m, i, o: cache.update({"r": o[0].detach()})
)
with torch.no_grad():
    model(**inp)
hook.remove()

resid_raw = cache["r"]   # [1, 9, 1024] (with batch)
print(f"resid_raw shape: {resid_raw.shape}")

# Encode with batch dim
fa_batch = sae16.encode(resid_raw)         # [1, 9, 16384]
print(f"feature_acts with batch dim: {fa_batch.shape}")
l0_batch = (fa_batch > 0).float().sum(-1).mean().item()
print(f"Mean L0 (with batch dim): {l0_batch:.1f}")

# Encode without batch dim
resid_sq = resid_raw.squeeze(0)            # [9, 1024]
fa_sq    = sae16.encode(resid_sq)          # [9, 16384]
print(f"feature_acts without batch dim: {fa_sq.shape}")
l0_sq = (fa_sq > 0).float().sum(-1).mean().item()
print(f"Mean L0 (without batch dim): {l0_sq:.1f}")

print(f"\nPer-token active counts (without batch dim):")
for i, tok in enumerate(tokens):
    n = int((fa_sq[i] > 0).sum().item())
    print(f"  {tok:<22}  active={n}")

print(f"\nPer-token active counts (with batch dim, indexing [0, i]):")
for i, tok in enumerate(tokens):
    n = int((fa_batch[0, i] > 0).sum().item())
    print(f"  {tok:<22}  active={n}")


# ── Original model BLiMP ─────────────────────────────────────────────────────
print("\n=== Original Model BLiMP ===")
print(f"{'Phenomenon':<20}  {'Gram CE':>8}  {'Ungram CE':>9}  {'':>4}")
correct_orig = 0
for gram, ungram, name in BLIMP_PAIRS:
    ce_g = ce_loss(gram)
    ce_u = ce_loss(ungram)
    ok   = ce_g < ce_u
    correct_orig += int(ok)
    print(f"  {name:<20}  {ce_g:>8.4f}  {ce_u:>9.4f}  {'✓' if ok else '✗'}")
print(f"\nOriginal BLiMP: {correct_orig}/{len(BLIMP_PAIRS)} = "
      f"{correct_orig/len(BLIMP_PAIRS)*100:.1f}%")


# ── SAE-patched BLiMP for layers 8, 12, 16 ───────────────────────────────────
for layer in [8, 12, 16]:
    print(f"\n=== Layer {layer:02d} SAE-Patched BLiMP ===")
    sae = load_sae(layer) if layer != 16 else sae16

    # L0 and MSE on test sentences
    l0s, mses = [], []
    for t in ["The dogs run.", "She smiled.", "Nobody left."]:
        r  = get_resid(t, layer)                   # [seq, 1024]
        fa = sae.encode(r.unsqueeze(0))             # [1, seq, 16384]
        re = sae.decode(fa)
        l0s.append((fa > 0).float().sum(-1).mean().item())
        mses.append(F.mse_loss(re, r.unsqueeze(0)).item())

    print(f"  Mean L0: {sum(l0s)/len(l0s):.1f}   MSE: {sum(mses)/len(mses):.4f}")
    print(f"  {'Phenomenon':<20}  {'Gram CE':>8}  {'Ungram CE':>9}  {'Orig':>5}  "
          f"{'SAE g':>8}  {'SAE u':>8}  {'SAE':>5}")

    sae_correct = 0
    for gram, ungram, name in BLIMP_PAIRS:
        ce_g  = ce_loss(gram)
        ce_u  = ce_loss(ungram)
        ce_gs = ce_loss_sae_patched(gram,   sae, layer)
        ce_us = ce_loss_sae_patched(ungram, sae, layer)
        ok_o  = ce_g  < ce_u
        ok_s  = ce_gs < ce_us
        sae_correct += int(ok_s)
        print(f"  {name:<20}  {ce_g:>8.4f}  {ce_u:>9.4f}  "
              f"{'✓' if ok_o else '✗':>5}  "
              f"{ce_gs:>8.4f}  {ce_us:>8.4f}  {'✓' if ok_s else '✗':>5}")

    print(f"\n  SAE BLiMP: {sae_correct}/{len(BLIMP_PAIRS)} = "
          f"{sae_correct/len(BLIMP_PAIRS)*100:.1f}%  "
          f"(drop: {correct_orig/len(BLIMP_PAIRS)*100 - sae_correct/len(BLIMP_PAIRS)*100:.1f}pp)")

print("\nDone.")