"""
train_sae_layer.py
══════════════════════════════════════════════════════════════════════════════
Trains ONE SAE on ONE layer of GPT-2 Medium (BabyLM).
Designed to be launched once per GPU via launch_all_layers.py.

Place this file at:   /fsxnew/mayank.kumar/SAELens/train_sae_layer.py

Usage (manual single layer):
    CUDA_VISIBLE_DEVICES=0 python train_sae_layer.py --layer 0
    CUDA_VISIBLE_DEVICES=1 python train_sae_layer.py --layer 4
    ...

Use launch_all_layers.py to spawn all 8 GPUs automatically.

══════════════════════════════════════════════════════════════════════════════
HYPERPARAMETER RATIONALE  (grounded in papers)
──────────────────────────────────────────────
Architecture: BatchTopK  [Bussmann et al., NeurIPS 2024; SAELens docs 2025]
  • Relaxes TopK to the batch level → variable L0 per sample, fixed mean L0.
  • Outperforms TopK and JumpReLU on GPT-2 Small and Gemma 2 2B.
  • Advantage over JumpReLU: sparsity is directly specified, no penalty sweep.
  • Saved as JumpReLU for efficient inference.

k = 64  (avg active features/token)
  • Bussmann et al. (2024) benchmark k = 32 on GPT-2 Small (d_in=768).
  • GPT-2 Medium has d_in=1024 (33% larger) → representations are richer.
  • Gao et al. (2024) Figure 1: k ∈ {32,64,128,256,512} all tested; k=64
    gives strong reconstruction at reasonable L0 for medium-sized models.
  • BabyLM data is simpler (child-directed speech) → no need for k>64.
  • k=64 is our pick; k=32 is a valid lower-bound alternative.

d_sae = 1024 × 16 = 16384  (expansion factor 16)
  • Gao et al. (2024): "larger autoencoders are generally better on all
    quality metrics." They scale up to 16M latents on GPT-4.
  • Bricken et al. (2023): 8× for GPT-2 small; we go 16× because:
      - H100s have 80GB VRAM — 16384 features is completely fine.
      - BabyLM representations are low-complexity, so features won't all
        be meaningful at 16×, but those that are will be cleaner.
      - Dead features are handled by aux_loss, so higher expansion is safe.
  • If you see >30% dead features after 5M tokens, switch to 8× (8192).

lr = 2e-4
  • Gao et al. Fig 3: optimal lr sweeps 2^-4 to 2^-7 (1e-4 to 6e-3).
    For d_sae=16384, 2e-4 is near-optimal.
  • Bussmann et al. use lr=2e-4 for GPT-2 Small experiments.

lr_warm_up_steps = 1000
  • SAELens docs: "warm-up is important to avoid dead neurons."
  • ~4% of total steps at 100M tokens / 4096 batch ≈ 24,414 steps.

training_tokens = 100_000_000  (full BabyLM dataset)
  • Use ALL available data. Gao et al. train to compute-MSE frontier.
  • 100M is the full BabyLM-2026-Strict budget — use it all.
  • Gao et al.: GPT-2 small SAEs converge around 1–4B tokens (WebText),
    but BabyLM representations are far simpler → 100M is sufficient.

context_size = 128
  • BabyLM utterances are short conversational turns (avg ~8 tokens).
  • 128 captures essentially all complete utterances.
  • Gao et al. use context_size=64; 128 gives better long-range context.

normalize_activations = "expected_average_only_in"
  • Required: rescales input activations to unit expected L2 norm.
  • Gao et al.: normalization prevents gaming the TopK loss via tiny latents.

Hook layer selection strategy  [Gao et al. 2024, Section 2.1]:
  • "We choose a layer near the end of the network... specifically 5/6 of
    the way into the network for GPT-4 series models, 3/4 for GPT-2 small."
  • For GPT-2 Medium (24 layers): 3/4 = layer 18, 5/6 = layer 20.
  • We train ALL 24 layers (0–23) to give complete model coverage,
    using one H100 per layer (8 GPUs → 8 layers in parallel).
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Verified imports from sae_lens v6.39.0
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner
from sae_lens.config import LoggingConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAEConfig

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "IParraMartin/gpt2-medium-bLM100M"
DATASET_PATH  = "BabyLM-community/BabyLM-2026-Strict"
BASE_SAVE_DIR = "/fsxnew/mayank.kumar/babylm_saes"

D_IN          = 1024      # GPT-2 Medium hidden dim
EXPANSION     = 16        # → d_sae = 16384
D_SAE         = D_IN * EXPANSION
K             = 64        # avg active features per token (BatchTopK)
LR            = 2e-4
TRAINING_TOKENS = 100_000_000   # full BabyLM budget
TRAIN_BATCH   = 4096
WARMUP_STEPS  = 1000


def make_config(layer: int, wandb_run_name: str) -> LanguageModelSAERunnerConfig:
    checkpoint_dir = os.path.join(BASE_SAVE_DIR, f"checkpoints", f"layer_{layer:02d}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    sae = BatchTopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        k=K,
        normalize_activations="expected_average_only_in",
        apply_b_dec_to_input=True,
        aux_loss_coefficient=1.0 / 32,   # Gao et al. 2024: aux scaled by 1/32
        topk_threshold_lr=0.01,
        rescale_acts_by_decoder_norm=True,
        decoder_init_norm=0.1,           # Anthropic April 2024 heuristic init
    )

    total_steps = TRAINING_TOKENS // TRAIN_BATCH
    print(f"  [Layer {layer:02d}] Total steps: {total_steps:,}  |  Warmup: {WARMUP_STEPS}")

    return LanguageModelSAERunnerConfig(
        sae=sae,

        # ── Model ──────────────────────────────────────────────────────────
        model_name=MODEL_NAME,
        model_class_name="AutoModelForCausalLM",
        hook_name=f"transformer.h.{layer}",

        # ── Dataset ────────────────────────────────────────────────────────
        dataset_path=DATASET_PATH,
        is_dataset_tokenized=False,
        streaming=True,
        prepend_bos=True,
        context_size=128,

        # ── Training ───────────────────────────────────────────────────────
        training_tokens=TRAINING_TOKENS,
        train_batch_size_tokens=TRAIN_BATCH,
        store_batch_size_prompts=16,
        n_batches_in_buffer=128,   # large buffer → better shuffle on fast SSDs

        # ── Optimiser ──────────────────────────────────────────────────────
        lr=LR,
        lr_scheduler_name="constant",
        lr_warm_up_steps=WARMUP_STEPS,
        lr_decay_steps=0,

        # ── Hardware ───────────────────────────────────────────────────────
        # CUDA_VISIBLE_DEVICES set by launcher → device="cuda" always hits
        # the one GPU assigned to this process.
        device="cuda",
        act_store_device="cpu",    # activations buffered on CPU to save VRAM
        dtype="float32",

        # ── Reproducibility ────────────────────────────────────────────────
        seed=42,

        # ── Checkpointing ──────────────────────────────────────────────────
        # Saves to /fsxnew/mayank.kumar/babylm_saes/checkpoints/layer_NN/
        checkpoint_path=checkpoint_dir,
        n_checkpoints=5,
        save_final_checkpoint=True,

        # ── W&B Logging ────────────────────────────────────────────────────
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="babylm-saes",
            wandb_entity=None,          # set to your W&B username if needed
            run_name=wandb_run_name,
            wandb_log_frequency=10,
            eval_every_n_wandb_logs=100,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer", type=int, required=True,
        help="Transformer layer index (0-23 for GPT-2 Medium)",
    )
    args = parser.parse_args()
    layer = args.layer

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device visible. Check CUDA_VISIBLE_DEVICES.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*62}")
    print(f"  Layer {layer:02d}  |  {MODEL_NAME}")
    print(f"  GPU  : {gpu_name}  ({vram_gb:.0f} GB)")
    print(f"  d_in={D_IN}  d_sae={D_SAE}  k={K}  tokens={TRAINING_TOKENS:,}")
    print(f"  Save : {BASE_SAVE_DIR}/checkpoints/layer_{layer:02d}/")
    print(f"{'='*62}\n")

    run_name = f"layer{layer:02d}-k{K}-d{D_SAE}-bLM100M"
    cfg      = make_config(layer, run_name)
    trained  = LanguageModelSAETrainingRunner(cfg).run()

    # Save final SAE to dedicated per-layer folder for easy HF upload
    final_dir = os.path.join(BASE_SAVE_DIR, "final", f"layer_{layer:02d}")
    os.makedirs(final_dir, exist_ok=True)
    trained.save_inference_model(final_dir)
    print(f"\n✓  Layer {layer:02d} done.  Saved to: {final_dir}")


if __name__ == "__main__":
    main()