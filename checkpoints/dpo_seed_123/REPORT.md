# DPO Training Summary

**Date:** 2025-12-09 04:28:25
**Save Directory:** `/kaggle/working/LLM_Alignment/checkpoints/dpo_seed_123`

## Model Configuration
- Base Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- LoRA: Enabled
  - Rank: 8
  - Alpha: 16
- Quantization: 4-bit

## Training Configuration
- Batch Size: 16
- Gradient Accumulation: 4
- Effective Batch Size: 64
- Learning Rate: 5e-05
- Epochs: 1
- Beta (Temperature): 0.1
- Loss Type: sigmoid
- Optimizer: adamw_torch
- Seed: 123

## Evaluation Metrics
- eval_loss: 0.2990
- eval_runtime: 155.3046
- eval_samples_per_second: 7.8940
- eval_steps_per_second: 0.4960
- eval_rewards/chosen: -1.1511
- eval_rewards/rejected: -2.7244
- eval_rewards/accuracies: 0.9273
- eval_rewards/margins: 1.5733
- eval_logps/chosen: -386.3573
- eval_logps/rejected: -483.4769
- eval_logits/chosen: 6.1672
- eval_logits/rejected: 5.8388
- epoch: 1.0000

## Perplexity (Alignment Tax)
- Perplexity: 16.7499
- Average Loss: 2.8184

## KL Divergence (vs Reference)
- Mean: -0.3074 ± 0.0838
- Median: -0.2981
- Range: [-0.5393, -0.0819]

## Files
- Final model: `final_model/`
- Tokenizer: `tokenizer/`
- Training logs: `logs/`
- Plots: `plots/`
- Evaluation metrics: `eval_metrics.json`
- Perplexity: `perplexity.json`
- KL divergence: `kl_divergence.json`