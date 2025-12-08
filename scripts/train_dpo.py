"""
DPO (Direct Preference Optimization) Training Script

This script implements DPO for aligning language models with human preferences.
DPO directly optimizes the policy on preference data without requiring a separate reward model.

Usage:
    python train_dpo.py \
      --method DPO \
      --batch_size 8 \
      --lr 1e-4 \
      --epochs 3 \
      --lora_r 8 \
      --beta 0.1 \
      --seed 42 \
      --save_dir ./checkpoints/dpo

Reference:
    Rafailov et al. (2023) - Direct Preference Optimization
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import copy
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers and HuggingFace
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset, Dataset as HFDataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

# TRL for DPO
from trl import DPOTrainer, DPOConfig

# Import config
from config.default_config import get_default_config
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DPOModelTrainer:
    """Main trainer class for DPO alignment"""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Metrics storage
        self.training_history = []
        
    def setup_paths(self):
        """Setup directory structure"""
        # Create save directory
        if self.args.save_dir:
            self.save_dir = Path(self.args.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"dpo_{self.args.seed}_{timestamp}"
            self.save_dir = Path(self.args.output_dir) / run_name
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        self.plots_dir = self.save_dir / "plots"
        
        for d in [self.checkpoint_dir, self.logs_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Save directory: {self.save_dir}")
    
    def save_config(self):
        """Save arguments and config"""
        # Save args
        args_path = self.save_dir / "args.json"
        with open(args_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        logger.info(f"Saved args to {args_path}")

        # Save config
        config_path = self.save_dir / "config.json"
        config_dict = {
            'data': vars(self.config.data) if hasattr(self.config.data, '__dict__') else {},
            'base_model': vars(self.config.base_model) if hasattr(self.config.base_model, '__dict__') else {},
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Saved config to {config_path}")
    
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        logger.info(f"Loading tokenizer: {self.args.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Set padding side for training
        self.tokenizer.padding_side = "left"  # Important for DPO
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.save_dir / "tokenizer")
        logger.info(f"Tokenizer saved")
    
    def load_datasets(self):
        """Load and prepare datasets"""
        logger.info("Loading datasets...")
        
        data_dir = Path(self.args.data_dir)
        
        # Load processed data
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        
        # Check if files exist
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Please run prepare_data.py first."
            )
        
        # Load data
        train_data = []
        with open(train_path, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
        
        val_data = []
        with open(val_path, 'r') as f:
            for line in f:
                val_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val examples")
        
        # Format for DPO
        # DPO expects: prompt, chosen, rejected
        def format_for_dpo(examples):
            formatted = []
            for ex in examples:
                formatted.append({
                    'prompt': ex['prompt'],
                    'chosen': ex['response_w'],
                    'rejected': ex['response_l'],
                })
            return formatted
        
        train_formatted = format_for_dpo(train_data)
        val_formatted = format_for_dpo(val_data)
        
        # Convert to HuggingFace Dataset
        self.train_dataset = HFDataset.from_list(train_formatted)
        self.val_dataset = HFDataset.from_list(val_formatted)
        
        logger.info(f"Datasets formatted for DPO")
        logger.info(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")
    
    def setup_models(self):
        """Initialize policy model and reference model"""
        logger.info(f"Loading base model: {self.args.model_name}")
        
        # Load policy model (trainable)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            load_in_8bit=self.args.load_in_8bit,
            load_in_4bit=self.args.load_in_4bit,
            device_map="auto",
            trust_remote_code=self.config.base_model.trust_remote_code,
            torch_dtype=torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16 if self.args.mixed_precision == "bf16" else "auto",
        )
        
        # Prepare for training if using quantization
        if self.args.load_in_8bit or self.args.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("Prepared model for quantized training")
        
        # Apply LoRA if specified
        if self.args.use_lora:
            logger.info(f"Applying LoRA with rank={self.args.lora_r}")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=self.config.lora.target_modules,
                bias=self.config.lora.bias,
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Setup reference model (frozen)
        if self.args.reference_model:
            # Load from specified path
            logger.info(f"Loading reference model from: {self.args.reference_model}")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.args.reference_model,
                load_in_8bit=self.args.load_in_8bit,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            
            # CRITICAL FIX: Disable gradient checkpointing for quantized reference model
            if self.args.load_in_8bit or self.args.load_in_4bit:
                if hasattr(self.ref_model, 'gradient_checkpointing_disable'):
                    self.ref_model.gradient_checkpointing_disable()
                    logger.info("✓ Disabled gradient checkpointing for reference model")
        else:
            # Create frozen copy of base model
            logger.info("Creating frozen reference model from base model")
            # DPOTrainer will handle this automatically if ref_model=None
            self.ref_model = None
        
        logger.info("Models loaded and configured")
    
    def train(self):
        """Main training loop using DPOTrainer"""
        logger.info("Starting DPO training...")
        
        # Set seed
        set_seed(self.args.seed)
        
        # Setup DPO training arguments
        training_args = DPOConfig(
            # Output and logging
            output_dir=str(self.checkpoint_dir),
            logging_dir=str(self.logs_dir),
            logging_steps=self.args.logging_steps,
            report_to=["tensorboard"],
            
            # Training hyperparameters
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            max_grad_norm=self.args.max_grad_norm,
            
            # DPO-specific parameters
            beta=self.args.beta,  # Temperature parameter
            loss_type=self.args.loss_type,  # sigmoid, hinge, ipo
            label_smoothing=self.args.label_smoothing,
            
            # Optimizer
            optim=self.args.optimizer,
            lr_scheduler_type=self.args.lr_scheduler_type,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=self.args.eval_steps,
            save_strategy="steps",
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Hardware
            fp16=(self.args.mixed_precision == "fp16"),
            bf16=(self.args.mixed_precision == "bf16"),
            dataloader_num_workers=self.args.num_workers,
            gradient_checkpointing=self.config.hardware.gradient_checkpointing,
            
            # Reproducibility
            seed=self.args.seed,
            data_seed=self.args.seed,
            
            # Generation (for logging samples during training)
            max_length=self.args.max_length,
            max_prompt_length=self.args.max_length // 2,
        )
        
        # Create DPO trainer - FIXED: use processing_class instead of tokenizer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,  # Changed from tokenizer
        )
        
        # Train
        logger.info("=" * 80)
        logger.info("DPO Training Started")
        logger.info("=" * 80)
        logger.info(f"Training examples: {len(self.train_dataset)}")
        logger.info(f"Validation examples: {len(self.val_dataset)}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Gradient accumulation: {self.args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Beta: {self.args.beta}")
        logger.info(f"Epochs: {self.args.epochs}")
        logger.info("=" * 80)
        
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        final_model_path = self.save_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # FIXED: For LoRA + quantization, save adapter only
        if self.args.use_lora:
            # Save only LoRA adapter weights
            self.model.save_pretrained(str(final_model_path))
            logger.info(f"✓ Saved LoRA adapter to {final_model_path}")
        else:
            # For full models, use trainer.save_model
            trainer.save_model(str(final_model_path))
            logger.info(f"✓ Saved full model to {final_model_path}")
        
        # Always save tokenizer
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save training state
        trainer.save_state()
        
        logger.info("=" * 80)
        logger.info("DPO Training Completed")
        logger.info("=" * 80)
        
        return trainer, train_result
    
    def evaluate(self, trainer):
        """Evaluate on validation set"""
        logger.info("Evaluating model...")
        
        # Evaluate on validation set
        eval_metrics = trainer.evaluate()
        
        logger.info(f"\nValidation metrics:")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save evaluation metrics
        with open(self.save_dir / "eval_metrics.json", 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        return eval_metrics
    
    def compute_perplexity(self):
        """Compute perplexity on instruction-following data"""
        logger.info("Computing perplexity on instruction-following data...")
        
        # Load instruction-following subset
        instr_path = Path(self.args.data_dir) / "instr_following_subset.jsonl"
        
        if not instr_path.exists():
            logger.warning(f"Instruction-following subset not found at {instr_path}")
            return None
        
        # Load data
        instr_data = []
        with open(instr_path, 'r') as f:
            for line in f:
                instr_data.append(json.loads(line))
        
        logger.info(f"Computing perplexity on {len(instr_data)} examples")
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for item in tqdm(instr_data[:1000], desc="Computing perplexity"):  # Limit to 1000 for speed
                # Format text
                text = self.config.data.prompt_template.format(prompt=item['prompt'])
                text += self.config.data.response_template.format(response=item['response'])
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.args.max_length
                ).to(self.model.device)
                
                # Get loss
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Accumulate
                num_tokens = inputs['input_ids'].shape[1]
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        # Save
        perplexity_result = {
            'perplexity': float(perplexity),
            'avg_loss': float(avg_loss),
            'num_examples': len(instr_data[:1000]),
            'total_tokens': int(total_tokens)
        }
        
        with open(self.save_dir / "perplexity.json", 'w') as f:
            json.dump(perplexity_result, f, indent=2)
        
        return perplexity_result
    
    def estimate_kl_divergence(self):
        """Estimate KL divergence between trained policy and reference"""
        logger.info("Estimating KL divergence vs reference model...")
        
        # Load reference model if not already loaded
        if self.ref_model is None:
            logger.info("Loading reference model for KL computation...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            logger.info("✓ Reference model loaded for KL computation")
        
        self.model.eval()
        self.ref_model.eval()
        
        # Sample from validation set
        num_samples = min(100, len(self.val_dataset))
        sample_indices = np.random.choice(len(self.val_dataset), num_samples, replace=False)
        
        kl_values = []
        
        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="Computing KL"):
                example = self.val_dataset[int(idx)]
                
                # Format prompt + chosen response
                text = example['prompt'] + " " + example['chosen']
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.args.max_length
                )
                
                # Get logits from both models (move inputs to each model's device)
                policy_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                policy_outputs = self.model(**policy_inputs)
                
                ref_inputs = {k: v.to(self.ref_model.device) for k, v in inputs.items()}
                ref_outputs = self.ref_model(**ref_inputs)
                
                policy_logits = policy_outputs.logits
                ref_logits = ref_outputs.logits
                
                # Compute log probabilities
                policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                
                # Move ref_logits to same device as policy_logits for comparison
                ref_log_probs = ref_log_probs.to(policy_log_probs.device)
                
                # Get actual token log probs
                token_ids = policy_inputs['input_ids'][:, 1:]  # Shift for next-token prediction
                
                # Gather log probs for actual tokens
                policy_token_log_probs = torch.gather(
                    policy_log_probs[:, :-1, :], 
                    dim=2, 
                    index=token_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                ref_token_log_probs = torch.gather(
                    ref_log_probs[:, :-1, :],
                    dim=2,
                    index=token_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                # Compute KL per token
                kl_per_token = policy_token_log_probs - ref_token_log_probs
                
                # Average over sequence
                kl_mean = kl_per_token.mean().item()
                kl_values.append(kl_mean)
        
        # Compute statistics
        kl_stats = {
            'kl_mean': float(np.mean(kl_values)),
            'kl_std': float(np.std(kl_values)),
            'kl_min': float(np.min(kl_values)),
            'kl_max': float(np.max(kl_values)),
            'kl_median': float(np.median(kl_values)),
            'num_samples': num_samples
        }
        
        logger.info(f"KL Divergence: {kl_stats['kl_mean']:.4f} ± {kl_stats['kl_std']:.4f}")
        
        # Save
        with open(self.save_dir / "kl_divergence.json", 'w') as f:
            json.dump(kl_stats, f, indent=2)
        
        return kl_stats
    
    def plot_training_curves(self):
        """Plot training curves from logs"""
        logger.info("Generating training curve plots...")
        
        # Load training state
        state_path = self.checkpoint_dir / "trainer_state.json"
        
        if not state_path.exists():
            logger.warning("Training state not found, skipping plots")
            return
        
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        log_history = state.get('log_history', [])
        
        if not log_history:
            logger.warning("No log history found")
            return
        
        # Extract metrics
        train_loss = []
        eval_loss = []
        steps = []
        eval_steps = []
        
        for entry in log_history:
            if 'loss' in entry:
                train_loss.append(entry['loss'])
                steps.append(entry.get('step', len(steps)))
            if 'eval_loss' in entry:
                eval_loss.append(entry['eval_loss'])
                eval_steps.append(entry.get('step', len(eval_steps)))
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training loss
        if train_loss:
            axes[0].plot(steps, train_loss, label='Train Loss', linewidth=2)
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Validation loss
        if eval_loss:
            axes[1].plot(eval_steps, eval_loss, label='Validation Loss', linewidth=2, color='orange')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Validation Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {self.plots_dir / 'training_curves.png'}")
    
    def create_summary(self, eval_metrics: Dict, perplexity_result: Optional[Dict], kl_stats: Optional[Dict]):
        """Create summary report"""
        logger.info("Creating summary report...")
        
        summary = {
            'method': 'DPO',
            'model_name': self.args.model_name,
            'use_lora': self.args.use_lora,
            'lora_r': self.args.lora_r if self.args.use_lora else None,
            'quantization': '8-bit' if self.args.load_in_8bit else ('4-bit' if self.args.load_in_4bit else 'none'),
            'seed': self.args.seed,
            'training': {
                'batch_size': self.args.batch_size,
                'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
                'effective_batch_size': self.args.batch_size * self.args.gradient_accumulation_steps,
                'learning_rate': self.args.learning_rate,
                'epochs': self.args.epochs,
                'optimizer': self.args.optimizer,
                'beta': self.args.beta,
                'loss_type': self.args.loss_type,
            },
            'eval_metrics': eval_metrics,
            'perplexity': perplexity_result,
            'kl_divergence': kl_stats,
        }
        
        # Save as JSON
        with open(self.save_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        report_lines = [
            "# DPO Training Summary",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Save Directory:** `{self.save_dir}`",
            "",
            "## Model Configuration",
            f"- Base Model: `{self.args.model_name}`",
            f"- LoRA: {'Enabled' if self.args.use_lora else 'Disabled'}",
        ]
        
        if self.args.use_lora:
            report_lines.append(f"  - Rank: {self.args.lora_r}")
            report_lines.append(f"  - Alpha: {self.args.lora_alpha}")
        
        report_lines.extend([
            f"- Quantization: {summary['quantization']}",
            "",
            "## Training Configuration",
            f"- Batch Size: {self.args.batch_size}",
            f"- Gradient Accumulation: {self.args.gradient_accumulation_steps}",
            f"- Effective Batch Size: {summary['training']['effective_batch_size']}",
            f"- Learning Rate: {self.args.learning_rate}",
            f"- Epochs: {self.args.epochs}",
            f"- Beta (Temperature): {self.args.beta}",
            f"- Loss Type: {self.args.loss_type}",
            f"- Optimizer: {self.args.optimizer}",
            f"- Seed: {self.args.seed}",
            "",
            "## Evaluation Metrics",
        ])
        
        for key, value in eval_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"- {key}: {value:.4f}")
            else:
                report_lines.append(f"- {key}: {value}")
        
        if perplexity_result:
            report_lines.extend([
                "",
                "## Perplexity (Alignment Tax)",
                f"- Perplexity: {perplexity_result['perplexity']:.4f}",
                f"- Average Loss: {perplexity_result['avg_loss']:.4f}",
            ])
        
        if kl_stats:
            report_lines.extend([
                "",
                "## KL Divergence (vs Reference)",
                f"- Mean: {kl_stats['kl_mean']:.4f} ± {kl_stats['kl_std']:.4f}",
                f"- Median: {kl_stats['kl_median']:.4f}",
                f"- Range: [{kl_stats['kl_min']:.4f}, {kl_stats['kl_max']:.4f}]",
            ])
        
        report_lines.extend([
            "",
            "## Files",
            "- Final model: `final_model/`",
            "- Tokenizer: `tokenizer/`",
            "- Training logs: `logs/`",
            "- Plots: `plots/`",
            "- Evaluation metrics: `eval_metrics.json`",
        ])
        
        if perplexity_result:
            report_lines.append("- Perplexity: `perplexity.json`")
        if kl_stats:
            report_lines.append("- KL divergence: `kl_divergence.json`")
        
        report_text = "\n".join(report_lines)
        
        with open(self.save_dir / "REPORT.md", 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary saved to {self.save_dir / 'summary.json'}")
        logger.info(f"Report saved to {self.save_dir / 'REPORT.md'}")
    
    def run(self):
        """Run complete DPO training pipeline"""
        logger.info("=" * 80)
        logger.info("Starting DPO Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Save configuration
            self.save_config()
            
            # Setup tokenizer
            self.setup_tokenizer()
            
            # Load datasets
            self.load_datasets()
            
            # Setup models
            self.setup_models()
            
            # Train
            trainer, train_result = self.train()
            
            # Evaluate
            eval_metrics = self.evaluate(trainer)
            
            # Compute perplexity
            perplexity_result = self.compute_perplexity()
            
            # Estimate KL divergence
            kl_stats = self.estimate_kl_divergence()
            
            # Plot training curves
            self.plot_training_curves()
            
            # Create summary
            self.create_summary(eval_metrics, perplexity_result, kl_stats)
            
            logger.info("=" * 80)
            logger.info("DPO Training Complete!")
            logger.info("=" * 80)
            logger.info(f"\nResults saved to: {self.save_dir}")
            logger.info(f"Best model: {self.save_dir / 'final_model'}")
            logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 0):.4f}")
            if perplexity_result:
                logger.info(f"Perplexity: {perplexity_result['perplexity']:.4f}")
            if kl_stats:
                logger.info(f"KL Divergence: {kl_stats['kl_mean']:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='DPO Training')

    # Basic args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # DPO specific
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default='sigmoid', choices=['sigmoid', 'hinge', 'ipo'])
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--reference_model', type=str, default=None)

    # Paths
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/processed')

    # Model
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-135M-Instruct')
    parser.add_argument('--max_length', type=int, default=512)

    # Quantization
    parser.add_argument('--load_in_8bit', action='store_true', default=False)
    parser.add_argument('--load_in_4bit', action='store_true', default=True)
    parser.add_argument('--mixed_precision', type=str, default='fp16')

    # LoRA
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Logging
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw_torch')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')

    args = parser.parse_args()

    # Get config
    config = get_default_config()

    # Create trainer
    trainer = DPOModelTrainer(args, config)

    # Run training
    trainer.run()


if __name__ == "__main__":
    main()