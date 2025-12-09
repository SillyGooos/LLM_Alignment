"""
PPO (Proximal Policy Optimization) Training Script

This script implements PPO for aligning language models with human preferences.
PPO uses a reward model and value function to optimize the policy via RL.

Supports both sparse rewards (final response only) and dense rewards (token-level).

Usage:
    # Sparse reward PPO
    python train_ppo.py \
      --method PPO \
      --reward_model_path ./models/reward_model/final_model \
      --reward_mode sparse \
      --batch_size 8 \
      --lr 1e-5 \
      --kl_coef 0.05 \
      --epochs 3 \
      --seed 42 \
      --save_dir ./checkpoints/ppo_sparse
    
    # Dense reward PPO
    python train_ppo.py \
      --method PPO \
      --reward_model_path ./models/reward_model/final_model \
      --reward_mode dense \
      --batch_size 8 \
      --lr 1e-5 \
      --kl_coef 0.05 \
      --epochs 3 \
      --seed 42 \
      --save_dir ./checkpoints/ppo_dense

Reference:
    Schulman et al. (2017) - Proximal Policy Optimization Algorithms
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import copy
from pathlib import Path
import sys
PROJECT_ROOT = Path("/kaggle/working/LLM_Alignment")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers and HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    GenerationConfig
,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset as HFDataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

# TRL for PPO
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)

# Import config
from config.default_config import get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_quantization_config(load_in_4bit=True, load_in_8bit=False, mixed_precision='fp16'):
    """
    Create BitsAndBytesConfig for optimal 4-bit quantization
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization
        load_in_8bit: Whether to use 8-bit quantization
        mixed_precision: Mixed precision setting ('fp16' or 'bf16')
    
    Returns:
        BitsAndBytesConfig or None
    """
    if load_in_4bit:
        import torch
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
            bnb_4bit_compute_dtype=torch.float16 if mixed_precision == "fp16" else torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Nested quantization
        )
        logger.info("✓ Created 4-bit quantization config (NF4 + double quantization)")
        return bnb_config
    elif load_in_8bit:
        logger.info("✓ Using 8-bit quantization")
        return None
    else:
        logger.info("✓ No quantization (full precision)")
        return None



class PPOModelTrainer:
    """Main trainer class for PPO alignment"""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None  # Policy model with value head
        self.ref_model = None  # Frozen reference model
        self.reward_model = None  # Frozen reward model
        self.train_dataset = None
        self.val_dataset = None
        
        # Metrics storage
        self.training_history = []
        self.reward_stats = []
        self.kl_stats = []
        
    def setup_paths(self):
        """Setup directory structure"""
        # Create save directory
        if self.args.save_dir:
            self.save_dir = Path(self.args.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ppo_{self.args.reward_mode}_{self.args.seed}_{timestamp}"
            self.save_dir = Path(self.args.output_dir) / run_name
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        self.plots_dir = self.save_dir / "plots"
        self.samples_dir = self.save_dir / "samples"
        
        for d in [self.checkpoint_dir, self.logs_dir, self.plots_dir, self.samples_dir]:
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
        
        # Set padding side
        self.tokenizer.padding_side = "left"
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.save_dir / "tokenizer")
        logger.info(f"Tokenizer saved")
    
    def load_reward_model(self):
        """Load frozen reward model"""
        logger.info(f"Loading reward model from: {self.args.reward_model_path}")
        
        try:
            # Try loading as a full model first
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.reward_model_path,
                num_labels=1,
                load_in_4bit=True,  # FIXED: Use 4-bit instead of 8-bit,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"Failed to load as full model: {e}")
            # Try loading as adapter
            try:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.args.model_name,
                    num_labels=1,
                    load_in_4bit=True,  # FIXED: Use 4-bit instead of 8-bit,
                    device_map="auto",
                    trust_remote_code=self.config.base_model.trust_remote_code,
                )
                self.reward_model = PeftModel.from_pretrained(base_model, self.args.reward_model_path)
            except Exception as e2:
                logger.error(f"Failed to load reward model: {e2}")
                raise
        
        # # CRITICAL FIX: Disable gradient checkpointing for quantized reward model
        # if hasattr(self.reward_model, 'gradient_checkpointing_disable'):
        #     self.reward_model.gradient_checkpointing_disable()
        #     logger.info("✓ Disabled gradient checkpointing for reward model")
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        self.reward_model.eval()
        logger.info("Reward model loaded and frozen")
    
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
        
        # Format for PPO (just need prompts)
        def format_for_ppo(examples):
            formatted = []
            for ex in examples:
                formatted.append({
                    'prompt': ex['prompt'],
                    'response_w': ex['response_w'],  # Keep for reference
                    'response_l': ex['response_l'],  # Keep for reference
                })
            return formatted
        
        train_formatted = format_for_ppo(train_data)
        val_formatted = format_for_ppo(val_data)
        
        # Convert to HuggingFace Dataset
        self.train_dataset = HFDataset.from_list(train_formatted)
        self.val_dataset = HFDataset.from_list(val_formatted)
        
        logger.info(f"Datasets formatted for PPO")
        logger.info(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")
    
    def setup_models(self):
        """Initialize policy model and reference model"""
        logger.info(f"Loading base model: {self.args.model_name}")
        
        # Create quantization config
        bnb_config = create_quantization_config(
            self.args.load_in_4bit,
            self.args.load_in_8bit,
            self.args.mixed_precision
        )
        
        # ✅ Load policy model WITH value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config if bnb_config else None,
            load_in_8bit=self.args.load_in_8bit if not bnb_config else False,
            device_map={"":0},
            trust_remote_code=self.config.base_model.trust_remote_code,
            torch_dtype=torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16,
        )
        
        # Track device
        self.model_device = next(self.model.parameters()).device
        logger.info(f"✓ Policy model on device: {self.model_device}")
        
        # Prepare for training if using quantization
        if self.args.load_in_8bit or self.args.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        
        # ✅ Create frozen reference model (no value head needed)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config if bnb_config else None,
            load_in_8bit=self.args.load_in_8bit if not bnb_config else False,
            device_map={"":0},
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        logger.info("Models loaded and configured")
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute reward for a prompt-response pair"""
        # Format text
        text = self.config.data.prompt_template.format(prompt=prompt)
        text += self.config.data.response_template.format(response=response)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.args.max_length
        ).to(self.reward_model.device)
        
        # Get reward
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            reward = outputs.logits.squeeze().item()
        
        return reward
    
    def compute_dense_rewards(self, prompt: str, response: str, response_tokens: List[int]) -> List[float]:
        """
        Compute token-level rewards for dense reward setup
        
        Strategy: Compute reward at each prefix and use differences as token rewards
        """
        # Decode tokens to get intermediate responses
        dense_rewards = []
        
        # Start with just prompt (baseline)
        prev_reward = 0.0
        
        # Compute reward at each token
        for i in range(1, len(response_tokens) + 1):
            # Decode prefix
            prefix = self.tokenizer.decode(response_tokens[:i], skip_special_tokens=True)
            
            # Compute reward for this prefix
            reward = self.compute_reward(prompt, prefix)
            
            # Token reward is the difference
            token_reward = reward - prev_reward
            dense_rewards.append(token_reward)
            
            prev_reward = reward
        
        return dense_rewards
    
    def generate_rollouts(self, prompts: List[str]) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """Generate responses for prompts"""
        responses = []
        response_tensors = []
        
        for prompt in prompts:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(
                self.config.data.prompt_template.format(prompt=prompt),
                return_tensors='pt'
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    do_sample=self.args.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract generated tokens (remove prompt)
            response_tokens = output[0, input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            responses.append(response_text)
            response_tensors.append(response_tokens)
        
        return responses, response_tensors
    
    def train(self):
        """Main PPO training loop"""
        logger.info("Starting PPO training...")
        
        # Set seed
        set_seed(self.args.seed)
        
        # ✅ CORRECT: Setup PPO config with TRL's actual parameter names
        ppo_config = PPOConfig(
            # Basic training params
            learning_rate=self.args.learning_rate,
            batch_size=self.args.batch_size,
            mini_batch_size=self.args.mini_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            
            # ✅ PPO-specific params (MATCHING TRL!)
            num_ppo_epochs=self.args.num_ppo_epochs,      # ✅ NOT ppo_epochs
            whiten_rewards=self.args.whiten_rewards,       # ✅ NEW parameter
            kl_coef=self.args.kl_coef,                     # ✅ NOT init_kl_coef
            cliprange=self.args.cliprange,                 # ✅ NOT clip_range
            cliprange_value=self.args.cliprange_value,     # ✅ NOT clip_range_vf
            vf_coef=self.args.vf_coef,
            gamma=self.args.gamma,
            lam=self.args.lam,
            # Seed
            seed=self.args.seed,
        )
        
        # ✅ CORRECT: PPOTrainer with correct parameter names
        ppo_trainer = PPOTrainer(
            args=ppo_config,                    # ✅ NOT 'config'
            processing_class=self.tokenizer,    # ✅ NOT 'tokenizer'
            model=self.model,                   # ✅ Complete model (with value head)
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            train_dataset=self.train_dataset,
            value_model = self.model.v_head
            # ❌ NO value_model parameter!
        )
        
        # Training loop
        logger.info("=" * 80)
        logger.info("PPO Training Started")
        logger.info("=" * 80)
        logger.info(f"Training examples: {len(self.train_dataset)}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Mini batch size: {self.args.mini_batch_size}")
        logger.info(f"KL coefficient: {self.args.kl_coef}")
        logger.info(f"Reward mode: {self.args.reward_mode}")
        logger.info(f"Max new tokens: {self.args.max_new_tokens}")
        logger.info("=" * 80)
        
        # Use TRL's built-in training
        ppo_trainer.train()
        
        return ppo_trainer
    
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
            for item in tqdm(instr_data[:1000], desc="Computing perplexity"):
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
                
                # Get loss from the base model (not value head)
                outputs = self.model.pretrained_model(**inputs, labels=inputs['input_ids'])
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
        
        self.model.eval()
        self.ref_model.eval()
        
        # Sample from validation set
        num_samples = min(100, len(self.val_dataset))
        sample_indices = np.random.choice(len(self.val_dataset), num_samples, replace=False)
        
        kl_values = []
        
        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="Computing KL"):
                example = self.val_dataset[int(idx)]
                
                # Format prompt
                prompt_text = self.config.data.prompt_template.format(prompt=example['prompt'])
                
                # Tokenize prompt
                prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model.device)
                
                # Generate response with policy
                policy_output = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=True,
                    temperature=self.args.temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                response_ids = policy_output.sequences[0, prompt_ids.shape[1]:]
                
                # Get log probs from both models
                full_ids = policy_output.sequences
                
                # Policy logits (full_ids already on correct device)
                policy_outputs = self.model.pretrained_model(input_ids=full_ids)
                policy_logits = policy_outputs.logits
                
                # Reference logits (move full_ids to ref_model device)
                ref_full_ids = full_ids.to(self.ref_model.pretrained_model.device)
                ref_outputs = self.ref_model.pretrained_model(input_ids=ref_full_ids)
                ref_logits = ref_outputs.logits.to(policy_logits.device)
                
                # Compute log probs
                policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                
                # Get token log probs
                token_ids = full_ids[:, 1:]
                
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
                
                # Compute KL
                kl_per_token = policy_token_log_probs - ref_token_log_probs
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
        """Plot training curves"""
        logger.info("Generating training curve plots...")
        
        if not self.training_history:
            logger.warning("No training history found")
            return
        
        df = pd.DataFrame(self.training_history)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('PPO Training Curves', fontsize=16)
        
        # Plot 1: Policy loss
        if 'ppo/loss/policy' in df.columns:
            axes[0, 0].plot(df['step'], df['ppo/loss/policy'], linewidth=2)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Policy Loss')
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Value loss
        if 'ppo/loss/value' in df.columns:
            axes[0, 1].plot(df['step'], df['ppo/loss/value'], linewidth=2, color='orange')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Value Loss')
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reward
        if 'ppo/mean_scores' in df.columns:
            axes[0, 2].plot(df['step'], df['ppo/mean_scores'], linewidth=2, color='green')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Mean Reward')
            axes[0, 2].set_title('Mean Reward')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: KL divergence
        if 'objective/kl' in df.columns:
            axes[1, 0].plot(df['step'], df['objective/kl'], linewidth=2, color='red')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Entropy
        if 'objective/entropy' in df.columns:
            axes[1, 1].plot(df['step'], df['objective/entropy'], linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].set_title('Policy Entropy')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Total loss
        if 'ppo/loss/total' in df.columns:
            axes[1, 2].plot(df['step'], df['ppo/loss/total'], linewidth=2, color='brown')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Total Loss')
            axes[1, 2].set_title('Total Loss')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {self.plots_dir / 'training_curves.png'}")
    
    def create_summary(self, perplexity_result: Optional[Dict], kl_stats: Optional[Dict]):
        """Create summary report"""
        logger.info("Creating summary report...")
        
        # Compute final metrics from training history
        final_metrics = {}
        if self.training_history:
            df = pd.DataFrame(self.training_history)
            for col in df.columns:
                if col not in ['epoch', 'batch', 'step']:
                    final_metrics[col] = float(df[col].iloc[-10:].mean())  # Average last 10
        
        summary = {
            'method': 'PPO',
            'reward_mode': self.args.reward_mode,
            'model_name': self.args.model_name,
            'use_lora': self.args.use_lora,
            'lora_r': self.args.lora_r if self.args.use_lora else None,
            'quantization': '8-bit' if self.args.load_in_8bit else ('4-bit' if self.args.load_in_4bit else 'none'),
            'seed': self.args.seed,
            'training': {
                'batch_size': self.args.batch_size,
                'mini_batch_size': self.args.mini_batch_size,
                'learning_rate': self.args.learning_rate,
                'epochs': self.args.epochs,
                'kl_coef': self.args.init_kl_coef,
                'target_kl': self.args.target_kl,
                'clip_range': self.args.clip_range,
                'ppo_epochs': self.args.ppo_epochs,
            },
            'final_metrics': final_metrics,
            'perplexity': perplexity_result,
            'kl_divergence': kl_stats,
        }
        
        # Save as JSON
        with open(self.save_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        report_lines = [
            "# PPO Training Summary",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Save Directory:** `{self.save_dir}`",
            "",
            "## Model Configuration",
            f"- Base Model: `{self.args.model_name}`",
            f"- Reward Mode: **{self.args.reward_mode}**",
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
            f"- Mini Batch Size: {self.args.mini_batch_size}",
            f"- Learning Rate: {self.args.learning_rate}",
            f"- Epochs: {self.args.epochs}",
            f"- KL Coefficient: {self.args.init_kl_coef}",
            f"- Target KL: {self.args.target_kl}",
            f"- Clip Range: {self.args.clip_range}",
            f"- PPO Epochs: {self.args.ppo_epochs}",
            f"- Seed: {self.args.seed}",
            "",
            "## Final Training Metrics",
        ])
        
        for key, value in final_metrics.items():
            report_lines.append(f"- {key}: {value:.4f}")
        
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
            "- Training history: `training_history.csv`",
            "- Training logs: `logs/`",
            "- Plots: `plots/`",
        ])
        
        report_text = "\n".join(report_lines)
        
        with open(self.save_dir / "REPORT.md", 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary saved to {self.save_dir / 'summary.json'}")
        logger.info(f"Report saved to {self.save_dir / 'REPORT.md'}")
    
    def run(self):
        """Run complete PPO training pipeline"""
        logger.info("=" * 80)
        logger.info("Starting PPO Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Save configuration
            self.save_config()
            
            # Setup tokenizer
            self.setup_tokenizer()
            
            # Load reward model
            self.load_reward_model()
            
            # Load datasets
            self.load_datasets()
            
            # Setup models
            self.setup_models()
            
            # Train
            ppo_trainer = self.train()
            
            # Compute perplexity
            perplexity_result = self.compute_perplexity()
            
            # Estimate KL divergence
            kl_stats = self.estimate_kl_divergence()
            
            # Plot training curves
            self.plot_training_curves()
            
            # Create summary
            self.create_summary(perplexity_result, kl_stats)
            
            logger.info("=" * 80)
            logger.info("PPO Training Complete!")
            logger.info("=" * 80)
            logger.info(f"\nResults saved to: {self.save_dir}")
            logger.info(f"Best model: {self.save_dir / 'final_model'}")
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

    parser = argparse.ArgumentParser(description='PPO Training')

    # Basic args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mini_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # PPO specific
    parser.add_argument('--reward_mode', type=str, default='sparse', choices=['sparse', 'dense'])
    parser.add_argument('--num_ppo_epochs', type=int, default=4)          # ✅ NOT ppo_epochs
    parser.add_argument('--whiten_rewards', action='store_true', default=False)  # ✅ NEW
    parser.add_argument('--kl_coef', type=float, default=0.05)            # ✅ NOT init_kl_coef
    parser.add_argument('--cliprange', type=float, default=0.2)           # ✅ NOT clip_range
    parser.add_argument('--cliprange_value', type=float, default=0.2)     # ✅ NOT clip_range_vf
    parser.add_argument('--vf_coef', type=float, default=0.1)             # ✅ Changed default to 0.1
    parser.add_argument('--gamma', type=float, default=1.0)               # ✅ Changed default to 1.0
    parser.add_argument('--lam', type=float, default=0.95)

    # Paths
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--reward_model_path', type=str, required=True)

    # Model
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-135M-Instruct')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)

    # Generation
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--do_sample', action='store_true', default=True)

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

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw_torch')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--save_steps', type=int, default=500)

    args = parser.parse_args()

    # Get config
    config = get_default_config()

    # Create trainer
    trainer = PPOModelTrainer(args, config)

    # Run training
    trainer.run()


if __name__ == "__main__":
    main()