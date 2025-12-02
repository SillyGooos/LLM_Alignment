"""
GRPO (Group Relative Policy Optimization) Training Script

This script implements GRPO for aligning language models with human preferences.
GRPO removes the need for a separate value function by normalizing rewards
within groups of sampled responses.

Usage:
    python train_grpo.py \
      --method GRPO \
      --reward_model_path ./models/reward_model/final_model \
      --group_size 8 \
      --advantage_normalization rank \
      --batch_size 4 \
      --lr 1e-5 \
      --epochs 3 \
      --seed 42 \
      --save_dir ./checkpoints/grpo

Reference:
    Shao et al. (2024) - DeepSeekMath: Pushing the Limits of Mathematical 
    Reasoning in Open Language Models
"""

import os
import sys
# Add project root to path for Kaggle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Transformers and HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset, Dataset as HFDataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

# Import config
from config.default_config import (
    parse_grpo_args,
    get_config_from_args,
    save_args_to_json,
    Config
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GRPOModelTrainer:
    """Main trainer class for GRPO alignment"""
    
    def __init__(self, args, config: Config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None  # Policy model (trainable)
        self.ref_model = None  # Frozen reference model
        self.reward_model = None  # Frozen reward model
        self.train_dataset = None
        self.val_dataset = None
        
        # Metrics storage
        self.training_history = []
        self.advantage_stats = []
        
    def setup_paths(self):
        """Setup directory structure"""
        # Create save directory
        if self.args.save_dir:
            self.save_dir = Path(self.args.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"grpo_g{self.args.group_size}_{self.args.seed}_{timestamp}"
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
        save_args_to_json(self.args, str(args_path))
        logger.info(f"Saved args to {args_path}")
        
        # Save config
        config_path = self.save_dir / "config.json"
        self.config.save(str(config_path))
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
                load_in_8bit=True,
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
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=self.config.base_model.trust_remote_code,
                )
                self.reward_model = PeftModel.from_pretrained(base_model, self.args.reward_model_path)
            except Exception as e2:
                logger.error(f"Failed to load reward model: {e2}")
                raise
        
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
        
        # Format for GRPO (just need prompts)
        def format_for_grpo(examples):
            formatted = []
            for ex in examples:
                formatted.append({
                    'prompt': ex['prompt'],
                })
            return formatted
        
        train_formatted = format_for_grpo(train_data)
        val_formatted = format_for_grpo(val_data)
        
        # Convert to HuggingFace Dataset
        self.train_dataset = HFDataset.from_list(train_formatted)
        self.val_dataset = HFDataset.from_list(val_formatted)
        
        logger.info(f"Datasets formatted for GRPO")
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
        
        # Create frozen reference model (copy before LoRA)
        logger.info("Creating frozen reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            load_in_8bit=True,
            device_map="auto",
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
    
    def generate_group(self, prompt: str, group_size: int) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate a group of candidate responses for a single prompt"""
        responses = []
        response_tensors = []
        
        # Tokenize prompt
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model.device)
        
        for _ in range(group_size):
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract generated tokens (remove prompt)
            response_tokens = output[0, input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            responses.append(response_text)
            response_tensors.append(response_tokens)
        
        return responses, response_tensors
    
    def compute_group_advantages(
        self, 
        rewards: List[float], 
        normalization: str = 'rank'
    ) -> np.ndarray:
        """
        Compute normalized advantages within a group
        
        Args:
            rewards: List of rewards for the group
            normalization: Method - 'rank', 'zscore', or 'minmax'
        
        Returns:
            Array of advantages
        """
        rewards = np.array(rewards)
        
        if normalization == 'rank':
            # Rank-based normalization
            # Higher rank = higher advantage
            ranks = stats.rankdata(rewards, method='average')
            advantages = (ranks - np.mean(ranks)) / (np.std(ranks) + 1e-8)
            
        elif normalization == 'zscore':
            # Z-score normalization
            if self.args.use_baseline:
                advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            else:
                advantages = rewards / (np.std(rewards) + 1e-8)
                
        elif normalization == 'minmax':
            # Min-max normalization to [-1, 1]
            r_min = np.min(rewards)
            r_max = np.max(rewards)
            if r_max - r_min > 1e-8:
                advantages = 2 * (rewards - r_min) / (r_max - r_min) - 1
            else:
                advantages = np.zeros_like(rewards)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        return advantages
    
    def compute_log_probs(
        self, 
        model: nn.Module,
        prompt_text: str,
        response_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for a response"""
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(model.device)
        
        # Concatenate prompt and response
        full_ids = torch.cat([prompt_ids, response_tokens.unsqueeze(0).to(model.device)], dim=1)
        
        # Get logits
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=full_ids)
            logits = outputs.logits
        
        # Compute log probs for response tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probs of actual response tokens
        response_log_probs = []
        for i, token_id in enumerate(response_tokens):
            # Position in full sequence
            pos = prompt_ids.shape[1] + i
            token_log_prob = log_probs[0, pos - 1, token_id]
            response_log_probs.append(token_log_prob)
        
        return torch.stack(response_log_probs)
    
    def grpo_loss(
        self,
        prompt: str,
        response_tokens: torch.Tensor,
        advantage: float
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a single response
        
        GRPO objective: maximize advantage-weighted log probability
        Loss = -advantage * log π(response | prompt)
        """
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        
        # Get policy log probs
        policy_log_probs = self.compute_log_probs(self.model, prompt_text, response_tokens)
        
        # Get reference log probs for KL
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(self.ref_model, prompt_text, response_tokens)
        
        # Policy loss: -advantage * mean(log_probs)
        policy_loss = -advantage * policy_log_probs.mean()
        
        # KL divergence (optional regularization - GRPO doesn't use explicit KL by default)
        # But we can add it for comparison with PPO
        kl_div = (policy_log_probs - ref_log_probs).mean()
        
        # Total loss
        loss = policy_loss
        
        return loss, policy_loss.item(), kl_div.item()
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """Single GRPO training step on a batch"""
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_rewards = []
        total_advantages = []
        
        num_groups = 0
        
        for prompt in batch_prompts:
            # Generate group of responses
            responses, response_tensors = self.generate_group(prompt, self.args.group_size)
            
            # Compute rewards for all responses in group
            rewards = [self.compute_reward(prompt, response) for response in responses]
            
            # Compute group-relative advantages
            advantages = self.compute_group_advantages(rewards, self.args.advantage_normalization)
            
            # Update policy for each response in group
            for response_tokens, advantage in zip(response_tensors, advantages):
                loss, policy_loss, kl_div = self.grpo_loss(prompt, response_tokens, advantage)
                
                # Backward pass
                loss.backward()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss
                total_kl += kl_div
            
            total_rewards.extend(rewards)
            total_advantages.extend(advantages.tolist())
            num_groups += 1
        
        # Gradient step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Return metrics
        metrics = {
            'loss': total_loss / (num_groups * self.args.group_size),
            'policy_loss': total_policy_loss / (num_groups * self.args.group_size),
            'kl_div': total_kl / (num_groups * self.args.group_size),
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_advantage': np.mean(total_advantages),
            'std_advantage': np.std(total_advantages),
            'num_groups': num_groups,
        }
        
        return metrics
    
    def train(self):
        """Main GRPO training loop"""
        logger.info("Starting GRPO training...")
        
        # Set seed
        set_seed(self.args.seed)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Setup learning rate scheduler
        total_steps = (len(self.train_dataset) // self.args.batch_size) * self.args.epochs
        
        if self.args.lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif self.args.lr_scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
            )
        else:
            scheduler = None
        
        # Training loop
        logger.info("=" * 80)
        logger.info("GRPO Training Started")
        logger.info("=" * 80)
        logger.info(f"Training examples: {len(self.train_dataset)}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Group size: {self.args.group_size}")
        logger.info(f"Advantage normalization: {self.args.advantage_normalization}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Epochs: {self.args.epochs}")
        logger.info("=" * 80)
        
        global_step = 0
        
        for epoch in range(self.args.epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.args.epochs}")
            logger.info(f"{'='*80}")
            
            self.model.train()
            
            # Shuffle dataset
            shuffled_indices = np.random.permutation(len(self.train_dataset))
            
            # Process in batches
            for batch_idx in tqdm(range(0, len(shuffled_indices), self.args.batch_size),
                                 desc=f"Epoch {epoch + 1}"):
                # Get batch indices
                batch_indices = shuffled_indices[batch_idx:batch_idx + self.args.batch_size]
                
                # Get batch data
                batch = [self.train_dataset[int(i)] for i in batch_indices]
                prompts = [item['prompt'] for item in batch]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Training step
                try:
                    metrics = self.train_step(prompts)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    if scheduler:
                        scheduler.step()
                    
                    # Record metrics
                    metrics['epoch'] = epoch
                    metrics['step'] = global_step
                    metrics['lr'] = optimizer.param_groups[0]['lr']
                    self.training_history.append(metrics)
                    
                    # Log periodically
                    if global_step % self.args.logging_steps == 0:
                        logger.info(
                            f"Step {global_step}: "
                            f"Loss={metrics['loss']:.4f}, "
                            f"Reward={metrics['mean_reward']:.4f}, "
                            f"KL={metrics['kl_div']:.4f}"
                        )
                    
                    global_step += 1
                    
                except Exception as e:
                    logger.warning(f"Training step failed: {e}")
                    continue
                
                # Save checkpoint periodically
                if global_step % self.config.grpo.save_freq == 0:
                    checkpoint_path = self.checkpoint_dir / f"checkpoint_{global_step}"
                    self.model.save_pretrained(checkpoint_path)
                    logger.info(f"Saved checkpoint at step {global_step}")
            
            # End of epoch - save checkpoint
            epoch_checkpoint_path = self.checkpoint_dir / f"epoch_{epoch + 1}"
            self.model.save_pretrained(epoch_checkpoint_path)
            logger.info(f"Saved epoch {epoch + 1} checkpoint")
        
        # Save final model
        logger.info("Saving final model...")
        self.model.save_pretrained(self.save_dir / "final_model")
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(self.save_dir / "training_history.csv", index=False)
        
        logger.info("=" * 80)
        logger.info("GRPO Training Completed")
        logger.info("=" * 80)
    
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
                )
                
                response_ids = policy_output[0, prompt_ids.shape[1]:]
                
                # Compute log probs
                policy_log_probs = self.compute_log_probs(self.model, prompt_text, response_ids)
                ref_log_probs = self.compute_log_probs(self.ref_model, prompt_text, response_ids)
                
                # Compute KL
                kl_per_token = policy_log_probs - ref_log_probs
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
        fig.suptitle('GRPO Training Curves', fontsize=16)
        
        # Plot 1: Loss
        axes[0, 0].plot(df['step'], df['loss'], linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Mean Reward
        axes[0, 1].plot(df['step'], df['mean_reward'], linewidth=2, color='green')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Mean Reward')
        axes[0, 1].set_title('Mean Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: KL Divergence
        axes[0, 2].plot(df['step'], df['kl_div'], linewidth=2, color='red')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('KL Divergence')
        axes[0, 2].set_title('KL Divergence')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Reward Distribution
        axes[1, 0].plot(df['step'], df['mean_reward'], linewidth=2, label='Mean', color='blue')
        axes[1, 0].fill_between(
            df['step'],
            df['mean_reward'] - df['std_reward'],
            df['mean_reward'] + df['std_reward'],
            alpha=0.3, color='blue'
        )
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Reward Distribution (Mean ± Std)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Advantage Distribution
        axes[1, 1].plot(df['step'], df['mean_advantage'], linewidth=2, label='Mean', color='purple')
        axes[1, 1].fill_between(
            df['step'],
            df['mean_advantage'] - df['std_advantage'],
            df['mean_advantage'] + df['std_advantage'],
            alpha=0.3, color='purple'
        )
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Advantage')
        axes[1, 1].set_title('Advantage Distribution (Mean ± Std)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Learning Rate
        if 'lr' in df.columns:
            axes[1, 2].plot(df['step'], df['lr'], linewidth=2, color='orange')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('Learning Rate Schedule')
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
                if col not in ['epoch', 'step']:
                    final_metrics[col] = float(df[col].iloc[-10:].mean())
        
        summary = {
            'method': 'GRPO',
            'model_name': self.args.model_name,
            'group_size': self.args.group_size,
            'advantage_normalization': self.args.advantage_normalization,
            'use_baseline': self.args.use_baseline,
            'use_lora': self.args.use_lora,
            'lora_r': self.args.lora_r if self.args.use_lora else None,
            'quantization': '8-bit' if self.args.load_in_8bit else ('4-bit' if self.args.load_in_4bit else 'none'),
            'seed': self.args.seed,
            'training': {
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'epochs': self.args.epochs,
                'optimizer': self.args.optimizer,
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
            "# GRPO Training Summary",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Save Directory:** `{self.save_dir}`",
            "",
            "## Model Configuration",
            f"- Base Model: `{self.args.model_name}`",
            f"- Group Size: **{self.args.group_size}**",
            f"- Advantage Normalization: **{self.args.advantage_normalization}**",
            f"- Use Baseline: {self.args.use_baseline}",
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
            f"- Learning Rate: {self.args.learning_rate}",
            f"- Epochs: {self.args.epochs}",
            f"- Optimizer: {self.args.optimizer}",
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
            "- Plots: `plots/`",
        ])
        
        report_text = "\n".join(report_lines)
        
        with open(self.save_dir / "REPORT.md", 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary saved to {self.save_dir / 'summary.json'}")
        logger.info(f"Report saved to {self.save_dir / 'REPORT.md'}")
    
    def run(self):
        """Run complete GRPO training pipeline"""
        logger.info("=" * 80)
        logger.info("Starting GRPO Training Pipeline")
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
            self.train()
            
            # Compute perplexity
            perplexity_result = self.compute_perplexity()
            
            # Estimate KL divergence
            kl_stats = self.estimate_kl_divergence()
            
            # Plot training curves
            self.plot_training_curves()
            
            # Create summary
            self.create_summary(perplexity_result, kl_stats)
            
            logger.info("=" * 80)
            logger.info("GRPO Training Complete!")
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
    # Parse arguments
    args = parse_grpo_args()
    
    # Get config
    config = get_config_from_args(args)
    
    # Create trainer
    trainer = GRPOModelTrainer(args, config)
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()