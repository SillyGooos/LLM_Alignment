"""
Custom PPO (Proximal Policy Optimization) Implementation

This script implements PPO from scratch for aligning language models with human preferences.
NO TRL dependencies - pure PyTorch implementation.

Supports both sparse rewards (final response only) and dense rewards (token-level).

Usage:
    # Sparse reward PPO
    python train_ppo_custom.py \
      --reward_model_path ./models/reward_model/final_model \
      --reward_mode sparse \
      --batch_size 8 \
      --learning_rate 1e-5 \
      --kl_coef 0.05 \
      --epochs 3 \
      --seed 42

Reference:
    Schulman et al. (2017) - Proximal Policy Optimization Algorithms
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys

PROJECT_ROOT = Path("/kaggle/working/LLM_Alignment")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

from config.default_config import get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_quantization_config(load_in_4bit=True, load_in_8bit=False, mixed_precision='fp16'):
    """Create BitsAndBytesConfig for 4-bit quantization"""
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if mixed_precision == "fp16" else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("✓ Created 4-bit quantization config (NF4)")
        return bnb_config
    elif load_in_8bit:
        logger.info("✓ Using 8-bit quantization")
        return None
    else:
        logger.info("✓ No quantization")
        return None


class ValueHead(nn.Module):
    """Simple value head for PPO"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        # Initialize with small weights
        self.value_head.weight.data.normal_(mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size, seq_len]
        """
        values = self.value_head(hidden_states).squeeze(-1)
        return values


class CustomPPOTrainer:
    """Custom PPO implementation from scratch"""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for PPO training; no GPU detected.")

        self.device = torch.device('cuda')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.policy_model = None  # Policy model (language model)
        self.value_head = None  # Value function head
        self.ref_model = None  # Frozen reference model
        self.reward_model = None  # Frozen reward model
        self.optimizer = None
        
        # Training data
        self.train_data = []
        self.val_data = []
        
        # Metrics
        self.training_history = []
        
    def setup_paths(self):
        """Setup directory structure"""
        if self.args.save_dir:
            self.save_dir = Path(self.args.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ppo_custom_{self.args.reward_mode}_seed{self.args.seed}_{timestamp}"
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
        """Save configuration"""
        args_path = self.save_dir / "args.json"
        with open(args_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        logger.info(f"Saved args to {args_path}")
    
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        logger.info(f"Loading tokenizer: {self.args.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token")
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.save_pretrained(self.save_dir / "tokenizer")
        logger.info("✓ Tokenizer ready")
    
    def load_reward_model(self):
        """Load frozen reward model"""
        logger.info(f"Loading reward model from: {self.args.reward_model_path}")
        
        try:
            # Try loading as full model
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.reward_model_path,
                num_labels=1,
                load_in_4bit=True,
                device_map={"": "cuda:0"},
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except:
            # Try loading as adapter
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                num_labels=1,
                load_in_4bit=True,
                device_map={"": "cuda:0"},
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            self.reward_model = PeftModel.from_pretrained(base_model, self.args.reward_model_path)
        
        # Freeze
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        self.reward_device = next(self.reward_model.parameters()).device
        logger.info(f"✓ Reward model loaded on {self.reward_device}")
    
    def load_datasets(self):
        """Load training data"""
        logger.info("Loading datasets...")
        
        data_dir = Path(self.args.data_dir)
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # Load JSONL
        with open(train_path, 'r') as f:
            for line in f:
                self.train_data.append(json.loads(line))
        
        with open(val_path, 'r') as f:
            for line in f:
                self.val_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val examples")
    
    def setup_models(self):
        """Initialize policy, value head, and reference models"""
        logger.info(f"Loading policy model: {self.args.model_name}")
        
        bnb_config = create_quantization_config(
            self.args.load_in_4bit,
            self.args.load_in_8bit,
            self.args.mixed_precision
        )
        
        # Load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config if bnb_config else None,
            device_map={"": "cuda:0"},
            trust_remote_code=self.config.base_model.trust_remote_code,
            torch_dtype=torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16,
        )
        
        if self.args.load_in_4bit or self.args.load_in_8bit:
            self.policy_model = prepare_model_for_kbit_training(self.policy_model)
        
        # Apply LoRA if requested
        if self.args.use_lora:
            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=self.args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            logger.info(f"✓ Applied LoRA (r={self.args.lora_r})")
        
        self.policy_device = next(self.policy_model.parameters()).device
        logger.info(f"✓ Policy model on {self.policy_device}")
        
        # Create value head
        hidden_size = self.policy_model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.policy_device)
        logger.info(f"✓ Value head created (hidden_size={hidden_size})")
        
        # Load reference model (frozen copy)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config if bnb_config else None,
            device_map={"": "cuda:0"},
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        self.ref_device = next(self.ref_model.parameters()).device
        logger.info(f"✓ Reference model on {self.ref_device}")
        
        # Setup optimizer (only for policy + value head)
        trainable_params = list(self.policy_model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        logger.info("✓ Optimizer ready")
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute reward for prompt-response pair"""
        text = self.config.data.prompt_template.format(prompt=prompt)
        text += self.config.data.response_template.format(response=response)
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.args.max_length
        )
        
        inputs = {k: v.to(self.reward_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            # ✅ EXPLICIT: Force reward to device then extract
            reward = outputs.logits.to(self.reward_device).squeeze().item()
        
        return reward
    
    def generate_response(self, prompt: str) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate response for a prompt
        
        Returns:
            response_text: Generated text
            response_ids: Token IDs [seq_len]
            log_probs: Log probabilities [seq_len]
            values: Value estimates [seq_len]
        """
        # Format prompt
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        
        # ✅ EXPLICIT: Tokenize and force to policy device
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.policy_device)
        
        # Generate with policy
        self.policy_model.eval()
        with torch.no_grad():
            outputs = self.policy_model.generate(
                prompt_ids,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # ✅ EXPLICIT: Extract response tokens and force to GPU
        full_ids = outputs.sequences[0].to(self.policy_device)
        response_ids = full_ids[prompt_ids.shape[1]:].to(self.policy_device)
        
        # Decode response (need CPU)
        response_text = self.tokenizer.decode(response_ids.cpu(), skip_special_tokens=True)
        
        # Compute log probs and values for response tokens
        with torch.no_grad():
            # ✅ EXPLICIT: Get model outputs on GPU
            model_outputs = self.policy_model(input_ids=full_ids.unsqueeze(0).to(self.policy_device), output_hidden_states=True)
            logits = model_outputs.logits[0].to(self.policy_device)
            hidden_states = model_outputs.hidden_states[-1][0].to(self.policy_device)
            
            # ✅ EXPLICIT: Compute log probs on GPU
            log_probs_all = F.log_softmax(logits, dim=-1).to(self.policy_device)
            
            # Get log probs of actual tokens
            response_log_probs = []
            for i in range(len(response_ids)):
                token_id = response_ids[i].to(self.policy_device)  # ✅ EXPLICIT
                pos = prompt_ids.shape[1] + i - 1
                if pos >= 0:
                    log_prob = log_probs_all[pos, token_id].to(self.policy_device)  # ✅ EXPLICIT
                else:
                    log_prob = torch.tensor(0.0, device=self.policy_device)  # ✅ EXPLICIT
                response_log_probs.append(log_prob)
            
            # ✅ EXPLICIT: Stack on GPU
            response_log_probs = torch.stack(response_log_probs).to(self.policy_device)
            
            # ✅ EXPLICIT: Compute values on GPU
            values = self.value_head(hidden_states.to(self.policy_device)).to(self.policy_device)
            response_values = values[prompt_ids.shape[1]:].to(self.policy_device)
        
        return response_text, response_ids, response_log_probs, response_values
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: [seq_len] - rewards for each token
            values: [seq_len] - value estimates for each token
        
        Returns:
            advantages: [seq_len]
            returns: [seq_len] - target values for value function
        """
        gamma = self.args.gamma
        lam = self.args.lam
        
        # Ensure tensors are on same device
        rewards = rewards.to(self.policy_device)
        values = values.to(self.policy_device)
        
        advantages = torch.zeros_like(rewards, device=self.policy_device)
        returns = torch.zeros_like(rewards, device=self.policy_device)
        
        # GAE calculation (backward pass)
        gae = 0
        next_value = 0  # Terminal value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1].item()
            
            # TD error
            delta = rewards[t].item() + gamma * next_value - values[t].item()
            
            # GAE
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            
            # Return (for value function target)
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def compute_policy_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute clipped PPO policy loss
        
        Args:
            old_log_probs: [seq_len] - log probs from rollout
            new_log_probs: [seq_len] - log probs from current policy
            advantages: [seq_len] - advantage estimates
        
        Returns:
            loss: scalar
        """
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate
        clip_range = self.args.cliprange
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        # Policy loss (negative because we want to maximize)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        
        return policy_loss
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute clipped value function loss
        
        Args:
            values: [seq_len] - current value estimates
            returns: [seq_len] - target returns
            old_values: [seq_len] - value estimates from rollout
        
        Returns:
            loss: scalar
        """
        # Unclipped loss
        value_loss1 = F.mse_loss(values, returns, reduction='none')
        
        # Clipped loss
        clip_range = self.args.cliprange_value
        clipped_values = old_values + torch.clamp(
            values - old_values,
            -clip_range,
            clip_range
        )
        value_loss2 = F.mse_loss(clipped_values, returns, reduction='none')
        
        # Take max (more conservative)
        value_loss = torch.max(value_loss1, value_loss2).mean()
        
        return value_loss
    
    def ppo_update(
        self,
        prompt: str,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update on a single example (accumulate gradients, no optimizer step)
        
        Returns:
            metrics: Dict of losses
        """
        # ✅ EXPLICIT: Ensure all tensors are on policy device
        response_ids = response_ids.to(self.policy_device)
        old_log_probs = old_log_probs.to(self.policy_device)
        old_values = old_values.to(self.policy_device)
        advantages = advantages.to(self.policy_device)
        returns = returns.to(self.policy_device)
        
        # Format full input
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        # ✅ EXPLICIT: Force prompt to GPU
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.policy_device)
        # ✅ EXPLICIT: Concatenate on GPU
        full_ids = torch.cat([prompt_ids[0].to(self.policy_device), response_ids.to(self.policy_device)]).to(self.policy_device)
        
        # Forward pass with current policy
        self.policy_model.train()
        # ✅ EXPLICIT: Force inputs to GPU
        model_outputs = self.policy_model(input_ids=full_ids.unsqueeze(0).to(self.policy_device), output_hidden_states=True)
        logits = model_outputs.logits[0].to(self.policy_device)
        hidden_states = model_outputs.hidden_states[-1][0].to(self.policy_device)
        
        # ✅ EXPLICIT: Compute new log probs on GPU
        log_probs_all = F.log_softmax(logits, dim=-1).to(self.policy_device)
        new_log_probs = []
        for i in range(len(response_ids)):
            token_id = response_ids[i].to(self.policy_device)  # ✅ EXPLICIT
            pos = prompt_ids.shape[1] + i - 1
            if pos >= 0:
                log_prob = log_probs_all[pos, token_id].to(self.policy_device)  # ✅ EXPLICIT
            else:
                log_prob = torch.tensor(0.0, device=self.policy_device)  # ✅ EXPLICIT
            new_log_probs.append(log_prob)
        new_log_probs = torch.stack(new_log_probs).to(self.policy_device)  # ✅ EXPLICIT
        
        # ✅ EXPLICIT: Compute new values on GPU
        new_values = self.value_head(hidden_states.to(self.policy_device)).to(self.policy_device)
        new_values = new_values[prompt_ids.shape[1]:].to(self.policy_device)
        
        # Compute losses
        policy_loss = self.compute_policy_loss(old_log_probs, new_log_probs, advantages).to(self.policy_device)
        value_loss = self.compute_value_loss(new_values, returns, old_values).to(self.policy_device)
        
        # KL penalty (policy vs reference)
        with torch.no_grad():
            # ✅ EXPLICIT: Ref model forward pass
            ref_outputs = self.ref_model(input_ids=full_ids.unsqueeze(0).to(self.ref_device))
            ref_logits = ref_outputs.logits[0].to(self.ref_device)
            ref_log_probs_all = F.log_softmax(ref_logits, dim=-1).to(self.ref_device)
            
            ref_log_probs = []
            for i in range(len(response_ids)):
                token_id = response_ids[i].to(self.ref_device)  # ✅ EXPLICIT
                pos = prompt_ids.shape[1] + i - 1
                if pos >= 0:
                    log_prob = ref_log_probs_all[pos, token_id].to(self.ref_device)  # ✅ EXPLICIT
                else:
                    log_prob = torch.tensor(0.0, device=self.ref_device)  # ✅ EXPLICIT
                ref_log_probs.append(log_prob)
            ref_log_probs = torch.stack(ref_log_probs).to(self.policy_device)  # ✅ EXPLICIT: Move to policy device
        
        # ✅ EXPLICIT: KL computation on policy device
        kl_div = (new_log_probs.to(self.policy_device) - ref_log_probs.to(self.policy_device)).mean().to(self.policy_device)
        
        # ✅ EXPLICIT: Total loss on GPU
        total_loss = (policy_loss + self.args.vf_coef * value_loss + self.args.kl_coef * kl_div).to(self.policy_device)
        
        # Backward (accumulate gradients - NO optimizer step here)
        total_loss.backward()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item(),
        }
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{self.args.epochs}")
        logger.info(f"{'='*80}")
        
        # Shuffle training data
        np.random.shuffle(self.train_data)
        
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'kl_div': [],
            'total_loss': [],
            'reward': [],
        }
        
        pbar = tqdm(self.train_data[:self.args.max_samples_per_epoch], desc=f"Epoch {epoch+1}")
        
        for example in pbar:
            prompt = example['prompt']
            
            # Zero gradients at the start
            self.optimizer.zero_grad()
            
            # Generate response
            response_text, response_ids, log_probs, values = self.generate_response(prompt)
            
            # Compute reward
            if self.args.reward_mode == 'sparse':
                # Single reward at the end
                reward_value = self.compute_reward(prompt, response_text)
                rewards = torch.zeros(len(response_ids), device=self.policy_device)
                rewards[-1] = reward_value  # Only last token gets reward
            else:  # dense
                # Token-level rewards
                rewards = []
                for i in range(len(response_ids)):
                    partial_text = self.tokenizer.decode(response_ids[:i+1].cpu(), skip_special_tokens=True)
                    reward_value = self.compute_reward(prompt, partial_text)
                    rewards.append(reward_value)
                rewards = torch.tensor(rewards, device=self.policy_device)
            
            # Compute advantages
            advantages, returns = self.compute_advantages(rewards, values)
            
            # Normalize advantages
            if self.args.whiten_rewards:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO updates (multiple epochs on same data)
            for ppo_epoch in range(self.args.num_ppo_epochs):
                metrics = self.ppo_update(
                    prompt,
                    response_ids,
                    log_probs,
                    values,
                    advantages,
                    returns
                )
            
            # Clip gradients and optimizer step (after all PPO epochs)
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_model.parameters()) + list(self.value_head.parameters()),
                self.args.max_grad_norm
            )
            self.optimizer.step()
            
            # Log metrics
            for key in epoch_metrics:
                if key == 'reward':
                    epoch_metrics[key].append(rewards.sum().item())
                elif key in metrics:
                    epoch_metrics[key].append(metrics[key])
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f"{np.mean(epoch_metrics['reward']):.3f}",
                'policy_loss': f"{np.mean(epoch_metrics['policy_loss']):.3f}",
                'kl': f"{np.mean(epoch_metrics['kl_div']):.4f}",
            })
        
        # Compute epoch averages
        epoch_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
        epoch_avg['epoch'] = epoch
        
        self.training_history.append(epoch_avg)
        
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Reward: {epoch_avg['reward']:.4f}")
        logger.info(f"  Policy Loss: {epoch_avg['policy_loss']:.4f}")
        logger.info(f"  Value Loss: {epoch_avg['value_loss']:.4f}")
        logger.info(f"  KL Divergence: {epoch_avg['kl_div']:.4f}")
        
        return epoch_avg
    
    def train(self):
        """Main training loop"""
        logger.info("\n" + "="*80)
        logger.info("Starting PPO Training")
        logger.info("="*80)
        logger.info(f"Training samples: {len(self.train_data)}")
        logger.info(f"Max samples per epoch: {self.args.max_samples_per_epoch}")
        logger.info(f"Reward mode: {self.args.reward_mode}")
        logger.info(f"PPO epochs per sample: {self.args.num_ppo_epochs}")
        logger.info(f"KL coefficient: {self.args.kl_coef}")
        logger.info("="*80 + "\n")
        
        for epoch in range(self.args.epochs):
            epoch_metrics = self.train_epoch(epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)
        
        # Save final model
        self.save_final_model()
        
        # Save training history
        self.save_training_history()
        
        # Plot curves
        self.plot_training_curves()
        
        logger.info("\n" + "="*80)
        logger.info("Training Complete!")
        logger.info("="*80)
        logger.info(f"Results saved to: {self.save_dir}")
    
    def save_checkpoint(self, epoch: int):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch+1}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save policy model
        if self.args.use_lora:
            self.policy_model.save_pretrained(checkpoint_path / "policy_lora")
        else:
            self.policy_model.save_pretrained(checkpoint_path / "policy")
        
        # Save value head
        torch.save(self.value_head.state_dict(), checkpoint_path / "value_head.pt")
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_path = self.save_dir / "final_model"
        final_path.mkdir(exist_ok=True)
        
        if self.args.use_lora:
            self.policy_model.save_pretrained(final_path / "policy_lora")
        else:
            self.policy_model.save_pretrained(final_path / "policy")
        
        torch.save(self.value_head.state_dict(), final_path / "value_head.pt")
        
        logger.info(f"✓ Final model saved to {final_path}")
    
    def save_training_history(self):
        """Save training history"""
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.logs_dir / "training_history.csv", index=False)
        
        with open(self.logs_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"✓ Training history saved")
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not self.training_history:
            return
        
        df = pd.DataFrame(self.training_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PPO Training Curves', fontsize=16)
        
        # Reward
        axes[0, 0].plot(df['epoch'], df['reward'], 'g-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Policy Loss
        axes[0, 1].plot(df['epoch'], df['policy_loss'], 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Value Loss
        axes[1, 0].plot(df['epoch'], df['value_loss'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # KL Divergence
        axes[1, 1].plot(df['epoch'], df['kl_div'], 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('KL Divergence')
        axes[1, 1].set_title('KL Divergence (vs Reference)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Training curves saved to {self.plots_dir}")
    
    def run(self):
        """Run complete training pipeline"""
        try:
            # Save config
            self.save_config()
            
            # Setup
            self.setup_tokenizer()
            self.load_reward_model()
            self.load_datasets()
            self.setup_models()
            
            # Train
            self.train()
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom PPO Training')
    
    # Basic
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_samples_per_epoch', type=int, default=500)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    
    # PPO
    parser.add_argument('--reward_mode', type=str, default='sparse', choices=['sparse', 'dense'])
    parser.add_argument('--num_ppo_epochs', type=int, default=4)
    parser.add_argument('--whiten_rewards', action='store_true')
    parser.add_argument('--kl_coef', type=float, default=0.05)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--cliprange_value', type=float, default=0.2)
    parser.add_argument('--vf_coef', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.95)
    
    # Optimization
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
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
    
    # Quantization
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true', default=True)
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    
    # LoRA
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get config
    config = get_default_config()
    
    # Create trainer
    trainer = CustomPPOTrainer(args, config)
    
    # Run
    trainer.run()


if __name__ == "__main__":
    main()