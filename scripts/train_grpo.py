"""
GRPO (Group Relative Policy Optimization) Training Script - FIXED

This script implements GRPO with proper batching and gradient accumulation
based on TRL's implementation.

Key fixes:
- Batched generation per group
- Batched reward computation
- Proper gradient accumulation
- Single optimizer step per batch
- Removed token clamping hack

Usage:
    python train_grpo_fixed.py \
      --reward_model_path ./models/reward_model/final_model \
      --group_size 8 \
      --batch_size 4 \
      --epochs 3 \
      --seed 42

Reference:
    Shao et al. (2024) - DeepSeekMath: Pushing the Limits of Mathematical 
    Reasoning in Open Language Models
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

PROJECT_ROOT = Path("/kaggle/working/LLM_Alignment")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    BitsAndBytesConfig
)
from datasets import Dataset as HFDataset
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


class FixedGRPOTrainer:
    """Fixed GRPO implementation with proper batching"""
    
    def _init_(self, args, config):
        self.args = args
        self.config = config
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GRPO training; no GPU detected.")

        self.device = torch.device('cuda')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.optimizer = None
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
            run_name = f"grpo_fixed_g{self.args.group_size}seed{self.args.seed}{timestamp}"
            self.save_dir = Path(self.args.output_dir) / run_name
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
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
            logger.info("Set pad_token to eos_token")
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.save_pretrained(self.save_dir / "tokenizer")
        logger.info("✓ Tokenizer ready")
    
    def load_reward_model(self):
        """Load frozen reward model"""
        logger.info(f"Loading reward model from: {self.args.reward_model_path}")
        
        bnb_config = create_quantization_config(
            self.args.load_in_4bit,
            self.args.load_in_8bit,
            self.args.mixed_precision
        )
        
        try:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.reward_model_path,
                num_labels=1,
                quantization_config=bnb_config,
                device_map={"": "cuda:0"},
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                num_labels=1,
                quantization_config=bnb_config,
                device_map={"": "cuda:0"},
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            self.reward_model = PeftModel.from_pretrained(base_model, self.args.reward_model_path)
        
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        self.reward_device = next(self.reward_model.parameters()).device
        logger.info(f"✓ Reward model on {self.reward_device}")
    
    def load_datasets(self):
        """Load training data"""
        logger.info("Loading datasets...")
        
        data_dir = Path(self.args.data_dir)
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        with open(train_path, 'r') as f:
            for line in f:
                self.train_data.append(json.loads(line))
        
        with open(val_path, 'r') as f:
            for line in f:
                self.val_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val examples")
    
    def setup_models(self):
        """Initialize models"""
        logger.info(f"Loading policy model: {self.args.model_name}")
        
        bnb_config = create_quantization_config(
            self.args.load_in_4bit,
            self.args.load_in_8bit,
            self.args.mixed_precision
        )
        
        # Policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config if bnb_config else None,
            device_map={"": "cuda:0"},
            trust_remote_code=self.config.base_model.trust_remote_code,
            torch_dtype=torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16,
        )
        
        if self.args.load_in_4bit or self.args.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        if self.args.use_lora:
            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=self.args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"✓ Applied LoRA (r={self.args.lora_r})")
        
        self.model_device = next(self.model.parameters()).device
        logger.info(f"✓ Policy model on {self.model_device}")
        
        # Reference model
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
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        logger.info("✓ Optimizer ready")
    
    def compute_rewards_batch(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Batch compute rewards"""
        texts = []
        for prompt, response in zip(prompts, responses):
            text = self.config.data.prompt_template.format(prompt=prompt)
            text += self.config.data.response_template.format(response=response)
            texts.append(text)
        
        # Tokenize in batch
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.args.max_length
        ).to(self.reward_device)
        
        # Compute rewards
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            rewards = outputs.logits.squeeze(-1)
        
        return rewards
    
    def generate_group_batched(self, prompt: str) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate group of responses in batch"""
        # Format prompt
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model_device)
        
        # Repeat for batch generation
        input_ids_batch = input_ids.repeat(self.args.group_size, 1)
        
        # Generate all responses in one batch
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids_batch,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract responses
        responses = []
        response_tokens_list = []
        
        for i in range(self.args.group_size):
            response_tokens = outputs[i, input_ids.shape[1]:].to(self.model_device)
            response_text = self.tokenizer.decode(response_tokens.cpu(), skip_special_tokens=True)
            responses.append(response_text)
            response_tokens_list.append(response_tokens)
        
        return responses, response_tokens_list
    
    def compute_log_probs(
        self,
        model: nn.Module,
        prompt_text: str,
        response_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for response"""
        # Determine target device
        if model == self.ref_model:
            target_device = self.ref_device
        else:
            target_device = self.model_device
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(target_device)
        
        # Concatenate
        full_ids = torch.cat([prompt_ids[0], response_tokens.to(target_device)]).unsqueeze(0)
        
        # Get logits
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=full_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probs for actual tokens
        response_log_probs = []
        prompt_len = prompt_ids.shape[1]
        
        for i, token_id in enumerate(response_tokens):
            pos = prompt_len + i - 1
            if pos >= 0:
                token_log_prob = log_probs[pos, token_id]
                response_log_probs.append(token_log_prob)
        
        return torch.stack(response_log_probs)
    
    def compute_group_advantages(self, rewards: np.ndarray, normalization: str = 'rank') -> np.ndarray:
        """Compute group-relative advantages"""
        if normalization == 'rank':
            ranks = stats.rankdata(rewards, method='average')
            advantages = (ranks - np.mean(ranks)) / (np.std(ranks) + 1e-8)
        elif normalization == 'zscore':
            if self.args.use_baseline:
                advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            else:
                advantages = rewards / (np.std(rewards) + 1e-8)
        elif normalization == 'minmax':
            r_min, r_max = np.min(rewards), np.max(rewards)
            if r_max - r_min > 1e-8:
                advantages = 2 * (rewards - r_min) / (r_max - r_min) - 1
            else:
                advantages = np.zeros_like(rewards)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        return advantages
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """Single GRPO training step with proper batching"""
        
        all_prompts = []
        all_responses = []
        all_response_tokens = []
        all_ref_log_probs = []
        
        # Step 1: Generate responses for all prompts (batched per group)
        for prompt in batch_prompts:
            prompt_text = self.config.data.prompt_template.format(prompt=prompt)
            
            # Generate group in batch
            responses, response_tokens_list = self.generate_group_batched(prompt)
            
            # Compute reference log probs for all responses
            for response_tokens in response_tokens_list:
                all_prompts.append(prompt)
                all_responses.append(self.tokenizer.decode(response_tokens.cpu(), skip_special_tokens=True))
                all_response_tokens.append(response_tokens)
                
                ref_log_probs = self.compute_log_probs(
                    self.ref_model,
                    prompt_text,
                    response_tokens
                )
                all_ref_log_probs.append(ref_log_probs)
        
        # Step 2: Compute rewards in batch
        rewards = self.compute_rewards_batch(all_prompts, all_responses)
        
        # Step 3: Compute advantages and losses per group
        total_loss = 0.0
        num_groups = len(batch_prompts)
        
        for group_idx in range(num_groups):
            group_start = group_idx * self.args.group_size
            group_end = group_start + self.args.group_size
            
            # Get group rewards
            group_rewards = rewards[group_start:group_end].cpu().numpy()
            
            # Compute advantages
            advantages = self.compute_group_advantages(
                group_rewards,
                self.args.advantage_normalization
            )
            
            # Update policy for each response in group
            for i in range(self.args.group_size):
                idx = group_start + i
                
                prompt = all_prompts[idx]
                response_tokens = all_response_tokens[idx]
                advantage = advantages[i]
                ref_log_probs = all_ref_log_probs[idx]
                
                # Compute policy log probs
                prompt_text = self.config.data.prompt_template.format(prompt=prompt)
                policy_log_probs = self.compute_log_probs(
                    self.model,
                    prompt_text,
                    response_tokens
                )
                
                # GRPO loss: -advantage * log_ratio
                log_ratio = policy_log_probs - ref_log_probs
                loss = -advantage * log_ratio.mean()
                
                # Normalize by total number of responses
                loss = loss / (num_groups * self.args.group_size)
                
                # Accumulate gradients
                loss.backward()
                
                total_loss += loss.item()
        
        # Return metrics
        return {
            'loss': total_loss,
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item(),
            'num_groups': num_groups,
        }
    
    def train(self):
        """Main training loop"""
        logger.info("\n" + "="*80)
        logger.info("Starting Fixed GRPO Training")
        logger.info("="*80)
        logger.info(f"Training samples: {len(self.train_data)}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Group size: {self.args.group_size}")
        logger.info(f"Advantage normalization: {self.args.advantage_normalization}")
        logger.info("="*80 + "\n")
        
        for epoch in range(self.args.epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.args.epochs}")
            logger.info(f"{'='*80}")
            
            self.model.train()
            
            # Shuffle data
            np.random.shuffle(self.train_data)
            
            # Training loop
            pbar = tqdm(range(0, len(self.train_data), self.args.batch_size), desc=f"Epoch {epoch+1}")
            
            for batch_idx in pbar:
                # Get batch
                batch = self.train_data[batch_idx:batch_idx + self.args.batch_size]
                prompts = [item['prompt'] for item in batch]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                try:
                    # Training step
                    metrics = self.train_step(prompts)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Log metrics
                    self.training_history.append({
                        'epoch': epoch,
                        'step': batch_idx // self.args.batch_size,
                        **metrics
                    })
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'reward': f"{metrics['mean_reward']:.3f}",
                    })
                    
                except Exception as e:
                    logger.warning(f"Training step failed: {e}")
                    continue
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch+1}"
            self.model.save_pretrained(checkpoint_path)
            logger.info(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        self.model.save_pretrained(self.save_dir / "final_model")
        logger.info(f"✓ Saved final model: {self.save_dir / 'final_model'}")
        
        # Save training history
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.logs_dir / "training_history.csv", index=False)
        logger.info(f"✓ Saved training history")
        
        logger.info("\n" + "="*80)
        logger.info("Training Complete!")
        logger.info("="*80)
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not self.training_history:
            return
        
        df = pd.DataFrame(self.training_history)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('GRPO Training Curves', fontsize=16)
        
        # Loss
        axes[0].plot(df.index, df['loss'], linewidth=2)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Reward
        axes[1].plot(df.index, df['mean_reward'], linewidth=2, color='green')
        axes[1].fill_between(
            df.index,
            df['mean_reward'] - df['std_reward'],
            df['mean_reward'] + df['std_reward'],
            alpha=0.3
        )
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Mean Reward')
        axes[1].set_title('Mean Reward ± Std')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved training curves")
    
    def run(self):
        """Run complete training pipeline"""
        try:
            self.save_config()
            self.setup_tokenizer()
            self.load_reward_model()
            self.load_datasets()
            self.setup_models()
            self.train()
            self.plot_training_curves()
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed GRPO Training')
    
    # Basic
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # GRPO
    parser.add_argument('--group_size', type=int, default=8)
    parser.add_argument('--advantage_normalization', type=str, default='rank', 
                       choices=['rank', 'zscore', 'minmax'])
    parser.add_argument('--use_baseline', action='store_true')
    
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
    trainer = FixedGRPOTrainer(args, config)
    
    # Run
    trainer.run()


if __name__ == "_main_":
    main()