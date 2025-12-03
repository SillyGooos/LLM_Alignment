"""
Training Callbacks for Per-Epoch KL and Perplexity Tracking

This module provides callbacks for DPO, PPO, and GRPO training to track:
- Perplexity on instruction-following data (catastrophic forgetting)
- KL divergence vs reference model (model drift)

These metrics are computed at the end of each epoch and logged to:
- training_log.jsonl (for easy graphing)
- Console output
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class PerEpochMetricsCallback(TrainerCallback):
    """
    Callback to compute perplexity and KL divergence at the end of each epoch.
    
    This callback:
    1. Triggers at the end of each epoch
    2. Computes perplexity on instruction-following subset
    3. Computes KL divergence vs reference model
    4. Logs metrics to training_log.jsonl
    5. Prints summary to console
    """
    
    def __init__(
        self,
        trainer_instance,
        save_dir: Path,
        instr_data_path: str,
        reference_model=None,
        tokenizer=None,
        max_samples: int = 500,  # Limit for speed
    ):
        """
        Args:
            trainer_instance: The trainer object (has model, tokenizer, etc.)
            save_dir: Directory to save logs
            instr_data_path: Path to instruction-following data
            reference_model: Reference model for KL computation (optional)
            tokenizer: Tokenizer (optional, will use trainer's if not provided)
            max_samples: Max samples to use for metrics (for speed)
        """
        self.trainer = trainer_instance
        self.save_dir = Path(save_dir)
        self.instr_data_path = instr_data_path
        self.reference_model = reference_model
        self.tokenizer = tokenizer or trainer_instance.tokenizer
        self.max_samples = max_samples
        
        # Create log file
        self.log_file = self.save_dir / "training_log.jsonl"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track metrics
        self.epoch_metrics = []
        
        logger.info(f"PerEpochMetricsCallback initialized")
        logger.info(f"  Log file: {self.log_file}")
        logger.info(f"  Max samples for metrics: {self.max_samples}")
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of each epoch"""
        
        epoch = int(state.epoch)
        logger.info(f"\n{'='*80}")
        logger.info(f"Computing per-epoch metrics for epoch {epoch}")
        logger.info(f"{'='*80}")
        
        # Get model
        model = self.trainer.model
        model.eval()
        
        # Compute metrics
        metrics = {
            'epoch': epoch,
            'global_step': state.global_step,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add training loss if available
        if len(state.log_history) > 0:
            # Find most recent training loss
            for log_entry in reversed(state.log_history):
                if 'loss' in log_entry:
                    metrics['train_loss'] = log_entry['loss']
                    break
        
        # Compute perplexity
        try:
            perplexity = self._compute_perplexity(model)
            metrics['perplexity'] = perplexity
            logger.info(f"  Perplexity: {perplexity:.4f}")
        except Exception as e:
            logger.warning(f"  Failed to compute perplexity: {e}")
            metrics['perplexity'] = None
        
        # Compute KL divergence (if reference model available)
        if self.reference_model is not None:
            try:
                kl_mean, kl_std = self._compute_kl_divergence(model)
                metrics['kl_divergence_mean'] = kl_mean
                metrics['kl_divergence_std'] = kl_std
                logger.info(f"  KL Divergence: {kl_mean:.4f} ± {kl_std:.4f}")
            except Exception as e:
                logger.warning(f"  Failed to compute KL: {e}")
                metrics['kl_divergence_mean'] = None
                metrics['kl_divergence_std'] = None
        
        # Save metrics
        self.epoch_metrics.append(metrics)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        logger.info(f"{'='*80}\n")
        
        # Return model to train mode
        model.train()
        
        return control
    
    def _compute_perplexity(self, model) -> float:
        """Compute perplexity on instruction-following data"""
        
        # Load instruction-following data
        try:
            import json
            with open(self.instr_data_path, 'r') as f:
                instr_data = [json.loads(line) for line in f]
        except:
            logger.warning(f"Could not load instruction data from {self.instr_data_path}")
            return None
        
        # Limit samples
        if len(instr_data) > self.max_samples:
            import random
            instr_data = random.sample(instr_data, self.max_samples)
        
        # Compute perplexity
        total_loss = 0.0
        num_tokens = 0
        
        with torch.no_grad():
            for item in instr_data:
                try:
                    # Format prompt and response
                    prompt = item.get('prompt', '')
                    response = item.get('response', item.get('chosen', ''))
                    
                    if not prompt or not response:
                        continue
                    
                    # Tokenize
                    full_text = f"Question: {prompt}\n\nAnswer: {response}"
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    # Get loss
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                    
                    # Accumulate
                    num_input_tokens = inputs['input_ids'].size(1)
                    total_loss += loss * num_input_tokens
                    num_tokens += num_input_tokens
                    
                except Exception as e:
                    continue
        
        if num_tokens == 0:
            return None
        
        # Compute perplexity
        avg_loss = total_loss / num_tokens
        perplexity = np.exp(avg_loss)
        
        return float(perplexity)
    
    def _compute_kl_divergence(self, model) -> tuple:
        """Compute KL divergence vs reference model"""
        
        # Load validation data
        try:
            import json
            with open(self.instr_data_path, 'r') as f:
                val_data = [json.loads(line) for line in f]
        except:
            logger.warning(f"Could not load data from {self.instr_data_path}")
            return None, None
        
        # Limit samples
        if len(val_data) > self.max_samples:
            import random
            val_data = random.sample(val_data, self.max_samples)
        
        # Compute KL
        kl_values = []
        
        with torch.no_grad():
            for item in val_data:
                try:
                    # Format prompt and response
                    prompt = item.get('prompt', '')
                    response = item.get('response', item.get('chosen', ''))
                    
                    if not prompt or not response:
                        continue
                    
                    # Tokenize
                    full_text = f"Question: {prompt}\n\nAnswer: {response}"
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    # Get policy logits
                    policy_outputs = model(**inputs)
                    policy_logits = policy_outputs.logits
                    
                    # Get reference logits
                    ref_outputs = self.reference_model(**inputs.to(self.reference_model.device))
                    ref_logits = ref_outputs.logits.to(policy_logits.device)
                    
                    # Compute KL per token
                    policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
                    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    
                    # KL(policy || ref)
                    kl_per_token = (
                        torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
                    ).sum(dim=-1)
                    
                    kl_mean = kl_per_token.mean().item()
                    kl_values.append(kl_mean)
                    
                except Exception as e:
                    continue
        
        if len(kl_values) == 0:
            return None, None
        
        kl_mean = float(np.mean(kl_values))
        kl_std = float(np.std(kl_values))
        
        return kl_mean, kl_std


class CustomPPOCallback:
    """
    Custom callback for PPO training (not using HuggingFace Trainer)
    
    Usage in PPO training loop:
        callback = CustomPPOCallback(save_dir, instr_data_path, reference_model, tokenizer)
        
        for epoch in range(num_epochs):
            # ... training code ...
            
            # At end of epoch
            metrics = callback.on_epoch_end(epoch, policy_model)
    """
    
    def __init__(
        self,
        save_dir: Path,
        instr_data_path: str,
        reference_model,
        tokenizer,
        max_samples: int = 500,
    ):
        self.save_dir = Path(save_dir)
        self.instr_data_path = instr_data_path
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        
        # Create log file
        self.log_file = self.save_dir / "training_log.jsonl"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch_metrics = []
        
        logger.info(f"CustomPPOCallback initialized")
        logger.info(f"  Log file: {self.log_file}")
    
    def on_epoch_end(
        self,
        epoch: int,
        model,
        train_loss: Optional[float] = None,
        global_step: Optional[int] = None,
    ) -> Dict:
        """
        Call this at the end of each epoch in PPO/GRPO training
        
        Args:
            epoch: Current epoch number
            model: Policy model
            train_loss: Average training loss for the epoch (optional)
            global_step: Global training step (optional)
        
        Returns:
            Dictionary of computed metrics
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Computing per-epoch metrics for epoch {epoch}")
        logger.info(f"{'='*80}")
        
        model.eval()
        
        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
        }
        
        if train_loss is not None:
            metrics['train_loss'] = train_loss
        
        if global_step is not None:
            metrics['global_step'] = global_step
        
        # Compute perplexity
        try:
            perplexity = self._compute_perplexity(model)
            metrics['perplexity'] = perplexity
            logger.info(f"  Perplexity: {perplexity:.4f}")
        except Exception as e:
            logger.warning(f"  Failed to compute perplexity: {e}")
            metrics['perplexity'] = None
        
        # Compute KL divergence
        try:
            kl_mean, kl_std = self._compute_kl_divergence(model)
            metrics['kl_divergence_mean'] = kl_mean
            metrics['kl_divergence_std'] = kl_std
            logger.info(f"  KL Divergence: {kl_mean:.4f} ± {kl_std:.4f}")
        except Exception as e:
            logger.warning(f"  Failed to compute KL: {e}")
            metrics['kl_divergence_mean'] = None
            metrics['kl_divergence_std'] = None
        
        # Save metrics
        self.epoch_metrics.append(metrics)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        logger.info(f"{'='*80}\n")
        
        model.train()
        
        return metrics
    
    def _compute_perplexity(self, model) -> float:
        """Compute perplexity on instruction-following data"""
        
        try:
            import json
            with open(self.instr_data_path, 'r') as f:
                instr_data = [json.loads(line) for line in f]
        except:
            return None
        
        if len(instr_data) > self.max_samples:
            import random
            instr_data = random.sample(instr_data, self.max_samples)
        
        total_loss = 0.0
        num_tokens = 0
        
        with torch.no_grad():
            for item in instr_data:
                try:
                    prompt = item.get('prompt', '')
                    response = item.get('response', item.get('chosen', ''))
                    
                    if not prompt or not response:
                        continue
                    
                    full_text = f"Question: {prompt}\n\nAnswer: {response}"
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                    
                    num_input_tokens = inputs['input_ids'].size(1)
                    total_loss += loss * num_input_tokens
                    num_tokens += num_input_tokens
                    
                except:
                    continue
        
        if num_tokens == 0:
            return None
        
        avg_loss = total_loss / num_tokens
        perplexity = np.exp(avg_loss)
        
        return float(perplexity)
    
    def _compute_kl_divergence(self, model) -> tuple:
        """Compute KL divergence vs reference model"""
        
        try:
            import json
            with open(self.instr_data_path, 'r') as f:
                val_data = [json.loads(line) for line in f]
        except:
            return None, None
        
        if len(val_data) > self.max_samples:
            import random
            val_data = random.sample(val_data, self.max_samples)
        
        kl_values = []
        
        with torch.no_grad():
            for item in val_data:
                try:
                    prompt = item.get('prompt', '')
                    response = item.get('response', item.get('chosen', ''))
                    
                    if not prompt or not response:
                        continue
                    
                    full_text = f"Question: {prompt}\n\nAnswer: {response}"
                    inputs = self.tokenizer(
                        full_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    policy_outputs = model(**inputs)
                    policy_logits = policy_outputs.logits
                    
                    ref_outputs = self.reference_model(**inputs.to(self.reference_model.device))
                    ref_logits = ref_outputs.logits.to(policy_logits.device)
                    
                    policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
                    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    
                    kl_per_token = (
                        torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
                    ).sum(dim=-1)
                    
                    kl_mean = kl_per_token.mean().item()
                    kl_values.append(kl_mean)
                    
                except:
                    continue
        
        if len(kl_values) == 0:
            return None, None
        
        return float(np.mean(kl_values)), float(np.std(kl_values))