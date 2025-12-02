"""
Reward Model Training Script

This script trains a reward model for use in PPO and GRPO alignment.
The reward model learns to predict scalar rewards for prompt-response pairs,
distinguishing between preferred (chosen) and less-preferred (rejected) responses.

Usage:
    python train_reward_model.py \
      --batch_size 16 \
      --lr 5e-5 \
      --epochs 3 \
      --seed 42 \
      --save_dir ./models/reward_model
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers and HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset, Dataset as HFDataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Import config
from config.default_config import (
    parse_reward_model_args,
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


class RewardDataset(Dataset):
    """
    Dataset for reward model training
    
    For pairwise training: creates pairs of (prompt, chosen) and (prompt, rejected)
    Labels: 1 for chosen, 0 for rejected
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int,
        prompt_template: str,
        response_template: str,
        pairwise: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.pairwise = pairwise
        
        # Create examples
        self.examples = self._create_examples()
    
    def _format_text(self, prompt: str, response: str) -> str:
        """Format prompt and response"""
        formatted_prompt = self.prompt_template.format(prompt=prompt)
        formatted_response = self.response_template.format(response=response)
        return formatted_prompt + formatted_response
    
    def _create_examples(self) -> List[Dict]:
        """Create training examples from data"""
        examples = []
        
        for item in self.data:
            prompt = item['prompt']
            chosen = item['response_w']
            rejected = item['response_l']
            
            if self.pairwise:
                # Create two examples: one for chosen (label=1), one for rejected (label=0)
                examples.append({
                    'text': self._format_text(prompt, chosen),
                    'label': 1,
                    'id': item.get('id', ''),
                    'type': 'chosen'
                })
                examples.append({
                    'text': self._format_text(prompt, rejected),
                    'label': 0,
                    'id': item.get('id', ''),
                    'type': 'rejected'
                })
            else:
                # Alternative: single example with both responses
                # Not implementing this variant for now
                pass
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            example['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example['label'], dtype=torch.float)
        }


class RewardModelTrainer:
    """Main trainer class for reward model"""
    
    def __init__(self, args, config: Config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Metrics storage
        self.metrics_history = []
        self.best_metric = -float('inf')
        
    def setup_paths(self):
        """Setup directory structure"""
        # Create save directory
        if self.args.save_dir:
            self.save_dir = Path(self.args.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(self.config.paths.models_reward) / f"run_{timestamp}"
        
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
        test_path = data_dir / "test.jsonl"
        
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
        
        test_data = []
        if test_path.exists():
            with open(test_path, 'r') as f:
                for line in f:
                    test_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
        
        # Create datasets
        self.train_dataset = RewardDataset(
            train_data,
            self.tokenizer,
            self.args.max_length,
            self.config.data.prompt_template,
            self.config.data.response_template,
            pairwise=True
        )
        
        self.val_dataset = RewardDataset(
            val_data,
            self.tokenizer,
            self.args.max_length,
            self.config.data.prompt_template,
            self.config.data.response_template,
            pairwise=True
        )
        
        if test_data:
            self.test_dataset = RewardDataset(
                test_data,
                self.tokenizer,
                self.args.max_length,
                self.config.data.prompt_template,
                self.config.data.response_template,
                pairwise=True
            )
        
        logger.info(f"Created datasets: train={len(self.train_dataset)}, val={len(self.val_dataset)}")
    
    def setup_model(self):
        """Initialize reward model"""
        logger.info(f"Loading model: {self.args.model_name}")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            num_labels=1,  # Scalar reward output
            load_in_8bit=self.args.load_in_8bit,
            load_in_4bit=self.args.load_in_4bit,
            device_map="auto",
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        
        # Prepare for training if using quantization
        if self.args.load_in_8bit or self.args.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("Prepared model for quantized training")
        
        # Apply LoRA if specified
        if self.args.use_lora:
            logger.info(f"Applying LoRA with rank={self.args.lora_r}")
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=self.config.lora.target_modules,
                bias=self.config.lora.bias,
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        logger.info(f"Model loaded and configured")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        
        # Convert to binary predictions (threshold at 0.5)
        binary_preds = (predictions > 0.5).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(labels, binary_preds)
        
        # For AUC, we need the continuous predictions
        try:
            auc = roc_auc_score(labels, predictions)
        except:
            auc = 0.5  # If AUC cannot be computed
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_preds, average='binary', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Set seed
        set_seed(self.args.seed)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size * 2,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            max_grad_norm=self.args.max_grad_norm,
            
            # Optimizer
            optim=self.args.optimizer,
            lr_scheduler_type=self.args.lr_scheduler_type,
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=self.args.eval_steps,
            save_strategy="steps",
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            greater_is_better=True,
            
            # Logging
            logging_dir=str(self.logs_dir),
            logging_steps=self.args.logging_steps,
            report_to=["tensorboard"],
            
            # Hardware
            fp16=(self.args.mixed_precision == "fp16"),
            bf16=(self.args.mixed_precision == "bf16"),
            dataloader_num_workers=self.args.num_workers,
            
            # Reproducibility
            seed=self.args.seed,
            data_seed=self.args.seed,
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.reward_model.early_stopping_patience,
            early_stopping_threshold=self.config.reward_model.early_stopping_threshold,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping],
        )
        
        # Train
        logger.info("=" * 80)
        logger.info("Training started")
        logger.info("=" * 80)
        
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model(str(self.save_dir / "final_model"))
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info("=" * 80)
        logger.info("Training completed")
        logger.info("=" * 80)
        
        return trainer
    
    def evaluate(self, trainer):
        """Evaluate on validation and test sets"""
        logger.info("Evaluating model...")
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(eval_dataset=self.val_dataset)
        logger.info(f"\nValidation metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save validation metrics
        with open(self.save_dir / "val_metrics.json", 'w') as f:
            json.dump(val_metrics, f, indent=2)
        
        # Evaluate on test set if available
        if self.test_dataset is not None:
            test_metrics = trainer.evaluate(eval_dataset=self.test_dataset)
            logger.info(f"\nTest metrics:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Save test metrics
            with open(self.save_dir / "test_metrics.json", 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        return val_metrics
    
    def compute_reward_scores(self, trainer):
        """Compute and save reward scores for validation set"""
        logger.info("Computing reward scores for validation set...")
        
        self.model.eval()
        
        results = []
        
        with torch.no_grad():
            for i in tqdm(range(len(self.val_dataset)), desc="Computing rewards"):
                item = self.val_dataset[i]
                
                # Move to device
                input_ids = item['input_ids'].unsqueeze(0).to(self.model.device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(self.model.device)
                
                # Get prediction
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                reward = outputs.logits.squeeze().item()
                
                results.append({
                    'index': i,
                    'label': item['labels'].item(),
                    'reward': reward,
                })
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(self.save_dir / "reward_scores.csv", index=False)
        logger.info(f"Saved reward scores to {self.save_dir / 'reward_scores.csv'}")
        
        # Compute statistics
        chosen_rewards = df[df['label'] == 1]['reward']
        rejected_rewards = df[df['label'] == 0]['reward']
        
        stats = {
            'chosen_mean': float(chosen_rewards.mean()),
            'chosen_std': float(chosen_rewards.std()),
            'rejected_mean': float(rejected_rewards.mean()),
            'rejected_std': float(rejected_rewards.std()),
            'margin': float(chosen_rewards.mean() - rejected_rewards.mean()),
        }
        
        logger.info(f"\nReward Statistics:")
        logger.info(f"  Chosen: {stats['chosen_mean']:.4f} ± {stats['chosen_std']:.4f}")
        logger.info(f"  Rejected: {stats['rejected_mean']:.4f} ± {stats['rejected_std']:.4f}")
        logger.info(f"  Margin: {stats['margin']:.4f}")
        
        # Save statistics
        with open(self.save_dir / "reward_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return df, stats
    
    def plot_results(self, reward_df: pd.DataFrame):
        """Generate visualization plots"""
        logger.info("Generating plots...")
        
        # Plot 1: Reward distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        chosen_rewards = reward_df[reward_df['label'] == 1]['reward']
        rejected_rewards = reward_df[reward_df['label'] == 0]['reward']
        
        # Histogram
        axes[0].hist(chosen_rewards, bins=50, alpha=0.6, label='Chosen', color='green')
        axes[0].hist(rejected_rewards, bins=50, alpha=0.6, label='Rejected', color='red')
        axes[0].set_xlabel('Reward Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Reward Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Violin plot
        data_for_violin = pd.DataFrame({
            'Reward': list(chosen_rewards) + list(rejected_rewards),
            'Type': ['Chosen'] * len(chosen_rewards) + ['Rejected'] * len(rejected_rewards)
        })
        sns.violinplot(data=data_for_violin, x='Type', y='Reward', ax=axes[1])
        axes[1].set_title('Reward Distribution (Violin Plot)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "reward_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Sample for visualization (to avoid overcrowding)
        sample_size = min(1000, len(reward_df) // 2)
        chosen_sample = chosen_rewards.sample(min(sample_size, len(chosen_rewards)))
        rejected_sample = rejected_rewards.sample(min(sample_size, len(rejected_rewards)))
        
        ax.scatter(range(len(chosen_sample)), chosen_sample, alpha=0.5, 
                  label='Chosen', color='green', s=10)
        ax.scatter(range(len(rejected_sample)), rejected_sample, alpha=0.5,
                  label='Rejected', color='red', s=10)
        
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Reward Score')
        ax.set_title('Reward Scores: Chosen vs Rejected')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "reward_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {self.plots_dir}")
    
    def create_summary(self, val_metrics: Dict, reward_stats: Dict):
        """Create summary report"""
        logger.info("Creating summary report...")
        
        summary = {
            'model_name': self.args.model_name,
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
            'validation_metrics': val_metrics,
            'reward_statistics': reward_stats,
        }
        
        # Save as JSON
        with open(self.save_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        report_lines = [
            "# Reward Model Training Summary",
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
            f"- Learning Rate: {self.args.learning_rate}",
            f"- Epochs: {self.args.epochs}",
            f"- Optimizer: {self.args.optimizer}",
            f"- Seed: {self.args.seed}",
            "",
            "## Validation Metrics",
        ])
        
        for key, value in val_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"- {key}: {value:.4f}")
            else:
                report_lines.append(f"- {key}: {value}")
        
        report_lines.extend([
            "",
            "## Reward Statistics",
            f"- Chosen (mean ± std): {reward_stats['chosen_mean']:.4f} ± {reward_stats['chosen_std']:.4f}",
            f"- Rejected (mean ± std): {reward_stats['rejected_mean']:.4f} ± {reward_stats['rejected_std']:.4f}",
            f"- **Margin**: {reward_stats['margin']:.4f}",
            "",
            "## Plots",
            "- Reward distributions: `plots/reward_distributions.png`",
            "- Reward scatter: `plots/reward_scatter.png`",
            "",
            "## Files",
            "- Final model: `final_model/`",
            "- Tokenizer: `tokenizer/`",
            "- Training logs: `logs/`",
            "- Reward scores: `reward_scores.csv`",
        ])
        
        report_text = "\n".join(report_lines)
        
        with open(self.save_dir / "REPORT.md", 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary saved to {self.save_dir / 'summary.json'}")
        logger.info(f"Report saved to {self.save_dir / 'REPORT.md'}")
    
    def run(self):
        """Run complete training pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Reward Model Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Save configuration
            self.save_config()
            
            # Setup tokenizer
            self.setup_tokenizer()
            
            # Load datasets
            self.load_datasets()
            
            # Setup model
            self.setup_model()
            
            # Train
            trainer = self.train()
            
            # Evaluate
            val_metrics = self.evaluate(trainer)
            
            # Compute reward scores
            reward_df, reward_stats = self.compute_reward_scores(trainer)
            
            # Plot results
            self.plot_results(reward_df)
            
            # Create summary
            self.create_summary(val_metrics, reward_stats)
            
            logger.info("=" * 80)
            logger.info("Reward Model Training Complete!")
            logger.info("=" * 80)
            logger.info(f"\nResults saved to: {self.save_dir}")
            logger.info(f"Best model: {self.save_dir / 'final_model'}")
            logger.info(f"Validation AUC: {val_metrics.get('eval_auc', 0):.4f}")
            logger.info(f"Reward Margin: {reward_stats['margin']:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_reward_model_args()
    
    # Get config
    config = get_config_from_args(args)
    
    # Create trainer
    trainer = RewardModelTrainer(args, config)
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()