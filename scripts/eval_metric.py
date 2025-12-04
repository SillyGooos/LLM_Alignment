"""
Evaluation Metrics Script

This script computes comprehensive metrics on generated outputs:
- Reward scores (from reward model)
- KL divergence (vs reference model)
- Perplexity
- Length statistics
- Compliance rates (for length-constrained prompts)
- Quality proxies

Usage:
    python eval_metrics.py \
      --generated_file ./eval/dpo_outputs.jsonl \
      --reward_model_path ./models/reward_model/final_model \
      --reference_model_path HuggingFaceTB/SmolLM2-135M-Instruct \
      --output_file ./eval/dpo_metrics.csv \
      --seed 42
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Transformers and HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
)
from peft import PeftModel
# Add project root to path for Kaggle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
# Import config
from config.default_config import get_default_config, Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsComputer:
    """Main class for computing evaluation metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.tokenizer = None
        self.reward_model = None
        self.reference_model = None
        self.policy_model = None
        
        # Data
        self.generated_outputs = None
        self.metrics = []
        
    def load_generated_outputs(self, path: str):
        """Load generated outputs"""
        logger.info(f"Loading generated outputs from: {path}")
        
        self.generated_outputs = []
        with open(path, 'r') as f:
            for line in f:
                self.generated_outputs.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.generated_outputs)} generated outputs")
    
    def setup_tokenizer(self, model_path: str):
        """Initialize tokenizer"""
        logger.info(f"Loading tokenizer from: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except:
            logger.warning(f"Could not load tokenizer from {model_path}, using base model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model.model_name,
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer loaded")
    
    def load_reward_model(self, reward_model_path: str):
        """Load frozen reward model"""
        if not reward_model_path:
            logger.warning("No reward model path provided, skipping reward computation")
            return
        
        logger.info(f"Loading reward model from: {reward_model_path}")
        
        try:
            # Try loading as full model
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_path,
                num_labels=1,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"Failed to load as full model: {e}")
            try:
                # Try loading as adapter
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.base_model.model_name,
                    num_labels=1,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=self.config.base_model.trust_remote_code,
                )
                self.reward_model = PeftModel.from_pretrained(base_model, reward_model_path)
            except Exception as e2:
                logger.error(f"Failed to load reward model: {e2}")
                self.reward_model = None
                return
        
        # CRITICAL FIX: Disable gradient checkpointing for quantized reward model
        if hasattr(self.reward_model, 'gradient_checkpointing_disable'):
            self.reward_model.gradient_checkpointing_disable()
            logger.info("✓ Disabled gradient checkpointing for reward model")
        
        self.reward_model.eval()
        logger.info("Reward model loaded")
    
    def load_reference_model(self, reference_model_path: str):
        """Load reference model for KL computation"""
        if not reference_model_path:
            logger.warning("No reference model path provided, skipping KL computation")
            return
        
        logger.info(f"Loading reference model from: {reference_model_path}")
        
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        
        # CRITICAL FIX: Disable gradient checkpointing for quantized reference model
        if hasattr(self.reference_model, 'gradient_checkpointing_disable'):
            self.reference_model.gradient_checkpointing_disable()
            logger.info("✓ Disabled gradient checkpointing for reference model")
        
        self.reference_model.eval()
        logger.info("Reference model loaded")
    
    def load_policy_model(self, policy_model_path: str):
        """Load policy model for KL computation"""
        logger.info(f"Loading policy model from: {policy_model_path}")
        
        model_path = Path(policy_model_path)
        adapter_config_path = model_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            logger.info("Detected LoRA adapter")
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', self.config.base_model.model_name)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            
            # CRITICAL FIX: Disable gradient checkpointing for quantized base model
            if hasattr(base_model, 'gradient_checkpointing_disable'):
                base_model.gradient_checkpointing_disable()
                logger.info("✓ Disabled gradient checkpointing for base model")
            
            self.policy_model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            
            # CRITICAL FIX: Disable gradient checkpointing for quantized policy model
            if hasattr(self.policy_model, 'gradient_checkpointing_disable'):
                self.policy_model.gradient_checkpointing_disable()
                logger.info("✓ Disabled gradient checkpointing for policy model")
        
        self.policy_model.eval()
        logger.info("Policy model loaded")
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute reward score"""
        if self.reward_model is None:
            return None
        
        # Format text
        text = self.config.data.prompt_template.format(prompt=prompt)
        text += self.config.data.response_template.format(response=response)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.base_model.max_length
        ).to(self.reward_model.device)
        
        # Get reward
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            reward = outputs.logits.squeeze().item()
        
        return reward
    
    def compute_kl_divergence(self, prompt: str, response: str) -> Optional[float]:
        """Compute KL divergence between policy and reference"""
        if self.policy_model is None or self.reference_model is None:
            return None
        
        # Format full text
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        response_text = self.config.data.response_template.format(response=response)
        full_text = prompt_text + response_text
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.base_model.max_length
        )
        
        prompt_length = len(self.tokenizer.encode(prompt_text))
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.policy_model.device)
        
        # Get logits from both models
        with torch.no_grad():
            policy_outputs = self.policy_model(input_ids=input_ids)
            policy_logits = policy_outputs.logits
            
            ref_outputs = self.reference_model(input_ids=input_ids.to(self.reference_model.device))
            ref_logits = ref_outputs.logits.to(policy_logits.device)
        
        # Compute log probs
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Get log probs for actual tokens (only response part)
        token_ids = input_ids[:, 1:]  # Shift for next-token prediction
        
        # Only compute KL for response tokens
        response_start = prompt_length - 1
        response_token_ids = token_ids[:, response_start:]
        
        if response_token_ids.shape[1] == 0:
            return 0.0
        
        # Gather log probs
        policy_token_log_probs = torch.gather(
            policy_log_probs[:, response_start:-1, :],
            dim=2,
            index=response_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        ref_token_log_probs = torch.gather(
            ref_log_probs[:, response_start:-1, :],
            dim=2,
            index=response_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute KL per token
        kl_per_token = policy_token_log_probs - ref_token_log_probs
        
        # Average over tokens
        kl_mean = kl_per_token.mean().item()
        
        return kl_mean
    
    def compute_perplexity(self, prompt: str, response: str) -> Optional[float]:
        """Compute perplexity of response under policy model"""
        if self.policy_model is None:
            return None
        
        # Format text
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        response_text = self.config.data.response_template.format(response=response)
        full_text = prompt_text + response_text
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.base_model.max_length
        ).to(self.policy_model.device)
        
        # Get loss
        with torch.no_grad():
            outputs = self.policy_model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
        
        # Compute perplexity
        perplexity = np.exp(loss)
        
        return perplexity
    
    def check_compliance(self, prompt: str, response: str, response_tokens: int) -> Dict:
        """Check if response complies with length constraints in prompt"""
        compliance_info = {
            'has_constraint': False,
            'constraint_type': None,
            'constraint_value': None,
            'actual_value': response_tokens,
            'compliant': None,
            'deviation': None,
        }
        
        # Common patterns for length constraints
        patterns = [
            (r'in (\d+) words or less', 'words'),
            (r'in (\d+) words', 'words'),
            (r'(\d+) words', 'words'),
            (r'in (\d+) sentences', 'sentences'),
            (r'one sentence', 'sentences', 1),
            (r'two sentences', 'sentences', 2),
            (r'briefly', 'brief', None),
            (r'concisely', 'brief', None),
            (r'in detail', 'detailed', None),
        ]
        
        for pattern_info in patterns:
            if len(pattern_info) == 2:
                pattern, constraint_type = pattern_info
                match = re.search(pattern, prompt.lower())
                if match:
                    compliance_info['has_constraint'] = True
                    compliance_info['constraint_type'] = constraint_type
                    compliance_info['constraint_value'] = int(match.group(1))
                    break
            else:
                pattern, constraint_type, value = pattern_info
                if re.search(pattern, prompt.lower()):
                    compliance_info['has_constraint'] = True
                    compliance_info['constraint_type'] = constraint_type
                    compliance_info['constraint_value'] = value
                    break
        
        # Check compliance
        if compliance_info['has_constraint']:
            if compliance_info['constraint_type'] == 'words':
                # Approximate words from tokens (rough estimate: tokens * 0.75)
                estimated_words = int(response_tokens * 0.75)
                compliance_info['actual_value'] = estimated_words
                
                if compliance_info['constraint_value']:
                    compliance_info['compliant'] = estimated_words <= compliance_info['constraint_value']
                    compliance_info['deviation'] = estimated_words - compliance_info['constraint_value']
            
            elif compliance_info['constraint_type'] == 'sentences':
                # Count sentences
                num_sentences = len(re.findall(r'[.!?]+', response))
                compliance_info['actual_value'] = num_sentences
                
                if compliance_info['constraint_value']:
                    compliance_info['compliant'] = num_sentences <= compliance_info['constraint_value']
                    compliance_info['deviation'] = num_sentences - compliance_info['constraint_value']
            
            elif compliance_info['constraint_type'] == 'brief':
                # Brief means < 50 tokens (arbitrary threshold)
                compliance_info['compliant'] = response_tokens < 50
                compliance_info['deviation'] = response_tokens - 50
            
            elif compliance_info['constraint_type'] == 'detailed':
                # Detailed means > 100 tokens (arbitrary threshold)
                compliance_info['compliant'] = response_tokens > 100
                compliance_info['deviation'] = 100 - response_tokens
        
        return compliance_info
    
    def compute_all_metrics(self, output: Dict) -> Dict:
        """Compute all metrics for a single output"""
        prompt = output['prompt']
        response = output['response']
        num_tokens = output['num_tokens']
        
        metrics = {
            'prompt_id': output['prompt_id'],
            'generation_config': output['generation_config'],
            'sample_idx': output.get('sample_idx', 0),
        }
        
        # Add category if available
        if 'category' in output:
            metrics['category'] = output['category']
        
        # Length metrics (already computed)
        metrics['length_tokens'] = num_tokens
        metrics['length_chars'] = output['num_chars']
        
        # Reward score
        try:
            reward = self.compute_reward(prompt, response)
            metrics['reward_score'] = reward
        except Exception as e:
            logger.warning(f"Failed to compute reward: {e}")
            metrics['reward_score'] = None
        
        # KL divergence
        try:
            kl = self.compute_kl_divergence(prompt, response)
            metrics['kl_divergence'] = kl
        except Exception as e:
            logger.warning(f"Failed to compute KL: {e}")
            metrics['kl_divergence'] = None
        
        # Perplexity
        try:
            ppl = self.compute_perplexity(prompt, response)
            metrics['perplexity'] = ppl
        except Exception as e:
            logger.warning(f"Failed to compute perplexity: {e}")
            metrics['perplexity'] = None
        
        # Compliance checking
        compliance = self.check_compliance(prompt, response, num_tokens)
        metrics.update({
            f'compliance_{k}': v for k, v in compliance.items()
        })
        
        # Log probability (if available in output)
        if 'avg_log_prob' in output:
            metrics['avg_log_prob'] = output['avg_log_prob']
            metrics['total_log_prob'] = output.get('total_log_prob', None)
        
        return metrics
    
    def compute_metrics_batch(self):
        """Compute metrics for all outputs"""
        logger.info("Computing metrics for all outputs...")
        
        for output in tqdm(self.generated_outputs, desc="Computing metrics"):
            try:
                metrics = self.compute_all_metrics(output)
                self.metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to compute metrics for output {output.get('prompt_id')}: {e}")
        
        logger.info(f"Computed metrics for {len(self.metrics)} outputs")
    
    def save_metrics(self, output_path: str):
        """Save metrics to file"""
        logger.info(f"Saving metrics to {output_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics CSV to {output_path}")
        
        # Also save as JSONL
        jsonl_path = Path(output_path).with_suffix('.jsonl')
        with open(jsonl_path, 'w') as f:
            for metric in self.metrics:
                f.write(json.dumps(metric) + '\n')
        logger.info(f"Saved metrics JSONL to {jsonl_path}")
    
    def create_summary(self, output_dir: Path):
        """Create summary statistics"""
        logger.info("Creating summary statistics...")
        
        df = pd.DataFrame(self.metrics)
        
        summary = {
            'num_outputs': len(df),
            'overall_stats': {},
            'stats_by_config': {},
        }
        
        # Overall statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['prompt_id', 'sample_idx']:
                summary['overall_stats'][col] = {
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                }
        
        # Stats by generation config
        for config in df['generation_config'].unique():
            config_df = df[df['generation_config'] == config]
            summary['stats_by_config'][config] = {}
            
            for col in ['length_tokens', 'reward_score', 'kl_divergence', 'perplexity']:
                if col in config_df.columns:
                    summary['stats_by_config'][config][col] = {
                        'mean': float(config_df[col].mean()) if not config_df[col].isna().all() else None,
                        'median': float(config_df[col].median()) if not config_df[col].isna().all() else None,
                        'std': float(config_df[col].std()) if not config_df[col].isna().all() else None,
                    }
        
        # Stats by category (if available)
        if 'category' in df.columns:
            summary['stats_by_category'] = {}
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                summary['stats_by_category'][category] = {
                    'count': len(cat_df),
                    'mean_length': float(cat_df['length_tokens'].mean()),
                    'mean_reward': float(cat_df['reward_score'].mean()) if 'reward_score' in cat_df.columns and not cat_df['reward_score'].isna().all() else None,
                }
        
        # Compliance statistics
        if 'compliance_compliant' in df.columns:
            compliant_df = df[df['compliance_has_constraint'] == True]
            if len(compliant_df) > 0:
                summary['compliance_stats'] = {
                    'num_constrained_prompts': len(compliant_df),
                    'compliance_rate': float(compliant_df['compliance_compliant'].mean()),
                    'mean_deviation': float(compliant_df['compliance_deviation'].mean()),
                }
        
        # Save summary
        summary_path = output_dir / "metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Metrics Summary")
        logger.info("=" * 80)
        logger.info(f"Total outputs: {summary['num_outputs']}")
        
        if 'length_tokens' in summary['overall_stats']:
            stats = summary['overall_stats']['length_tokens']
            logger.info(f"Length: {stats['mean']:.1f} ± {stats['std']:.1f} tokens")
        
        if 'reward_score' in summary['overall_stats']:
            stats = summary['overall_stats']['reward_score']
            if stats['mean'] is not None:
                logger.info(f"Reward: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'kl_divergence' in summary['overall_stats']:
            stats = summary['overall_stats']['kl_divergence']
            if stats['mean'] is not None:
                logger.info(f"KL divergence: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'compliance_stats' in summary:
            comp_stats = summary['compliance_stats']
            logger.info(f"Compliance rate: {comp_stats['compliance_rate']:.2%} ({comp_stats['num_constrained_prompts']} prompts)")
        
        logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('--generated_file', type=str, required=True,
                        help='Path to generated outputs JSONL file')
    parser.add_argument('--reward_model_path', type=str, default=None,
                        help='Path to reward model')
    parser.add_argument('--reference_model_path', type=str, default=None,
                        help='Path to reference model (for KL computation)')
    parser.add_argument('--policy_model_path', type=str, default=None,
                        help='Path to policy model (for perplexity and KL)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save metrics CSV')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get config
    config = get_default_config()
    
    # Create computer
    computer = MetricsComputer(config)
    
    # Load generated outputs
    computer.load_generated_outputs(args.generated_file)
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        generated_path = Path(args.generated_file)
        output_path = generated_path.with_name(generated_path.stem + '_metrics.csv')
    
    output_dir = Path(output_path).parent
    
    # Setup tokenizer (use policy model path or reward model path)
    tokenizer_path = args.policy_model_path or args.reward_model_path or config.base_model.model_name
    computer.setup_tokenizer(tokenizer_path)
    
    # Load models
    if args.reward_model_path:
        computer.load_reward_model(args.reward_model_path)
    
    if args.reference_model_path:
        computer.load_reference_model(args.reference_model_path)
    
    if args.policy_model_path:
        computer.load_policy_model(args.policy_model_path)
    
    # Compute metrics
    computer.compute_metrics_batch()
    
    # Save metrics
    computer.save_metrics(str(output_path))
    
    # Create summary
    computer.create_summary(output_dir)
    
    logger.info("=" * 80)
    logger.info("Metrics Computation Complete!")
    logger.info("=" * 80)
    logger.info(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()