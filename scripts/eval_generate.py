"""
Evaluation Generation Script

This script generates responses from trained models on a test set of prompts.
Supports multiple generation strategies (greedy, top-k, temperature variants).

Usage:
    python eval_generate.py \
      --model_path ./checkpoints/dpo/final_model \
      --test_file ./eval/testset_50.jsonl \
      --output_file ./eval/dpo_outputs.jsonl \
      --temperature 0.7 \
      --num_samples 1 \
      --seed 42
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
from tqdm import tqdm

# Transformers and HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    GenerationConfig
)
from peft import PeftModel

# Import config
from config.default_config import (
    parse_eval_args,
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


class ModelEvaluator:
    """Main class for generating model outputs"""
    
    def __init__(self, args, config: Config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.test_prompts = None
        self.outputs = []
        
    def setup_paths(self):
        """Setup output directory"""
        if self.args.output_file:
            self.output_path = Path(self.args.output_file)
            self.output_dir = self.output_path.parent
        else:
            # Auto-generate output path
            model_name = Path(self.args.model_path).name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("eval") / model_name
            self.output_path = self.output_dir / f"outputs_{timestamp}.jsonl"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Output file: {self.output_path}")
    
    def save_config(self):
        """Save arguments"""
        args_path = self.output_dir / "eval_args.json"
        save_args_to_json(self.args, str(args_path))
        logger.info(f"Saved args to {args_path}")
    
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        logger.info(f"Loading tokenizer from: {self.args.model_path}")
        
        try:
            # Try loading from model path
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_path,
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except:
            # Fall back to base model
            logger.warning(f"Could not load tokenizer from {self.args.model_path}, using base model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        logger.info(f"Tokenizer loaded")
    
    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from: {self.args.model_path}")
        
        model_path = Path(self.args.model_path)
        
        # Check if it's a PEFT/LoRA model
        adapter_config_path = model_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            logger.info("Detected LoRA adapter, loading with PEFT")
            
            # Load adapter config to get base model
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get('base_model_name_or_path', self.args.model_name)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                load_in_8bit=self.args.load_in_8bit,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
                torch_dtype=torch.float16 if self.args.mixed_precision == "fp16" else "auto",
            )
            
            # Load adapter
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
            logger.info("Loaded model with LoRA adapter")
            
        else:
            # Load full model
            logger.info("Loading full model")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                load_in_8bit=self.args.load_in_8bit,
                device_map="auto",
                trust_remote_code=self.config.base_model.trust_remote_code,
                torch_dtype=torch.float16 if self.args.mixed_precision == "fp16" else "auto",
            )
        
        self.model.eval()
        logger.info("Model loaded and set to eval mode")
    
    def load_test_prompts(self):
        """Load test prompts"""
        logger.info(f"Loading test prompts from: {self.args.test_file}")
        
        test_path = Path(self.args.test_file)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Load prompts
        self.test_prompts = []
        with open(test_path, 'r') as f:
            for line in f:
                self.test_prompts.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.test_prompts)} test prompts")
        
        # Log category distribution if available
        if 'category' in self.test_prompts[0]:
            categories = {}
            for item in self.test_prompts:
                cat = item.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info("Category distribution:")
            for cat, count in sorted(categories.items()):
                logger.info(f"  {cat}: {count}")
    
    def generate_response(
        self,
        prompt: str,
        generation_config: Dict
    ) -> Tuple[str, Dict]:
        """Generate a single response for a prompt"""
        
        # Format prompt
        formatted_prompt = self.config.data.prompt_template.format(prompt=prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.args.max_length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode response (remove prompt)
        generated_ids = outputs.sequences[0, inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute statistics
        stats = {
            'num_tokens': len(generated_ids),
            'num_chars': len(response),
        }
        
        # Compute average log probability if scores available
        if hasattr(outputs, 'scores') and outputs.scores:
            log_probs = []
            for i, (score, token_id) in enumerate(zip(outputs.scores, generated_ids)):
                token_log_prob = torch.log_softmax(score, dim=-1)[0, token_id].item()
                log_probs.append(token_log_prob)
            
            stats['avg_log_prob'] = np.mean(log_probs)
            stats['total_log_prob'] = np.sum(log_probs)
        
        return response, stats
    
    def generate_all(self):
        """Generate responses for all test prompts"""
        logger.info("Generating responses...")
        
        # Set seed
        set_seed(self.args.seed)
        
        # Define generation configs
        generation_configs = [
            {
                'name': 'greedy',
                'config': {
                    'max_new_tokens': 256,
                    'do_sample': False,
                }
            },
            {
                'name': f'temp_{self.args.temperature}',
                'config': {
                    'max_new_tokens': 256,
                    'do_sample': True,
                    'temperature': self.args.temperature,
                    'top_k': self.args.top_k,
                    'top_p': self.args.top_p,
                }
            },
        ]
        
        # Add additional temperature variants if specified
        if self.args.temperature != 0.2:
            generation_configs.append({
                'name': 'temp_0.2',
                'config': {
                    'max_new_tokens': 256,
                    'do_sample': True,
                    'temperature': 0.2,
                    'top_k': self.args.top_k,
                    'top_p': self.args.top_p,
                }
            })
        
        if self.args.temperature != 1.0:
            generation_configs.append({
                'name': 'temp_1.0',
                'config': {
                    'max_new_tokens': 256,
                    'do_sample': True,
                    'temperature': 1.0,
                    'top_k': self.args.top_k,
                    'top_p': self.args.top_p,
                }
            })
        
        # Add top-k variant
        generation_configs.append({
            'name': 'top_k_5',
            'config': {
                'max_new_tokens': 256,
                'do_sample': True,
                'top_k': 5,
                'temperature': 0.7,
            }
        })
        
        # Generate for each prompt and config
        total_generations = len(self.test_prompts) * len(generation_configs) * self.args.num_samples
        
        with tqdm(total=total_generations, desc="Generating") as pbar:
            for prompt_item in self.test_prompts:
                prompt_id = prompt_item.get('id', len(self.outputs))
                prompt = prompt_item['prompt']
                
                for gen_config in generation_configs:
                    for sample_idx in range(self.args.num_samples):
                        try:
                            # Generate response
                            response, stats = self.generate_response(
                                prompt,
                                gen_config['config']
                            )
                            
                            # Store output
                            output = {
                                'prompt_id': prompt_id,
                                'prompt': prompt,
                                'response': response,
                                'generation_config': gen_config['name'],
                                'sample_idx': sample_idx,
                                'num_tokens': stats['num_tokens'],
                                'num_chars': stats['num_chars'],
                            }
                            
                            # Add optional fields if available
                            if 'avg_log_prob' in stats:
                                output['avg_log_prob'] = stats['avg_log_prob']
                                output['total_log_prob'] = stats['total_log_prob']
                            
                            if 'category' in prompt_item:
                                output['category'] = prompt_item['category']
                            
                            if 'expected_behavior' in prompt_item:
                                output['expected_behavior'] = prompt_item['expected_behavior']
                            
                            self.outputs.append(output)
                            
                        except Exception as e:
                            logger.warning(f"Generation failed for prompt {prompt_id}: {e}")
                        
                        pbar.update(1)
        
        logger.info(f"Generated {len(self.outputs)} responses")
    
    def save_outputs(self):
        """Save generated outputs"""
        logger.info(f"Saving outputs to {self.output_path}")
        
        # Save as JSONL
        with open(self.output_path, 'w') as f:
            for output in self.outputs:
                f.write(json.dumps(output) + '\n')
        
        logger.info(f"Saved {len(self.outputs)} outputs")
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame(self.outputs)
        csv_path = self.output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Also saved as CSV: {csv_path}")
        
        # Create summary statistics
        self.create_summary()
    
    def create_summary(self):
        """Create summary statistics"""
        logger.info("Creating summary statistics...")
        
        df = pd.DataFrame(self.outputs)
        
        # Overall statistics
        summary = {
            'model_path': str(self.args.model_path),
            'test_file': str(self.args.test_file),
            'num_prompts': len(self.test_prompts),
            'num_outputs': len(self.outputs),
            'generation_configs': df['generation_config'].unique().tolist(),
            'overall_stats': {
                'mean_tokens': float(df['num_tokens'].mean()),
                'median_tokens': float(df['num_tokens'].median()),
                'std_tokens': float(df['num_tokens'].std()),
                'min_tokens': int(df['num_tokens'].min()),
                'max_tokens': int(df['num_tokens'].max()),
                'mean_chars': float(df['num_chars'].mean()),
            }
        }
        
        # Stats by generation config
        summary['stats_by_config'] = {}
        for config_name in df['generation_config'].unique():
            config_df = df[df['generation_config'] == config_name]
            summary['stats_by_config'][config_name] = {
                'mean_tokens': float(config_df['num_tokens'].mean()),
                'median_tokens': float(config_df['num_tokens'].median()),
                'std_tokens': float(config_df['num_tokens'].std()),
            }
        
        # Stats by category if available
        if 'category' in df.columns:
            summary['stats_by_category'] = {}
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                summary['stats_by_category'][category] = {
                    'count': len(cat_df),
                    'mean_tokens': float(cat_df['num_tokens'].mean()),
                    'median_tokens': float(cat_df['num_tokens'].median()),
                    'std_tokens': float(cat_df['num_tokens'].std()),
                }
        
        # Save summary
        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Generation Summary")
        logger.info("=" * 80)
        logger.info(f"Total outputs: {summary['num_outputs']}")
        logger.info(f"Mean tokens: {summary['overall_stats']['mean_tokens']:.1f}")
        logger.info(f"Median tokens: {summary['overall_stats']['median_tokens']:.1f}")
        logger.info(f"Token range: [{summary['overall_stats']['min_tokens']}, {summary['overall_stats']['max_tokens']}]")
        
        if 'stats_by_category' in summary:
            logger.info("\nStats by category:")
            for cat, stats in summary['stats_by_category'].items():
                logger.info(f"  {cat}: {stats['mean_tokens']:.1f} tokens (n={stats['count']})")
        
        logger.info("=" * 80)
    
    def run(self):
        """Run complete evaluation generation pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Evaluation Generation")
        logger.info("=" * 80)
        
        try:
            # Save config
            self.save_config()
            
            # Setup tokenizer
            self.setup_tokenizer()
            
            # Load model
            self.load_model()
            
            # Load test prompts
            self.load_test_prompts()
            
            # Generate responses
            self.generate_all()
            
            # Save outputs
            self.save_outputs()
            
            logger.info("=" * 80)
            logger.info("Evaluation Generation Complete!")
            logger.info("=" * 80)
            logger.info(f"Outputs saved to: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_eval_args()
    
    # Get config
    config = get_config_from_args(args)
    
    # Create evaluator
    evaluator = ModelEvaluator(args, config)
    
    # Run generation
    evaluator.run()


if __name__ == "__main__":
    main()