"""
Data Preparation Script for Alignment Methods Assignment

This script:
1. Downloads/loads ORCA DPO Pairs dataset
2. Preprocesses and formats data into canonical structure
3. Splits into train/val/test sets
4. Tokenizes and validates data
5. Generates statistics and histograms
6. Creates instruction-following subset for perplexity evaluation
7. Saves processed datasets in JSONL format
"""

import os
import sys
# Add project root to path for Kaggle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace datasets and transformers
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# Import config
from config.default_config import get_default_config, Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataStatistics:
    """Statistics for a dataset split"""
    num_examples: int
    prompt_lengths: Dict[str, float]  # min, max, mean, median, std
    response_w_lengths: Dict[str, float]
    response_l_lengths: Dict[str, float]
    prompt_token_lengths: Dict[str, float]
    response_w_token_lengths: Dict[str, float]
    response_l_token_lengths: Dict[str, float]


class DataPreparator:
    """Main class for data preparation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.raw_dataset = None
        self.processed_dataset = None
        self.statistics = {}
        
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        logger.info(f"Loading tokenizer: {self.config.base_model.model_name}")
        
        tokenizer_name = (
            self.config.base_model.tokenizer_name 
            if self.config.base_model.tokenizer_name 
            else self.config.base_model.model_name
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.base_model.trust_remote_code,
            padding_side=self.config.base_model.padding_side,
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
    def load_raw_dataset(self):
        """Load raw ORCA DPO Pairs dataset"""
        logger.info(f"Loading dataset: {self.config.data.dataset_name}")
        
        try:
            # Try loading from HuggingFace Hub
            self.raw_dataset = load_dataset(
                self.config.data.dataset_name,
                self.config.data.dataset_config,
                trust_remote_code=True
            )
            logger.info(f"Loaded dataset from HuggingFace Hub")
            
        except Exception as e:
            logger.error(f"Failed to load from HuggingFace Hub: {e}")
            # Try loading from local path
            local_path = self.config.paths.data_raw / "orca_dpo_pairs"
            if local_path.exists():
                logger.info(f"Attempting to load from local path: {local_path}")
                self.raw_dataset = load_dataset(str(local_path))
            else:
                raise ValueError(f"Cannot load dataset from Hub or local path")
        
        # Print dataset info
        logger.info(f"Dataset structure: {self.raw_dataset}")
        if isinstance(self.raw_dataset, DatasetDict):
            for split_name, split_data in self.raw_dataset.items():
                logger.info(f"  {split_name}: {len(split_data)} examples")
        
    def preprocess_example(self, example: Dict) -> Optional[Dict]:
        """
        Preprocess a single example into canonical format
        
        Expected input format (ORCA DPO Pairs):
        - 'prompt' or 'question': the input prompt
        - 'chosen' or 'response_w': preferred response
        - 'rejected' or 'response_l': less-preferred response
        
        Output format:
        - 'id': unique identifier
        - 'prompt': formatted prompt
        - 'response_w': preferred response (chosen)
        - 'response_l': less-preferred response (rejected)
        """
        try:
            # Extract prompt (try different field names)
            prompt = None
            for field in ['prompt', 'question', 'instruction', 'input']:
                if field in example and example[field]:
                    prompt = example[field]
                    break
            
            if prompt is None:
                logger.warning("No prompt field found in example")
                return None
            
            # Extract preferred response
            response_w = None
            for field in ['chosen', 'response_w', 'positive', 'better_response']:
                if field in example and example[field]:
                    response_w = example[field]
                    break
            
            # Extract less-preferred response
            response_l = None
            for field in ['rejected', 'response_l', 'negative', 'worse_response']:
                if field in example and example[field]:
                    response_l = example[field]
                    break
            
            if response_w is None or response_l is None:
                logger.warning("Missing response field(s) in example")
                return None
            
            # Clean and validate
            prompt = str(prompt).strip()
            response_w = str(response_w).strip()
            response_l = str(response_l).strip()
            
            # Apply length filters
            if len(prompt) < self.config.data.min_prompt_length:
                return None
            if len(response_w) < self.config.data.min_response_length:
                return None
            if len(response_l) < self.config.data.min_response_length:
                return None
            
            # Create canonical format
            processed = {
                'id': example.get('id', None),
                'prompt': prompt,
                'response_w': response_w,
                'response_l': response_l,
            }
            
            # Add metadata if available
            if 'source' in example:
                processed['source'] = example['source']
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error preprocessing example: {e}")
            return None
    
    def format_prompt_response(self, prompt: str, response: str) -> str:
        """Format prompt and response according to template"""
        formatted_prompt = self.config.data.prompt_template.format(prompt=prompt)
        formatted_response = self.config.data.response_template.format(response=response)
        return formatted_prompt + formatted_response
    
    def tokenize_example(self, example: Dict) -> Dict:
        """Tokenize an example and add token counts"""
        
        # Format full sequences
        full_w = self.format_prompt_response(example['prompt'], example['response_w'])
        full_l = self.format_prompt_response(example['prompt'], example['response_l'])
        
        # Tokenize
        tokens_w = self.tokenizer(
            full_w,
            truncation=True,
            max_length=self.config.base_model.max_length,
            return_attention_mask=False,
            add_special_tokens=True
        )
        
        tokens_l = self.tokenizer(
            full_l,
            truncation=True,
            max_length=self.config.base_model.max_length,
            return_attention_mask=False,
            add_special_tokens=True
        )
        
        tokens_prompt = self.tokenizer(
            self.config.data.prompt_template.format(prompt=example['prompt']),
            return_attention_mask=False,
            add_special_tokens=True
        )
        
        # Add token counts
        example['token_count_prompt'] = len(tokens_prompt['input_ids'])
        example['token_count_w'] = len(tokens_w['input_ids'])
        example['token_count_l'] = len(tokens_l['input_ids'])
        
        # Add character counts
        example['char_count_prompt'] = len(example['prompt'])
        example['char_count_w'] = len(example['response_w'])
        example['char_count_l'] = len(example['response_l'])
        
        # Check if truncated
        example['truncated_w'] = len(tokens_w['input_ids']) >= self.config.base_model.max_length
        example['truncated_l'] = len(tokens_l['input_ids']) >= self.config.base_model.max_length
        
        return example
    
    def process_dataset(self):
        """Process entire dataset"""
        logger.info("Processing dataset...")
        
        # Get the appropriate split
        if isinstance(self.raw_dataset, DatasetDict):
            if self.config.data.train_split in self.raw_dataset:
                dataset = self.raw_dataset[self.config.data.train_split]
            else:
                # Use first available split
                split_name = list(self.raw_dataset.keys())[0]
                dataset = self.raw_dataset[split_name]
                logger.info(f"Using split: {split_name}")
        else:
            dataset = self.raw_dataset
        
        # Preprocess all examples
        processed_examples = []
        skipped = 0
        
        for idx, example in enumerate(tqdm(dataset, desc="Preprocessing")):
            # Add ID if missing
            if 'id' not in example or example['id'] is None:
                example['id'] = f"example_{idx}"
            
            processed = self.preprocess_example(example)
            if processed is not None:
                processed_examples.append(processed)
            else:
                skipped += 1
        
        logger.info(f"Processed {len(processed_examples)} examples, skipped {skipped}")
        
        # Convert to Dataset
        self.processed_dataset = Dataset.from_list(processed_examples)
        
        # Tokenize
        logger.info("Tokenizing examples...")
        self.processed_dataset = self.processed_dataset.map(
            self.tokenize_example,
            desc="Tokenizing",
            num_proc=0
        )
        
        logger.info(f"Final dataset size: {len(self.processed_dataset)}")
    
    def create_splits(self) -> DatasetDict:
        """Split dataset into train/val/test"""
        logger.info("Creating train/val/test splits...")
        
        # Set seed for reproducibility
        np.random.seed(self.config.data.seed)
        
        total_size = len(self.processed_dataset)
        
        # Calculate split sizes
        train_size = int(total_size * self.config.data.train_ratio)
        val_size = int(total_size * self.config.data.val_ratio)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create splits
        splits = self.processed_dataset.train_test_split(
            test_size=val_size + test_size,
            seed=self.config.data.seed
        )
        
        train_dataset = splits['train']
        temp_dataset = splits['test']
        
        # Further split into val and test
        val_test_splits = temp_dataset.train_test_split(
            test_size=test_size / (val_size + test_size),
            seed=self.config.data.seed
        )
        
        val_dataset = val_test_splits['train']
        test_dataset = val_test_splits['test']
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        })
        
        # Add split labels
        for split_name, split_data in dataset_dict.items():
            split_data = split_data.map(
                lambda x: {'split': split_name},
                desc=f"Adding split label to {split_name}"
            )
            dataset_dict[split_name] = split_data
        
        return dataset_dict
    
    def compute_statistics(self, dataset_dict: DatasetDict):
        """Compute statistics for each split"""
        logger.info("Computing statistics...")
        
        for split_name, split_data in dataset_dict.items():
            logger.info(f"Computing statistics for {split_name}...")
            
            # Extract lengths
            prompt_chars = [ex['char_count_prompt'] for ex in split_data]
            response_w_chars = [ex['char_count_w'] for ex in split_data]
            response_l_chars = [ex['char_count_l'] for ex in split_data]
            
            prompt_tokens = [ex['token_count_prompt'] for ex in split_data]
            response_w_tokens = [ex['token_count_w'] for ex in split_data]
            response_l_tokens = [ex['token_count_l'] for ex in split_data]
            
            # Compute statistics
            stats = DataStatistics(
                num_examples=len(split_data),
                prompt_lengths={
                    'min': float(np.min(prompt_chars)),
                    'max': float(np.max(prompt_chars)),
                    'mean': float(np.mean(prompt_chars)),
                    'median': float(np.median(prompt_chars)),
                    'std': float(np.std(prompt_chars))
                },
                response_w_lengths={
                    'min': float(np.min(response_w_chars)),
                    'max': float(np.max(response_w_chars)),
                    'mean': float(np.mean(response_w_chars)),
                    'median': float(np.median(response_w_chars)),
                    'std': float(np.std(response_w_chars))
                },
                response_l_lengths={
                    'min': float(np.min(response_l_chars)),
                    'max': float(np.max(response_l_chars)),
                    'mean': float(np.mean(response_l_chars)),
                    'median': float(np.median(response_l_chars)),
                    'std': float(np.std(response_l_chars))
                },
                prompt_token_lengths={
                    'min': float(np.min(prompt_tokens)),
                    'max': float(np.max(prompt_tokens)),
                    'mean': float(np.mean(prompt_tokens)),
                    'median': float(np.median(prompt_tokens)),
                    'std': float(np.std(prompt_tokens))
                },
                response_w_token_lengths={
                    'min': float(np.min(response_w_tokens)),
                    'max': float(np.max(response_w_tokens)),
                    'mean': float(np.mean(response_w_tokens)),
                    'median': float(np.median(response_w_tokens)),
                    'std': float(np.std(response_w_tokens))
                },
                response_l_token_lengths={
                    'min': float(np.min(response_l_tokens)),
                    'max': float(np.max(response_l_tokens)),
                    'mean': float(np.mean(response_l_tokens)),
                    'median': float(np.median(response_l_tokens)),
                    'std': float(np.std(response_l_tokens))
                }
            )
            
            self.statistics[split_name] = stats
            
            # Print summary
            logger.info(f"\n{split_name.upper()} Statistics:")
            logger.info(f"  Examples: {stats.num_examples}")
            logger.info(f"  Prompt tokens: {stats.prompt_token_lengths['mean']:.1f} ± {stats.prompt_token_lengths['std']:.1f}")
            logger.info(f"  Response_W tokens: {stats.response_w_token_lengths['mean']:.1f} ± {stats.response_w_token_lengths['std']:.1f}")
            logger.info(f"  Response_L tokens: {stats.response_l_token_lengths['mean']:.1f} ± {stats.response_l_token_lengths['std']:.1f}")
    
    def save_datasets(self, dataset_dict: DatasetDict):
        """Save processed datasets to JSONL files"""
        logger.info("Saving datasets...")
        
        output_dir = self.config.paths.data_processed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset_dict.items():
            output_path = output_dir / f"{split_name}.jsonl"
            
            logger.info(f"Saving {split_name} to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in tqdm(split_data, desc=f"Saving {split_name}"):
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(split_data)} examples to {output_path}")
    
    def save_statistics(self):
        """Save statistics to JSON and CSV"""
        logger.info("Saving statistics...")
        
        output_dir = self.config.paths.data_processed
        
        # Save as JSON
        stats_json_path = output_dir / "statistics.json"
        with open(stats_json_path, 'w') as f:
            stats_dict = {k: asdict(v) for k, v in self.statistics.items()}
            json.dump(stats_dict, f, indent=2)
        logger.info(f"Saved statistics to {stats_json_path}")
        
        # Save as CSV for easy viewing
        stats_csv_path = output_dir / "statistics.csv"
        rows = []
        for split_name, stats in self.statistics.items():
            row = {'split': split_name, 'num_examples': stats.num_examples}
            
            # Add all metrics
            for metric_name, metric_dict in [
                ('prompt_tokens', stats.prompt_token_lengths),
                ('response_w_tokens', stats.response_w_token_lengths),
                ('response_l_tokens', stats.response_l_token_lengths),
                ('prompt_chars', stats.prompt_lengths),
                ('response_w_chars', stats.response_w_lengths),
                ('response_l_chars', stats.response_l_lengths)
            ]:
                for stat_name, value in metric_dict.items():
                    row[f"{metric_name}_{stat_name}"] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(stats_csv_path, index=False)
        logger.info(f"Saved statistics CSV to {stats_csv_path}")
    
    def plot_length_distributions(self, dataset_dict: DatasetDict):
        """Plot token length distributions"""
        logger.info("Generating length distribution plots...")
        
        output_dir = self.config.paths.data_processed / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for all splits
        data_for_plot = defaultdict(lambda: defaultdict(list))
        
        for split_name, split_data in dataset_dict.items():
            data_for_plot[split_name]['prompt'] = [ex['token_count_prompt'] for ex in split_data]
            data_for_plot[split_name]['response_w'] = [ex['token_count_w'] for ex in split_data]
            data_for_plot[split_name]['response_l'] = [ex['token_count_l'] for ex in split_data]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Token Length Distributions', fontsize=16)
        
        colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
        
        # Plot prompts
        for split_name, data in data_for_plot.items():
            axes[0].hist(data['prompt'], bins=50, alpha=0.5, 
                        label=split_name, color=colors.get(split_name, 'gray'))
        axes[0].set_xlabel('Token Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prompt Lengths')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot response_w
        for split_name, data in data_for_plot.items():
            axes[1].hist(data['response_w'], bins=50, alpha=0.5,
                        label=split_name, color=colors.get(split_name, 'gray'))
        axes[1].set_xlabel('Token Count')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Preferred Response (yW) Lengths')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot response_l
        for split_name, data in data_for_plot.items():
            axes[2].hist(data['response_l'], bins=50, alpha=0.5,
                        label=split_name, color=colors.get(split_name, 'gray'))
        axes[2].set_xlabel('Token Count')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Less-Preferred Response (yL) Lengths')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / "token_length_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {plot_path}")
        plt.close()
    
    def create_instruction_following_subset(self, dataset_dict: DatasetDict):
        """Create small instruction-following subset for perplexity evaluation"""
        logger.info("Creating instruction-following subset for perplexity...")
        
        # Use validation set
        val_data = dataset_dict['val']
        
        # Sample subset
        subset_size = min(self.config.data.instr_following_size, len(val_data))
        
        indices = np.random.RandomState(self.config.data.seed).choice(
            len(val_data), size=subset_size, replace=False
        )
        
        subset = val_data.select(indices)
        
        # Save as JSONL
        output_path = self.config.paths.data_processed / "instr_following_subset.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in subset:
                # For perplexity, we only need prompt and preferred response
                entry = {
                    'id': example['id'],
                    'prompt': example['prompt'],
                    'response': example['response_w'],  # Use preferred response
                    'token_count': example['token_count_w']
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(subset)} examples to {output_path}")
    
    def run(self):
        """Run complete data preparation pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Data Preparation Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Setup tokenizer
        self.setup_tokenizer()
        
        # Step 2: Load raw dataset
        self.load_raw_dataset()
        
        # Step 3: Process dataset
        self.process_dataset()
        
        # Step 4: Create splits
        dataset_dict = self.create_splits()
        
        # Step 5: Compute statistics
        self.compute_statistics(dataset_dict)
        
        # Step 6: Save datasets
        self.save_datasets(dataset_dict)
        
        # Step 7: Save statistics
        self.save_statistics()
        
        # Step 8: Plot distributions
        self.plot_length_distributions(dataset_dict)
        
        # Step 9: Create instruction-following subset
        self.create_instruction_following_subset(dataset_dict)
        
        logger.info("=" * 80)
        logger.info("Data Preparation Complete!")
        logger.info("=" * 80)
        logger.info(f"\nProcessed data saved to: {self.config.paths.data_processed}")
        logger.info(f"Train examples: {len(dataset_dict['train'])}")
        logger.info(f"Val examples: {len(dataset_dict['val'])}")
        logger.info(f"Test examples: {len(dataset_dict['test'])}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare ORCA DPO Pairs dataset")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file (optional, uses defaults if not provided)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading config from {args.config}")
        config = Config.load(args.config)
    else:
        logger.info("Using default config")
        config = get_default_config()
    
    # Override seed if provided
    if args.seed is not None:
        config.data.seed = args.seed
        logger.info(f"Using seed: {args.seed}")
    
    # Run preparation
    preparator = DataPreparator(config)
    preparator.run()


if __name__ == "__main__":
    main()
