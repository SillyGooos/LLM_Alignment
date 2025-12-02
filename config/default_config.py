"""
Default Configuration for Alignment Methods Assignment
Adapted for Kaggle Notebook Environment

Directory Structure:
/kaggle/working/LLM_Alignment/
├── config/
├── scripts/
├── data/
├── models/
└── eval/

Usage:
    python scripts/train_reward_model.py --batch_size 16 --lr 5e-5 --epochs 3 --seed 42
    python scripts/train_dpo.py --method DPO --lr 1e-4 --lora_r 8 --epochs 3 --seed 42
    python scripts/train_ppo.py --method PPO --reward_mode sparse --kl_coef 0.05 --seed 42
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import json

# =============================================================================
# PROJECT PATHS (Kaggle Environment)
# =============================================================================

@dataclass
class PathConfig:
    """Directory structure for the project (Kaggle-adapted)"""
    
    # Root directory - detect if running on Kaggle or locally
    project_root: Path = field(default_factory=lambda: Path.cwd())
    
    # Data directories (relative to project root)
    data_root: Path = field(default_factory=lambda: Path("data"))
    data_raw: Path = field(default_factory=lambda: Path("data/raw"))
    data_processed: Path = field(default_factory=lambda: Path("data/processed"))
    
    # Model checkpoints (relative to project root)
    models_root: Path = field(default_factory=lambda: Path("models"))
    models_baseline: Path = field(default_factory=lambda: Path("models/baseline"))
    models_dpo: Path = field(default_factory=lambda: Path("models/dpo"))
    models_ppo: Path = field(default_factory=lambda: Path("models/ppo"))
    models_grpo: Path = field(default_factory=lambda: Path("models/grpo"))
    models_reward: Path = field(default_factory=lambda: Path("models/reward_model"))
    
    # Evaluation directories (relative to project root)
    eval_root: Path = field(default_factory=lambda: Path("eval"))
    
    # Logs and checkpoints (relative to project root)
    logs_root: Path = field(default_factory=lambda: Path("logs"))
    checkpoints_root: Path = field(default_factory=lambda: Path("checkpoints"))
    
    def __post_init__(self):
        """Resolve all paths relative to project root"""
        # Make all paths absolute relative to project root
        for field_name in self.__dataclass_fields__:
            if field_name != 'project_root':
                path = getattr(self, field_name)
                if isinstance(path, Path):
                    setattr(self, field_name, self.project_root / path)
    
    def create_dirs(self):
        """Create all necessary directories"""
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if isinstance(path, Path) and field_name != 'project_root':
                path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    
    # Dataset
    dataset_name: str = "Intel/orca_dpo_pairs"
    dataset_subset: Optional[str] = None
    
    # Splits
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    
    # Sampling
    max_samples: Optional[int] = None  # None = use all data
    max_train_samples: Optional[int] = 50000  # Limit for faster training
    max_val_samples: Optional[int] = 5000
    max_test_samples: Optional[int] = 5000
    
    # Instruction following subset
    instr_following_size: int = 10000  # For dense reward PPO
    
    # Text processing
    max_length: int = 512
    truncation: bool = True
    
    # Seed
    seed: int = 42
    
    # Template formats
    prompt_template: str = "Question: {question}\n\nAnswer:"
    response_template: str = " {response}"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for base model"""
    
    # Base model
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer_name: Optional[str] = None  # Use model_name if None
    
    # Model settings
    use_auth_token: bool = False
    trust_remote_code: bool = True
    
    # Memory optimization
    load_in_8bit: bool = True  # Use 8-bit quantization
    device_map: str = "auto"
    torch_dtype: str = "float16"  # Options: float32, float16, bfloat16
    
    # Generation
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

# =============================================================================
# REWARD MODEL CONFIGURATION
# =============================================================================

@dataclass
class RewardModelConfig:
    """Configuration for reward model training"""
    
    # Base model
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_accuracy"
    load_best_model_at_end: bool = True
    
    # Seed
    seed: int = 42

# =============================================================================
# DPO CONFIGURATION
# =============================================================================

@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization"""
    
    # Method
    method: str = "DPO"
    
    # Base model
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # DPO hyperparameters
    beta: float = 0.1  # Temperature parameter for DPO loss
    reference_free: bool = False  # Use reference-free variant
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    
    # Seed
    seed: int = 42

# =============================================================================
# PPO CONFIGURATION
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for Proximal Policy Optimization"""
    
    # Method
    method: str = "PPO"
    reward_mode: str = "sparse"  # 'sparse' or 'dense'
    
    # Base models
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_path: str = "models/reward_model/final_model"
    
    # PPO hyperparameters
    kl_coef: float = 0.05  # KL divergence coefficient
    clip_range: float = 0.2  # PPO clipping parameter
    vf_coef: float = 0.5  # Value function coefficient
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Generation settings for PPO
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Seed
    seed: int = 42

# =============================================================================
# GRPO CONFIGURATION
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization"""
    
    # Method
    method: str = "GRPO"
    
    # Base models
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_path: str = "models/reward_model/final_model"
    
    # GRPO hyperparameters
    group_size: int = 8  # Number of responses per prompt (K)
    advantage_normalization: str = "rank"  # 'rank', 'zscore', or 'minmax'
    use_baseline: bool = False  # Use baseline for advantage estimation
    temperature: float = 0.7  # Sampling temperature
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_new_tokens: int = 256
    top_k: int = 50
    top_p: float = 0.95
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Seed
    seed: int = 42

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline"""
    
    # Test set
    test_file: str = "eval/testset_50.jsonl"
    
    # Generation configs to test
    generation_configs: List[str] = field(default_factory=lambda: [
        "greedy",
        "temp_0.7",
        "temp_0.2",
        "temp_1.0",
        "top_k_5"
    ])
    
    # Number of samples per prompt
    num_samples: int = 1
    
    # Models to evaluate
    models_to_eval: List[str] = field(default_factory=lambda: [
        "baseline",
        "dpo",
        "ppo_sparse",
        "ppo_dense",
        "grpo"
    ])
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "reward",
        "kl_divergence",
        "perplexity",
        "length",
        "compliance"
    ])
    
    # Reward hacking tests
    perturbations: List[str] = field(default_factory=lambda: [
        "filler_phrases",
        "sentence_reorder",
        "synonym_replace",
        "formatting",
        "alignment_keywords"
    ])
    
    # Seed
    seed: int = 42

# =============================================================================
# ARGPARSE BUILDERS
# =============================================================================

def get_data_preprocessing_parser() -> argparse.ArgumentParser:
    """Get argument parser for data preprocessing"""
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    
    config = DataConfig()
    paths = PathConfig()
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default=config.dataset_name)
    parser.add_argument("--max_samples", type=int, default=config.max_samples)
    parser.add_argument("--max_train_samples", type=int, default=config.max_train_samples)
    parser.add_argument("--max_val_samples", type=int, default=config.max_val_samples)
    parser.add_argument("--max_test_samples", type=int, default=config.max_test_samples)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # Path arguments
    parser.add_argument("--output_dir", type=str, default=str(paths.data_processed))
    
    return parser

def get_reward_model_parser() -> argparse.ArgumentParser:
    """Get argument parser for reward model training"""
    parser = argparse.ArgumentParser(description="Reward Model Training")
    
    config = RewardModelConfig()
    paths = PathConfig()
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default=config.base_model)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=config.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=config.use_lora)
    parser.add_argument("--lora_r", type=int, default=config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=config.lora_dropout)
    
    # Path arguments
    parser.add_argument("--train_file", type=str, default=str(paths.data_processed / "train.jsonl"))
    parser.add_argument("--val_file", type=str, default=str(paths.data_processed / "val.jsonl"))
    parser.add_argument("--save_dir", type=str, default=str(paths.models_reward))
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=config.eval_steps)
    parser.add_argument("--save_steps", type=int, default=config.save_steps)
    
    return parser

def get_dpo_parser() -> argparse.ArgumentParser:
    """Get argument parser for DPO training"""
    parser = argparse.ArgumentParser(description="DPO Training")
    
    config = DPOConfig()
    paths = PathConfig()
    
    # Method
    parser.add_argument("--method", type=str, default=config.method, choices=["DPO"])
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default=config.base_model)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=config.beta)
    parser.add_argument("--reference_free", action="store_true", default=config.reference_free)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=config.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=config.use_lora)
    parser.add_argument("--lora_r", type=int, default=config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=config.lora_dropout)
    
    # Path arguments
    parser.add_argument("--train_file", type=str, default=str(paths.data_processed / "train.jsonl"))
    parser.add_argument("--val_file", type=str, default=str(paths.data_processed / "val.jsonl"))
    parser.add_argument("--save_dir", type=str, default=str(paths.checkpoints_root / "dpo"))
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=config.eval_steps)
    parser.add_argument("--save_steps", type=int, default=config.save_steps)
    
    return parser

def get_ppo_parser() -> argparse.ArgumentParser:
    """Get argument parser for PPO training"""
    parser = argparse.ArgumentParser(description="PPO Training")
    
    config = PPOConfig()
    paths = PathConfig()
    
    # Method
    parser.add_argument("--method", type=str, default=config.method, choices=["PPO"])
    parser.add_argument("--reward_mode", type=str, default=config.reward_mode, 
                       choices=["sparse", "dense"])
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default=config.base_model)
    parser.add_argument("--reward_model_path", type=str, default=config.reward_model_path)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # PPO hyperparameters
    parser.add_argument("--kl_coef", type=float, default=config.kl_coef)
    parser.add_argument("--clip_range", type=float, default=config.clip_range)
    parser.add_argument("--vf_coef", type=float, default=config.vf_coef)
    parser.add_argument("--gamma", type=float, default=config.gamma)
    parser.add_argument("--lam", type=float, default=config.lam)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--mini_batch_size", type=int, default=config.mini_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup_steps", type=int, default=config.warmup_steps)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=config.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=config.temperature)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=config.use_lora)
    parser.add_argument("--lora_r", type=int, default=config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=config.lora_dropout)
    
    # Path arguments
    parser.add_argument("--train_file", type=str, default=str(paths.data_processed / "train.jsonl"))
    parser.add_argument("--save_dir", type=str, default=str(paths.checkpoints_root / "ppo"))
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps)
    parser.add_argument("--save_steps", type=int, default=config.save_steps)
    
    return parser

def get_grpo_parser() -> argparse.ArgumentParser:
    """Get argument parser for GRPO training"""
    parser = argparse.ArgumentParser(description="GRPO Training")
    
    config = GRPOConfig()
    paths = PathConfig()
    
    # Method
    parser.add_argument("--method", type=str, default=config.method, choices=["GRPO"])
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default=config.base_model)
    parser.add_argument("--reward_model_path", type=str, default=config.reward_model_path)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # GRPO hyperparameters
    parser.add_argument("--group_size", type=int, default=config.group_size)
    parser.add_argument("--advantage_normalization", type=str, default=config.advantage_normalization,
                       choices=["rank", "zscore", "minmax"])
    parser.add_argument("--use_baseline", action="store_true", default=config.use_baseline)
    parser.add_argument("--temperature", type=float, default=config.temperature)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=config.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=config.max_new_tokens)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=config.use_lora)
    parser.add_argument("--lora_r", type=int, default=config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=config.lora_dropout)
    
    # Path arguments
    parser.add_argument("--train_file", type=str, default=str(paths.data_processed / "train.jsonl"))
    parser.add_argument("--save_dir", type=str, default=str(paths.checkpoints_root / "grpo"))
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps)
    parser.add_argument("--save_steps", type=int, default=config.save_steps)
    
    return parser

def get_eval_generation_parser() -> argparse.ArgumentParser:
    """Get argument parser for evaluation generation"""
    parser = argparse.ArgumentParser(description="Evaluation Generation")
    
    config = EvalConfig()
    model_config = ModelConfig()
    paths = PathConfig()
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=model_config.temperature)
    parser.add_argument("--top_k", type=int, default=model_config.top_k)
    parser.add_argument("--top_p", type=float, default=model_config.top_p)
    parser.add_argument("--max_new_tokens", type=int, default=model_config.max_new_tokens)
    parser.add_argument("--num_samples", type=int, default=config.num_samples)
    
    # Path arguments
    parser.add_argument("--test_file", type=str, default=config.test_file)
    parser.add_argument("--output_file", type=str, required=True)
    
    return parser

def get_eval_metrics_parser() -> argparse.ArgumentParser:
    """Get argument parser for evaluation metrics"""
    parser = argparse.ArgumentParser(description="Evaluation Metrics")
    
    config = EvalConfig()
    paths = PathConfig()
    
    # Model arguments
    parser.add_argument("--generated_file", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--reference_model_path", type=str, required=True)
    parser.add_argument("--policy_model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # Path arguments
    parser.add_argument("--output_file", type=str, required=True)
    
    return parser

def get_perturbations_parser() -> argparse.ArgumentParser:
    """Get argument parser for reward hacking tests"""
    parser = argparse.ArgumentParser(description="Reward Hacking Tests")
    
    config = EvalConfig()
    paths = PathConfig()
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=config.seed)
    
    # Path arguments
    parser.add_argument("--test_file", type=str, default=config.test_file)
    parser.add_argument("--output_file", type=str, required=True)
    
    return parser

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_config(config: Any, save_path: str):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if hasattr(config, '__dataclass_fields__'):
        config_dict = asdict(config)
    else:
        config_dict = vars(config)
    
    # Convert Path objects to strings
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(load_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(load_path, 'r') as f:
        return json.load(f)

# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Testing configuration...")
    
    # Test path configuration
    paths = PathConfig()
    print(f"\nProject root: {paths.project_root}")
    print(f"Data processed: {paths.data_processed}")
    print(f"Models reward: {paths.models_reward}")
    
    # Create directories
    paths.create_dirs()
    print("\n✓ All directories created")
    
    # Test configurations
    data_config = DataConfig()
    model_config = ModelConfig()
    reward_config = RewardModelConfig()
    dpo_config = DPOConfig()
    ppo_config = PPOConfig()
    grpo_config = GRPOConfig()
    eval_config = EvalConfig()
    
    print("\n✓ All configurations loaded")
    print(f"  Data config: {data_config.dataset_name}")
    print(f"  Model config: {model_config.model_name}")
    print(f"  DPO beta: {dpo_config.beta}")
    print(f"  PPO kl_coef: {ppo_config.kl_coef}")
    print(f"  GRPO group_size: {grpo_config.group_size}")