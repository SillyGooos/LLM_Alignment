"""
Default Configuration for Alignment Methods Assignment
Adapted for Kaggle Notebook Environment
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import json

@dataclass
class PathConfig:
    """Directory structure for the project"""
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_root: Path = field(default_factory=lambda: Path("data"))
    data_raw: Path = field(default_factory=lambda: Path("data/raw"))
    data_processed: Path = field(default_factory=lambda: Path("data/processed"))
    models_root: Path = field(default_factory=lambda: Path("models"))
    models_baseline: Path = field(default_factory=lambda: Path("models/baseline"))
    models_dpo: Path = field(default_factory=lambda: Path("models/dpo"))
    models_ppo: Path = field(default_factory=lambda: Path("models/ppo"))
    models_grpo: Path = field(default_factory=lambda: Path("models/grpo"))
    models_reward: Path = field(default_factory=lambda: Path("models/reward_model"))
    eval_root: Path = field(default_factory=lambda: Path("eval"))
    logs_root: Path = field(default_factory=lambda: Path("logs"))
    checkpoints_root: Path = field(default_factory=lambda: Path("checkpoints"))
    
    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            if field_name != 'project_root':
                path = getattr(self, field_name)
                if isinstance(path, Path):
                    setattr(self, field_name, self.project_root / path)
    
    def create_dirs(self):
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if isinstance(path, Path) and field_name != 'project_root':
                path.mkdir(parents=True, exist_ok=True)

@dataclass
class DataConfig:
    # Dataset
    dataset_name: str = "Intel/orca_dpo_pairs"
    dataset_subset: Optional[str] = None
    dataset_config: Optional[str] = None
    train_split: str = "train"
    
    # Splits
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    # Sampling
    max_samples: Optional[int] = None
    max_train_samples: Optional[int] = 50000
    max_val_samples: Optional[int] = 5000
    max_test_samples: Optional[int] = 5000
    
    # Filtering
    min_prompt_length: int = 10
    min_response_length: int = 10
    
    # Instruction following subset
    instr_following_size: int = 10000
    
    # Text processing
    max_length: int = 512
    truncation: bool = True
    
    # Seed
    seed: int = 42
    
    # Template formats
    prompt_template: str = "Question: {prompt}\n\nAnswer:"
    response_template: str = " {response}"
@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer_name: Optional[str] = None
    use_auth_token: bool = False
    trust_remote_code: bool = True
    load_in_8bit: bool = True
    device_map: str = "auto"
    torch_dtype: str = "float16"
    padding_side: str = "left"
    max_length: int = 512
    truncation: bool = True
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

@dataclass
class RewardModelConfig:
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_accuracy"
    load_best_model_at_end: bool = True
    seed: int = 42

@dataclass
class DPOConfig:
    method: str = "DPO"
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    beta: float = 0.1
    reference_free: bool = False
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    seed: int = 42

@dataclass
class PPOConfig:
    method: str = "PPO"
    reward_mode: str = "sparse"
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_path: str = "models/reward_model/final_model"
    kl_coef: float = 0.05
    clip_range: float = 0.2
    vf_coef: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    num_epochs: int = 3
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42

@dataclass
class GRPOConfig:
    method: str = "GRPO"
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_path: str = "models/reward_model/final_model"
    group_size: int = 8
    advantage_normalization: str = "rank"
    use_baseline: bool = False
    temperature: float = 0.7
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_new_tokens: int = 256
    top_k: int = 50
    top_p: float = 0.95
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42

@dataclass
class EvalConfig:
    test_file: str = "eval/testset_50.jsonl"
    generation_configs: List[str] = field(default_factory=lambda: ["greedy", "temp_0.7", "temp_0.2", "temp_1.0", "top_k_5"])
    num_samples: int = 1
    models_to_eval: List[str] = field(default_factory=lambda: ["baseline", "dpo", "ppo_sparse", "ppo_dense", "grpo"])
    metrics: List[str] = field(default_factory=lambda: ["reward", "kl_divergence", "perplexity", "length", "compliance"])
    perturbations: List[str] = field(default_factory=lambda: ["filler_phrases", "sentence_reorder", "synonym_replace", "formatting", "alignment_keywords"])
    seed: int = 42

@dataclass
class PerturbationConfig:
    reward_delta_threshold: float = 0.1

@dataclass  
class HardwareConfig:
    gradient_checkpointing: bool = True

@dataclass
class LoRAConfig:
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

def get_default_config():
    class Config:
        def __init__(self):
            self.paths = PathConfig()
            self.data = DataConfig()
            self.base_model = ModelConfig()
            self.reward_model = RewardModelConfig()
            self.dpo = DPOConfig()
            self.ppo = PPOConfig()
            self.grpo = GRPOConfig()
            self.eval = EvalConfig()
    return Config()

class Config:
    def __init__(self):
        self.paths = PathConfig()
        self.data = DataConfig()
        self.base_model = ModelConfig()
        self.reward_model = RewardModelConfig()
        self.dpo = DPOConfig()
        self.ppo = PPOConfig()
        self.grpo = GRPOConfig()
        self.eval = EvalConfig()
