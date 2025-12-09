"""
GRPO (Group Relative Policy Optimization) Training Script - SIMPLIFIED WORKING VERSION

This is a SIMPLE, WORKING implementation. No fancy batching, just correctness.

Usage:
    python train_grpo_fixed.py \
      --reward_model_path ./models/reward_model/final_model \
      --group_size 4 \
      --batch_size 2 \
      --epochs 3 \
      --seed 42
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

from config.default_config import get_default_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_quantization_config(load_in_4bit=True, load_in_8bit=False, mixed_precision='fp16'):
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if mixed_precision == "fp16" else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


class SimpleGRPOTrainer:
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        
        self.device = torch.device('cuda')
        self.setup_paths()
        
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.optimizer = None
        self.train_data = []
        self.training_history = []
        
    def setup_paths(self):
        if self.args.save_dir:
            self.save_dir = Path(self.args.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(self.args.output_dir) / f"grpo_g{self.args.group_size}_{timestamp}"
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        self.plots_dir = self.save_dir / "plots"
        
        for d in [self.checkpoint_dir, self.logs_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def setup_tokenizer(self):
        logger.info(f"Loading tokenizer: {self.args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logger.info("✓ Tokenizer ready")
    
    def load_reward_model(self):
        logger.info("Loading reward model...")
        bnb_config = create_quantization_config(
            self.args.load_in_4bit, self.args.load_in_8bit, self.args.mixed_precision
        )
        
        try:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.args.reward_model_path,
                num_labels=1,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
        except:
            base = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name,
                num_labels=1,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=self.config.base_model.trust_remote_code,
            )
            self.reward_model = PeftModel.from_pretrained(base, self.args.reward_model_path)
        
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        self.reward_device = next(self.reward_model.parameters()).device
        logger.info(f"✓ Reward model on {self.reward_device}")
    
    def load_datasets(self):
        data_dir = Path(self.args.data_dir)
        with open(data_dir / "train.jsonl", 'r') as f:
            for line in f:
                self.train_data.append(json.loads(line))
        logger.info(f"Loaded {len(self.train_data)} train examples")
    
    def setup_models(self):
        logger.info("Loading models...")
        bnb_config = create_quantization_config(
            self.args.load_in_4bit, self.args.load_in_8bit, self.args.mixed_precision
        )
        
        # Policy
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=self.config.base_model.trust_remote_code,
            torch_dtype=torch.float16,
        )
        if self.args.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
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
        
        self.model_device = next(self.model.parameters()).device
        logger.info(f"✓ Policy on {self.model_device}")
        
        # Reference
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=self.config.base_model.trust_remote_code,
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        self.ref_device = next(self.ref_model.parameters()).device
        logger.info(f"✓ Reference on {self.ref_device}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
    
    def compute_reward(self, prompt: str, response: str) -> float:
        text = self.config.data.prompt_template.format(prompt=prompt)
        text += self.config.data.response_template.format(response=response)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.reward_device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.reward_model(**inputs).logits.squeeze().item()
    
    def generate_response(self, prompt: str) -> Tuple[str, torch.Tensor]:
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model_device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_ids = outputs[0, prompt_ids.shape[1]:].to(self.model_device)
        response_text = self.tokenizer.decode(response_ids.cpu(), skip_special_tokens=True)
        return response_text, response_ids
    
    def compute_log_probs(self, prompt: str, response_ids: torch.Tensor, model, device) -> torch.Tensor:
        prompt_text = self.config.data.prompt_template.format(prompt=prompt)
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        full_ids = torch.cat([prompt_ids[0], response_ids.to(device)]).unsqueeze(0)
        
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            logits = model(input_ids=full_ids).logits[0]
        
        log_probs_all = F.log_softmax(logits, dim=-1)
        prompt_len = prompt_ids.shape[1]
        
        log_probs = []
        for i, tok_id in enumerate(response_ids.to(device)):
            pos = prompt_len + i - 1
            if 0 <= pos < logits.shape[0]:
                log_probs.append(log_probs_all[pos, tok_id.item()])
        
        return torch.stack(log_probs) if log_probs else torch.tensor(0.0, device=device)
    
    def train_step(self, prompt: str) -> Dict:
        # Generate group
        responses = []
        response_ids_list = []
        rewards = []
        
        for _ in range(self.args.group_size):
            resp_text, resp_ids = self.generate_response(prompt)
            rew = self.compute_reward(prompt, resp_text)
            responses.append(resp_text)
            response_ids_list.append(resp_ids)
            rewards.append(rew)
        
        rewards = np.array(rewards)
        
        # Advantages
        ranks = stats.rankdata(rewards, method='average')
        advantages = (ranks - np.mean(ranks)) / (np.std(ranks) + 1e-8)
        
        # Losses
        total_loss = 0.0
        for i in range(self.args.group_size):
            policy_lp = self.compute_log_probs(prompt, response_ids_list[i], self.model, self.model_device)
            
            with torch.no_grad():
                ref_lp = self.compute_log_probs(prompt, response_ids_list[i], self.ref_model, self.ref_device)
                ref_lp = ref_lp.to(self.model_device)
            
            if policy_lp.numel() > 0:
                log_ratio = policy_lp - ref_lp
                loss = -advantages[i] * log_ratio.mean() / self.args.group_size
                loss.backward()
                total_loss += loss.item()
        
        return {'loss': total_loss, 'mean_reward': rewards.mean(), 'std_reward': rewards.std()}
    
    def train(self):
        logger.info(f"\nTraining: {len(self.train_data)} samples, group_size={self.args.group_size}\n")
        
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.epochs}")
            self.model.train()
            np.random.shuffle(self.train_data)
            
            pbar = tqdm(range(0, len(self.train_data), self.args.batch_size))
            for batch_idx in pbar:
                batch = self.train_data[batch_idx:batch_idx+self.args.batch_size]
                
                self.optimizer.zero_grad()
                
                batch_loss = []
                batch_reward = []
                for item in batch:
                    try:
                        metrics = self.train_step(item['prompt'])
                        batch_loss.append(metrics['loss'])
                        batch_reward.append(metrics['mean_reward'])
                        self.training_history.append({'epoch': epoch, **metrics})
                    except Exception as e:
                        logger.warning(f"Step failed: {e}")
                
                if batch_loss:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    pbar.set_postfix({'loss': f"{np.mean(batch_loss):.4f}", 'reward': f"{np.mean(batch_reward):.3f}"})
            
            # Save
            self.model.save_pretrained(self.checkpoint_dir / f"epoch_{epoch+1}")
        
        self.model.save_pretrained(self.save_dir / "final_model")
        
        if self.training_history:
            pd.DataFrame(self.training_history).to_csv(self.logs_dir / "history.csv", index=False)
        
        logger.info("\n✓ Training complete")
    
    def run(self):
        try:
            with open(self.save_dir / "args.json", 'w') as f:
                json.dump(vars(self.args), f, indent=2)
            self.setup_tokenizer()
            self.load_reward_model()
            self.load_datasets()
            self.setup_models()
            self.train()
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--use_baseline', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--reward_model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-135M-Instruct')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true', default=True)
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    args = parser.parse_args()
    set_seed(args.seed)
    config = get_default_config()
    trainer = SimpleGRPOTrainer(args, config)
    trainer.run()


if __name__ == "__main__":
    main()