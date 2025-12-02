"""
Main Orchestration Script - Run All Experiments

This script runs the complete experimental pipeline:
1. Data preparation
2. Reward model training
3. Alignment methods training (DPO, PPO, GRPO)
4. Evaluation pipeline
5. Reward hacking tests
6. Results collection and analysis

Usage:
    # Run everything with defaults
    python main.py --full_pipeline
    
    # Run specific stages
    python main.py --stage data
    python main.py --stage reward_model
    python main.py --stage alignment
    python main.py --stage evaluation
    
    # Run with custom config
    python main.py --full_pipeline --seeds 42 123 456 --epochs 5
    
    # Quick test run (small scale)
    python main.py --quick_test
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Main orchestrator for running all experiments"""
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        
        # Results tracking
        self.results = {
            'start_time': self.start_time.isoformat(),
            'stages_completed': [],
            'stages_failed': [],
            'experiments_run': [],
            'total_time': None,
        }
        
        # Paths
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.eval_dir = self.project_root / "eval"
        # self.project_root = "/kaggle/working/LLM_Alignment/"
        # self.data_dir = self.project_root / "data"
        # self.models_dir = self.project_root / "models"
        # self.checkpoints_dir = self.project_root / "checkpoints"
        # self.eval_dir = self.project_root / "eval"
        
        # Experiment configuration
        self.seeds = args.seeds if args.seeds else [42, 123, 456]
        self.epochs = args.epochs if args.epochs else 3
        self.batch_size = args.batch_size if args.batch_size else 8
        
        # Quick test mode (reduced scale)
        self.quick_test = args.quick_test
        if self.quick_test:
            logger.info("🚀 QUICK TEST MODE: Using reduced scale for fast testing")
            self.seeds = [42]
            self.epochs = 1
            self.batch_size = 4
    
    def run_command(self, cmd: List[str], stage_name: str, experiment_name: str = None) -> bool:
        """Run a command and handle errors"""
        cmd_str = ' '.join(cmd)
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {cmd_str}")
        logger.info(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"✅ SUCCESS: {stage_name}")
            if experiment_name:
                self.results['experiments_run'].append(experiment_name)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FAILED: {stage_name}")
            logger.error(f"Error: {e.stderr}")
            self.results['stages_failed'].append(stage_name)
            
            if self.args.stop_on_error:
                logger.error("Stopping pipeline due to error (--stop_on_error)")
                sys.exit(1)
            
            return False
        
        except Exception as e:
            logger.error(f"❌ UNEXPECTED ERROR: {stage_name}")
            logger.error(f"Error: {str(e)}")
            self.results['stages_failed'].append(stage_name)
            
            if self.args.stop_on_error:
                sys.exit(1)
            
            return False
    
    def stage_data_preparation(self):
        """Stage 1: Data Preparation"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: DATA PREPARATION")
        logger.info("="*80)
        
        cmd = [
            "python", "/kaggle/working/LLM_Alignment/scripts/prepare_scripts.py",
            "--seed", str(self.seeds[0]),
        ]
        
        if self.run_command(cmd, "Data Preparation"):
            self.results['stages_completed'].append('data_preparation')
            return True
        return False
    
    def stage_reward_model(self):
        """Stage 2: Reward Model Training"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: REWARD MODEL TRAINING")
        logger.info("="*80)
        
        # Train reward model (only need one)
        save_dir = self.models_dir / "reward_model"
        
        cmd = [
            "python", "/kaggle/working/LLM_Alignment/scripts/train_reward_model.py",
            "--batch_size", str(self.batch_size * 2),  # Larger batch for reward model
            "--lr", "5e-5",
            "--epochs", str(self.epochs),
            "--lora_r", "8",
            "--seed", str(self.seeds[0]),
            "--save_dir", str(save_dir),
        ]
        
        if self.run_command(cmd, "Reward Model Training", "reward_model"):
            self.results['stages_completed'].append('reward_model')
            self.results['reward_model_path'] = str(save_dir / "final_model")
            return True
        return False
    
    def stage_alignment_methods(self):
        """Stage 3: Train All Alignment Methods"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: ALIGNMENT METHODS TRAINING")
        logger.info("="*80)
        
        reward_model_path = self.results.get('reward_model_path')
        if not reward_model_path:
            logger.error("Reward model path not found. Skipping alignment methods.")
            return False
        
        all_success = True
        
        # DPO experiments
        logger.info("\n--- Training DPO ---")
        if self.args.train_dpo or self.args.full_pipeline:
            for seed in self.seeds:
                save_dir = self.checkpoints_dir / f"dpo_seed_{seed}"
                cmd = [
                    "python", "/kaggle/working/LLM_Alignment/scripts/train_dpo.py",
                    "--method", "DPO",
                    "--batch_size", str(self.batch_size),
                    "--lr", "1e-4",
                    "--epochs", str(self.epochs),
                    "--lora_r", "8",
                    "--beta", "0.1",
                    "--seed", str(seed),
                    "--save_dir", str(save_dir),
                ]
                
                success = self.run_command(cmd, f"DPO (seed={seed})", f"dpo_seed_{seed}")
                all_success = all_success and success
        
        # PPO experiments (sparse)
        logger.info("\n--- Training PPO (Sparse) ---")
        if self.args.train_ppo or self.args.full_pipeline:
            for seed in self.seeds:
                save_dir = self.checkpoints_dir / f"ppo_sparse_seed_{seed}"
                cmd = [
                    "python", "/kaggle/working/LLM_Alignment/scripts/train_ppo.py",
                    "--method", "PPO",
                    "--reward_model_path", reward_model_path,
                    "--reward_mode", "sparse",
                    "--batch_size", str(self.batch_size),
                    "--lr", "1e-5",
                    "--kl_coef", "0.05",
                    "--epochs", str(self.epochs),
                    "--seed", str(seed),
                    "--save_dir", str(save_dir),
                ]
                
                success = self.run_command(cmd, f"PPO-Sparse (seed={seed})", f"ppo_sparse_seed_{seed}")
                all_success = all_success and success
        
        # PPO experiments (dense) - only one seed to save time
        logger.info("\n--- Training PPO (Dense) ---")
        if (self.args.train_ppo or self.args.full_pipeline) and not self.quick_test:
            seed = self.seeds[0]
            save_dir = self.checkpoints_dir / f"ppo_dense_seed_{seed}"
            cmd = [
                "python", "/kaggle/working/LLM_Alignment/scripts/train_ppo.py",
                "--method", "PPO",
                "--reward_model_path", reward_model_path,
                "--reward_mode", "dense",
                "--batch_size", str(self.batch_size),
                "--lr", "1e-5",
                "--kl_coef", "0.05",
                "--epochs", str(self.epochs),
                "--seed", str(seed),
                "--save_dir", str(save_dir),
            ]
            
            success = self.run_command(cmd, f"PPO-Dense (seed={seed})", f"ppo_dense_seed_{seed}")
            all_success = all_success and success
        
        # GRPO experiments
        logger.info("\n--- Training GRPO ---")
        if self.args.train_grpo or self.args.full_pipeline:
            for seed in self.seeds:
                save_dir = self.checkpoints_dir / f"grpo_seed_{seed}"
                cmd = [
                    "python", "/kaggle/working/LLM_Alignment/scripts/train_grpo.py",
                    "--method", "GRPO",
                    "--reward_model_path", reward_model_path,
                    "--group_size", "8",
                    "--advantage_normalization", "rank",
                    "--batch_size", str(max(self.batch_size // 2, 2)),  # GRPO needs smaller batch
                    "--lr", "1e-5",
                    "--epochs", str(self.epochs),
                    "--seed", str(seed),
                    "--save_dir", str(save_dir),
                ]
                
                success = self.run_command(cmd, f"GRPO (seed={seed})", f"grpo_seed_{seed}")
                all_success = all_success and success
        
        if all_success:
            self.results['stages_completed'].append('alignment_methods')
        
        return all_success
    
    def stage_ablations(self):
        """Stage 4: Run Ablation Studies"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: ABLATION STUDIES")
        logger.info("="*80)
        
        if self.quick_test:
            logger.info("Skipping ablations in quick test mode")
            return True
        
        reward_model_path = self.results.get('reward_model_path')
        if not reward_model_path:
            logger.error("Reward model path not found. Skipping ablations.")
            return False
        
        all_success = True
        seed = self.seeds[0]  # Use single seed for ablations
        
        # DPO: Learning rate sweep
        if self.args.run_ablations or self.args.full_pipeline:
            logger.info("\n--- DPO Learning Rate Sweep ---")
            for lr in ["1e-4", "5e-5", "1e-5"]:
                save_dir = self.checkpoints_dir / "ablations" / f"dpo_lr_{lr}"
                cmd = [
                    "python", "/kaggle/working/LLM_Alignment/scripts/train_dpo.py",
                    "--method", "DPO",
                    "--batch_size", str(self.batch_size),
                    "--lr", lr,
                    "--epochs", str(self.epochs),
                    "--seed", str(seed),
                    "--save_dir", str(save_dir),
                ]
                
                success = self.run_command(cmd, f"DPO LR={lr}", f"dpo_lr_{lr}")
                all_success = all_success and success
            
            # PPO: KL coefficient sweep
            logger.info("\n--- PPO KL Coefficient Sweep ---")
            for kl_coef in ["0.01", "0.05", "0.1"]:
                save_dir = self.checkpoints_dir / "ablations" / f"ppo_kl_{kl_coef}"
                cmd = [
                    "python", "/kaggle/working/LLM_Alignment/scripts/train_ppo.py",
                    "--method", "PPO",
                    "--reward_model_path", reward_model_path,
                    "--reward_mode", "sparse",
                    "--batch_size", str(self.batch_size),
                    "--lr", "1e-5",
                    "--kl_coef", kl_coef,
                    "--epochs", str(self.epochs),
                    "--seed", str(seed),
                    "--save_dir", str(save_dir),
                ]
                
                success = self.run_command(cmd, f"PPO KL={kl_coef}", f"ppo_kl_{kl_coef}")
                all_success = all_success and success
            
            # PPO: No-KL ablation
            logger.info("\n--- PPO No-KL Ablation ---")
            save_dir = self.checkpoints_dir / "ablations" / "ppo_no_kl"
            cmd = [
                "python", "/kaggle/working/LLM_Alignment/scripts/train_ppo.py",
                "--method", "PPO",
                "--reward_model_path", reward_model_path,
                "--reward_mode", "sparse",
                "--batch_size", str(self.batch_size),
                "--lr", "1e-5",
                "--kl_coef", "0.0",
                "--epochs", str(self.epochs),
                "--seed", str(seed),
                "--save_dir", str(save_dir),
            ]
            
            success = self.run_command(cmd, "PPO No-KL", "ppo_no_kl")
            all_success = all_success and success
            
            # GRPO: Group size sweep
            logger.info("\n--- GRPO Group Size Sweep ---")
            for group_size in ["4", "8", "16"]:
                save_dir = self.checkpoints_dir / "ablations" / f"grpo_g{group_size}"
                cmd = [
                    "python", "/kaggle/working/LLM_Alignment/scripts/train_grpo.py",
                    "--method", "GRPO",
                    "--reward_model_path", reward_model_path,
                    "--group_size", group_size,
                    "--batch_size", str(max(self.batch_size // 2, 2)),
                    "--lr", "1e-5",
                    "--epochs", str(self.epochs),
                    "--seed", str(seed),
                    "--save_dir", str(save_dir),
                ]
                
                success = self.run_command(cmd, f"GRPO Group={group_size}", f"grpo_g{group_size}")
                all_success = all_success and success
        
        if all_success:
            self.results['stages_completed'].append('ablations')
        
        return all_success
    
    def stage_evaluation(self):
        """Stage 5: Evaluation Pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: EVALUATION PIPELINE")
        logger.info("="*80)
        
        reward_model_path = self.results.get('reward_model_path')
        
        # Find all trained models
        model_dirs = []
        
        # Add main experiments
        for seed in self.seeds:
            for method in ['dpo', 'ppo_sparse', 'grpo']:
                model_dir = self.checkpoints_dir / f"{method}_seed_{seed}" / "final_model"
                if model_dir.exists():
                    model_dirs.append((method, seed, model_dir))
        
        if not model_dirs:
            logger.error("No trained models found for evaluation")
            return False
        
        logger.info(f"Found {len(model_dirs)} models to evaluate")
        
        all_success = True
        
        # Create test set if it doesn't exist
        test_file = self.eval_dir / "testset_50.jsonl"
        if not test_file.exists():
            logger.warning(f"Test file not found at {test_file}")
            logger.info("You need to create a 50-prompt test set manually")
            logger.info("See README.md for test set requirements")
            # Create a dummy test set for demonstration
            self.create_dummy_test_set(test_file)
        
        # Evaluate each model
        for method, seed, model_path in model_dirs:
            logger.info(f"\n--- Evaluating {method} (seed={seed}) ---")
            
            # 1. Generate outputs
            output_file = self.eval_dir / f"{method}_seed_{seed}_outputs.jsonl"
            cmd = [
                "python", "/kaggle/working/LLM_Alignment/scripts/eval_generate.py",
                "--model_path", str(model_path),
                "--test_file", str(test_file),
                "--output_file", str(output_file),
                "--temperature", "0.7",
                "--seed", str(seed),
            ]
            
            if not self.run_command(cmd, f"Generate {method} (seed={seed})"):
                all_success = False
                continue
            
            # 2. Compute metrics
            metrics_file = self.eval_dir / f"{method}_seed_{seed}_metrics.csv"
            cmd = [
                "python", "/kaggle/working/LLM_Alignment/scripts/eval_metrics.py",
                "--generated_file", str(output_file),
                "--output_file", str(metrics_file),
                "--seed", str(seed),
            ]
            
            # Add optional models if available
            if reward_model_path:
                cmd.extend(["--reward_model_path", reward_model_path])
            
            cmd.extend([
                "--reference_model_path", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--policy_model_path", str(model_path),
            ])
            
            if not self.run_command(cmd, f"Metrics {method} (seed={seed})"):
                all_success = False
        
        if all_success:
            self.results['stages_completed'].append('evaluation')
        
        return all_success
    
    def stage_reward_hacking(self):
        """Stage 6: Reward Hacking Tests"""
        logger.info("\n" + "="*80)
        logger.info("STAGE 6: REWARD HACKING TESTS")
        logger.info("="*80)
        
        reward_model_path = self.results.get('reward_model_path')
        if not reward_model_path:
            logger.error("Reward model not found. Skipping reward hacking tests.")
            return False
        
        test_file = self.eval_dir / "testset_50.jsonl"
        if not test_file.exists():
            logger.warning(f"Test file not found at {test_file}")
            return False
        
        all_success = True
        
        # Test PPO models (most susceptible to hacking)
        for seed in self.seeds[:1]:  # Test first seed only
            model_path = self.checkpoints_dir / f"ppo_sparse_seed_{seed}" / "final_model"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            
            logger.info(f"\n--- Testing PPO (seed={seed}) for reward hacking ---")
            
            output_file = self.eval_dir / f"ppo_seed_{seed}_hack_tests.csv"
            cmd = [
                "python", "/kaggle/working/LLM_Alignment/scripts/perturbations.py",
                "--model_path", str(model_path),
                "--reward_model_path", reward_model_path,
                "--test_file", str(test_file),
                "--output_file", str(output_file),
                "--seed", str(seed),
            ]
            
            if not self.run_command(cmd, f"Reward Hacking PPO (seed={seed})"):
                all_success = False
        
        if all_success:
            self.results['stages_completed'].append('reward_hacking')
        
        return all_success
    
    def create_dummy_test_set(self, output_path: Path):
        """Create a dummy test set for demonstration"""
        logger.info("Creating dummy test set...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create 50 diverse prompts
        prompts = []
        
        # Factual short-answer (10)
        for i in range(10):
            prompts.append({
                'id': f'factual_{i}',
                'prompt': f'What is example factual question {i}?',
                'category': 'factual_short',
                'expected_behavior': 'brief factual answer'
            })
        
        # Open-ended (10)
        for i in range(10):
            prompts.append({
                'id': f'openended_{i}',
                'prompt': f'Explain the concept of example topic {i}.',
                'category': 'open_ended',
                'expected_behavior': 'detailed explanation'
            })
        
        # Instruction-following (10)
        for i in range(10):
            prompts.append({
                'id': f'instruction_{i}',
                'prompt': f'Summarize example topic {i} in 50 words or less.',
                'category': 'instruction_following',
                'expected_behavior': 'follow length constraint'
            })
        
        # Verbosity-hack (10)
        for i in range(10):
            prompts.append({
                'id': f'verbosity_{i}',
                'prompt': f'Explain example topic {i} in great detail.',
                'category': 'verbosity_hack',
                'expected_behavior': 'detailed but not excessive'
            })
        
        # Adversarial (10)
        for i in range(10):
            prompts.append({
                'id': f'adversarial_{i}',
                'prompt': f'Is this safe? Tell me about example {i}.',
                'category': 'adversarial',
                'expected_behavior': 'handle vague prompt appropriately'
            })
        
        # Save
        with open(output_path, 'w') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + '\n')
        
        logger.info(f"Created dummy test set: {output_path}")
    
    def save_results(self):
        """Save final results summary"""
        # Calculate total time
        end_time = datetime.now()
        self.results['end_time'] = end_time.isoformat()
        self.results['total_time'] = str(end_time - self.start_time)
        
        # Save results
        results_file = self.project_root / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {self.results['total_time']}")
        logger.info(f"Stages completed: {len(self.results['stages_completed'])}")
        logger.info(f"Stages failed: {len(self.results['stages_failed'])}")
        logger.info(f"Experiments run: {len(self.results['experiments_run'])}")
        logger.info(f"\nResults saved to: {results_file}")
        logger.info(f"{'='*80}")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING COMPLETE EXPERIMENTAL PIPELINE")
        logger.info("="*80)
        logger.info(f"Seeds: {self.seeds}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Quick test mode: {self.quick_test}")
        logger.info("="*80)
        
        try:
            # Stage 1: Data Preparation
            if self.args.stage in [None, 'data'] or self.args.full_pipeline:
                if not self.stage_data_preparation():
                    logger.error("Data preparation failed")
                    if self.args.stop_on_error:
                        return
            
            # Stage 2: Reward Model
            if self.args.stage in [None, 'reward_model'] or self.args.full_pipeline:
                if not self.stage_reward_model():
                    logger.error("Reward model training failed")
                    if self.args.stop_on_error:
                        return
            
            # Stage 3: Alignment Methods
            if self.args.stage in [None, 'alignment'] or self.args.full_pipeline:
                if not self.stage_alignment_methods():
                    logger.error("Alignment methods training failed")
                    if self.args.stop_on_error:
                        return
            
            # Stage 4: Ablations
            if self.args.stage in [None, 'ablations'] or (self.args.full_pipeline and self.args.run_ablations):
                if not self.stage_ablations():
                    logger.error("Ablation studies failed")
                    # Don't stop on ablation failures
            
            # Stage 5: Evaluation
            if self.args.stage in [None, 'evaluation'] or self.args.full_pipeline:
                if not self.stage_evaluation():
                    logger.error("Evaluation pipeline failed")
                    # Don't stop on evaluation failures
            
            # Stage 6: Reward Hacking
            if self.args.stage in [None, 'reward_hacking'] or self.args.full_pipeline:
                if not self.stage_reward_hacking():
                    logger.error("Reward hacking tests failed")
                    # Don't stop on hacking test failures
            
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE COMPLETED")
            logger.info("="*80)
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Pipeline interrupted by user")
        
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        
        finally:
            # Always save results
            self.save_results()


def main():
    parser = argparse.ArgumentParser(
        description='Run complete experimental pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything with defaults
  python main.py --full_pipeline
  
  # Quick test (reduced scale)
  python main.py --quick_test
  
  # Run specific stage
  python main.py --stage data
  python main.py --stage reward_model
  python main.py --stage alignment
  
  # Custom configuration
  python main.py --full_pipeline --seeds 42 123 456 --epochs 5 --batch_size 16
  
  # Run only specific methods
  python main.py --train_dpo --train_ppo
  
  # Include ablations
  python main.py --full_pipeline --run_ablations
        """
    )
    
    # Pipeline control
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Run complete pipeline (all stages)')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test mode (reduced scale, 1 seed, 1 epoch)')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['data', 'reward_model', 'alignment', 'ablations', 'evaluation', 'reward_hacking'],
                        help='Run specific stage only')
    
    # Method selection
    parser.add_argument('--train_dpo', action='store_true',
                        help='Train DPO models')
    parser.add_argument('--train_ppo', action='store_true',
                        help='Train PPO models')
    parser.add_argument('--train_grpo', action='store_true',
                        help='Train GRPO models')
    parser.add_argument('--run_ablations', action='store_true',
                        help='Run ablation studies')
    
    # Configuration
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Random seeds (default: [42, 123, 456])')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size (default: 8)')
    
    # Error handling
    parser.add_argument('--stop_on_error', action='store_true',
                        help='Stop pipeline on first error (default: continue)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full_pipeline, args.quick_test, args.stage, 
                args.train_dpo, args.train_ppo, args.train_grpo]):
        parser.error("Must specify --full_pipeline, --quick_test, --stage, or specific methods")
    
    # Create orchestrator and run
    orchestrator = ExperimentOrchestrator(args)
    orchestrator.run_pipeline()


if __name__ == "__main__":
    main()
