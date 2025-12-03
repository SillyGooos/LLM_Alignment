"""
Checkpoint cleanup utility for PPO and GRPO training

Keeps only the N most recent checkpoints to save disk space.
"""

import shutil
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_n: int = 2):
    """
    Keep only the N most recent checkpoints, delete older ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_n: Number of most recent checkpoints to keep
    """
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint directories
    checkpoint_dirs = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and (d.name.startswith('checkpoint_') or d.name.startswith('epoch_'))
    ]
    
    if len(checkpoint_dirs) <= keep_n:
        return  # Nothing to delete
    
    # Sort by modification time (newest first)
    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep the N most recent, delete the rest
    to_delete = checkpoint_dirs[keep_n:]
    
    for checkpoint in to_delete:
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Deleted old checkpoint: {checkpoint.name}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint.name}: {e}")


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Get the path to the most recent checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to most recent checkpoint, or None if no checkpoints exist
    """
    if not checkpoint_dir.exists():
        return None
    
    # Get all checkpoint directories
    checkpoint_dirs = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and (d.name.startswith('checkpoint_') or d.name.startswith('epoch_'))
    ]
    
    if not checkpoint_dirs:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoint_dirs[0]