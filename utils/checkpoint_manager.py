"""
Checkpoint manager for resuming training
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
from datasets import Dataset, DatasetDict


class CheckpointManager:
    """Manages training checkpoints for resumability"""
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Track completed phases
        self.status_file = os.path.join(checkpoint_dir, 'status.json')
        self.status = self._load_status()
    
    def _load_status(self) -> Dict[str, bool]:
        """Load checkpoint status"""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_status(self):
        """Save checkpoint status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def check_phase_completed(self, phase: str) -> bool:
        """Check if a phase is already completed"""
        return self.status.get(phase, False)
    
    def mark_phase_complete(self, phase: str):
        """Mark a phase as completed"""
        self.status[phase] = True
        self._save_status()
    
    def save_checkpoint(self, phase: str, data: Any):
        """
        Save checkpoint data for a phase.
        
        Args:
            phase: Phase identifier
            data: Data to save (can be Dataset, dict, etc.)
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{phase}.pkl')
        
        # Handle different data types
        if isinstance(data, (Dataset, DatasetDict)):
            # Save datasets to disk in arrow format
            dataset_path = os.path.join(self.checkpoint_dir, f'{phase}.arrow')
            data.save_to_disk(dataset_path)
            # Save path reference
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'type': 'dataset', 'path': dataset_path}, f)
        elif isinstance(data, dict) and any(isinstance(v, Dataset) for v in data.values()):
            # Save dict of datasets
            dataset_paths = {}
            for key, value in data.items():
                if isinstance(value, Dataset):
                    ds_path = os.path.join(self.checkpoint_dir, f'{phase}_{key}.arrow')
                    value.save_to_disk(ds_path)
                    dataset_paths[key] = ds_path
                else:
                    dataset_paths[key] = value
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'type': 'dataset_dict', 'paths': dataset_paths}, f)
        else:
            # Save other data types with pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'type': 'other', 'data': data}, f)
        
        self.mark_phase_complete(phase)
    
    def load_checkpoint(self, phase: str) -> Any:
        """
        Load checkpoint data for a phase.
        
        Args:
            phase: Phase identifier
        
        Returns:
            Saved data
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{phase}.pkl')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        if checkpoint['type'] == 'dataset':
            from datasets import load_from_disk
            return load_from_disk(checkpoint['path'])
        elif checkpoint['type'] == 'dataset_dict':
            from datasets import load_from_disk
            result = {}
            for key, value in checkpoint['paths'].items():
                if isinstance(value, str) and value.endswith('.arrow'):
                    result[key] = load_from_disk(value)
                else:
                    result[key] = value
            return result
        else:
            return checkpoint['data']
    
    def list_completed_phases(self) -> List[str]:
        """List all completed phases"""
        return [phase for phase, completed in self.status.items() if completed]
    
    def clear_checkpoints(self):
        """Clear all checkpoints"""
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.status = {}
        self._save_status()
