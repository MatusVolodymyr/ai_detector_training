"""
Simplified dataset processing pipeline with incremental saving.
Processes dataset in chunks and saves progress to allow resuming if interrupted.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
import gc
import torch
import logging
import warnings

# Suppress verbose logging and progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Reduce TQDM verbosity
from tqdm import tqdm
tqdm.pandas(disable=True)  # Disable pandas progress bars
import datasets
datasets.disable_progress_bar()  # Disable datasets progress bars

# Import our modules
import sys
sys.path.append('/root/projects/ai_detector-data/modules')

from stylometric_extraction import (
    initialize_models as init_stylometric_models,
    process_dataset_batch,
    Config
)
from prediction_service_dataset import (
    sequential_model_predictions,
    ModelManager,
    PredictionService
)


class IncrementalDatasetProcessor:
    """Process dataset in chunks with incremental saving."""
    
    def __init__(self, model_config: Dict[str, Tuple[str, str]], 
                 chunk_size: int = 50000,
                 enable_perplexity: bool = True):
        """
        Initialize the incremental processor.
        
        Args:
            model_config: Dictionary mapping model IDs to (model_path, window_column)
            chunk_size: Number of examples to process per chunk
            enable_perplexity: Whether to calculate perplexity features
        """
        self.model_config = model_config
        self.chunk_size = chunk_size
        self.enable_perplexity = enable_perplexity
        self.stylometric_initialized = False
        
        # Configure stylometric processing
        Config.enable_perplexity = enable_perplexity
        Config.enable_stylometric = True
        Config.perplexity_batch_size = 8  # Conservative batch size
    
    def initialize_stylometric_models(self):
        """Initialize stylometric models once."""
        if not self.stylometric_initialized:
            init_stylometric_models()
            self.stylometric_initialized = True
    
    def add_stylometric_features_to_chunk(self, chunk: Dataset, batch_size: int = 100) -> Dataset:
        """Add stylometric features to a single chunk."""
        self.initialize_stylometric_models()
        
        def process_batch(examples):
            return process_dataset_batch(examples)
        
        chunk_with_features = chunk.map(
            process_batch,
            batched=True,
            batch_size=batch_size,
            desc="Stylometric features"
        )
        
        return chunk_with_features
    
    def add_model_predictions_to_chunk(self, chunk: Dataset, batch_size: int = 100) -> Dataset:
        """Add model predictions to a single chunk."""
        return sequential_model_predictions(chunk, self.model_config, batch_size)
    
    def process_single_chunk(self, chunk: Dataset, chunk_idx: int, 
                           include_stylometric: bool = True,
                           include_predictions: bool = True,
                           batch_size: int = 100) -> Dataset:
        """Process a single chunk with all features."""
        print(f"\n📦 Chunk {chunk_idx + 1} ({len(chunk):,} examples)")
        
        result_chunk = chunk
        
        # Add stylometric features
        if include_stylometric:
            print("   🔧 Adding stylometric features...")
            result_chunk = self.add_stylometric_features_to_chunk(result_chunk, batch_size)
            print("   ✅ Stylometric features complete")
        
        # Add model predictions
        if include_predictions:
            print("   🤖 Adding model predictions...")
            result_chunk = self.add_model_predictions_to_chunk(result_chunk, batch_size)
            print("   ✅ Model predictions complete")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result_chunk
    
    def save_chunk(self, chunk: Dataset, output_dir: str, chunk_idx: int):
        """Save a processed chunk."""
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.arrow")
        chunk.save_to_disk(chunk_path)
        print(f"   💾 Saved chunk {chunk_idx + 1}")
    
    def load_existing_chunks(self, output_dir: str) -> List[Dataset]:
        """Load any existing processed chunks."""
        chunks = []
        if not os.path.exists(output_dir):
            return chunks
        
        chunk_files = [f for f in os.listdir(output_dir) if f.startswith("chunk_") and f.endswith(".arrow")]
        chunk_files.sort()
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(output_dir, chunk_file)
            try:
                chunk = load_from_disk(chunk_path)
                chunks.append(chunk)
            except Exception:
                pass  # Skip failed chunks silently
        
        if chunks:
            print(f"📂 Found {len(chunks)} existing chunks")
        
        return chunks
    
    def get_resume_info(self, output_dir: str, total_examples: int) -> Tuple[int, int]:
        """Get information about where to resume processing."""
        if not os.path.exists(output_dir):
            return 0, 0
        
        chunk_files = [f for f in os.listdir(output_dir) if f.startswith("chunk_") and f.endswith(".arrow")]
        
        if not chunk_files:
            return 0, 0
        
        # Count processed examples
        processed_examples = 0
        max_chunk_idx = -1
        
        for chunk_file in chunk_files:
            try:
                # Extract chunk index from filename
                chunk_idx = int(chunk_file.split("_")[1].split(".")[0])
                max_chunk_idx = max(max_chunk_idx, chunk_idx)
                
                # Load chunk to count examples
                chunk_path = os.path.join(output_dir, chunk_file)
                chunk = load_from_disk(chunk_path)
                processed_examples += len(chunk)
            except Exception as e:
                print(f"⚠️  Error reading chunk {chunk_file}: {e}")
        
        next_chunk_idx = max_chunk_idx + 1
        return next_chunk_idx, processed_examples
    
    def process_dataset(self, dataset: Dataset, output_dir: str,
                       include_stylometric: bool = True,
                       include_predictions: bool = True,
                       batch_size: int = 100,
                       resume: bool = True) -> str:
        """
        Process the complete dataset in chunks.
        
        Args:
            dataset: Input dataset to process
            output_dir: Directory to save processed chunks
            include_stylometric: Whether to add stylometric features
            include_predictions: Whether to add model predictions
            batch_size: Batch size for processing
            resume: Whether to resume from existing chunks
        
        Returns:
            Path to the final concatenated dataset
        """
        total_examples = len(dataset)
        total_chunks = (total_examples + self.chunk_size - 1) // self.chunk_size
        
        print(f"🚀 Starting incremental dataset processing")
        print(f"Total examples: {total_examples:,}")
        print(f"Chunk size: {self.chunk_size:,}")
        print(f"Total chunks: {total_chunks}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for resuming
        start_chunk_idx = 0
        processed_examples = 0
        
        if resume:
            start_chunk_idx, processed_examples = self.get_resume_info(output_dir, total_examples)
            if start_chunk_idx > 0:
                print(f"📋 Resuming from chunk {start_chunk_idx + 1}")
                print(f"Already processed: {processed_examples:,} examples")
        
        # Process chunks
        for chunk_idx in range(start_chunk_idx, total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, total_examples)
            
            print(f"\n🔄 Processing chunk {chunk_idx + 1}/{total_chunks} (examples {start_idx:,}-{end_idx-1:,})")
            
            # Extract chunk
            chunk = dataset.select(range(start_idx, end_idx))
            
            try:
                # Process chunk
                processed_chunk = self.process_single_chunk(
                    chunk, chunk_idx,
                    include_stylometric=include_stylometric,
                    include_predictions=include_predictions,
                    batch_size=batch_size
                )
                
                # Save chunk
                self.save_chunk(processed_chunk, output_dir, chunk_idx)
                
                print(f"✅ Chunk {chunk_idx + 1}/{total_chunks} complete!")
                
            except Exception as e:
                print(f"❌ Chunk {chunk_idx + 1} failed: {e}")
                print(f"💡 Resume by re-running the processing command")
                raise e
        
        print(f"\n🎉 All chunks processed successfully!")
        
        # Concatenate all chunks
        final_path = self.concatenate_chunks(output_dir)
        
        return final_path
    
    def concatenate_chunks(self, output_dir: str) -> str:
        """Concatenate all processed chunks into a single dataset."""
        print(f"\n📦 Concatenating chunks...")
        
        # Load all chunks
        chunks = self.load_existing_chunks(output_dir)
        
        if not chunks:
            raise ValueError("No chunks found to concatenate!")
        
        # Concatenate
        final_dataset = concatenate_datasets(chunks)
        
        # Save final dataset
        final_path = os.path.join(output_dir, "final_dataset.arrow")
        final_dataset.save_to_disk(final_path)
        
        print(f"✅ Final dataset: {len(final_dataset):,} examples, {len(final_dataset.column_names)} columns")
        print(f"📁 Saved to: {final_path}")
        
        return final_path
    
    def get_processing_status(self, output_dir: str, total_examples: int) -> Dict:
        """Get current processing status."""
        if not os.path.exists(output_dir):
            return {
                "status": "not_started",
                "processed_chunks": 0,
                "processed_examples": 0,
                "remaining_examples": total_examples,
                "progress_percent": 0.0
            }
        
        next_chunk_idx, processed_examples = self.get_resume_info(output_dir, total_examples)
        remaining_examples = total_examples - processed_examples
        progress_percent = (processed_examples / total_examples) * 100
        
        return {
            "status": "in_progress" if remaining_examples > 0 else "completed",
            "processed_chunks": next_chunk_idx,
            "processed_examples": processed_examples,
            "remaining_examples": remaining_examples,
            "progress_percent": progress_percent
        }


def create_processor(model_config: Optional[Dict[str, Tuple[str, str]]] = None,
                    chunk_size: int = 50000,
                    enable_perplexity: bool = True) -> IncrementalDatasetProcessor:
    """Create a processor with default or custom configuration."""
    
    if model_config is None:
        # Default model configuration
        model_config = {
            "s1_1": ("NeksoN/mdeberta_s1_1_v2", "window_1"),
            "s3_1": ("NeksoN/mdeberta_s3_1_v2", "window_3"),
            "s5_1": ("NeksoN/mdeberta_s5_1_v2", "window_5")
        }
    
    return IncrementalDatasetProcessor(model_config, chunk_size, enable_perplexity)


# Example usage
if __name__ == "__main__":
    print("Incremental Dataset Processor")
    print("=" * 50)
    
    # Create processor
    processor = create_processor()
    
    # Example usage (uncomment to test)
    # dataset = load_from_disk("your_dataset_path")
    # final_path = processor.process_dataset(
    #     dataset=dataset,
    #     output_dir="processed_chunks",
    #     include_stylometric=True,
    #     include_predictions=True,
    #     batch_size=100,
    #     resume=True
    # )
    
    print("Processor ready! Use process_dataset() to start processing.")
