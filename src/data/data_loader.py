"""
Data Loader: Download and load Spotify dataset from Kaggle
Pipeline Setup: Phase 1 - Download 50K Spotify sample
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    SPOTIFY_RAW_PATH,
    PHASE_1_SAMPLE_SIZE,
    PHASE_1_SEED,
    KAGGLE_SPOTIFY_DATASET,
    KAGGLE_SPOTIFY_FILE,
    LOG_FORMAT,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class SpotifyDataLoader:
    """Load and validate Spotify dataset from Kaggle"""
    
    def __init__(self):
        self.dataset = None
        self.sample = None
        
    def download_from_kaggle(self):
        """Download Spotify dataset using kagglehub"""
        try:
            import kagglehub
            logger.info(f"Downloading dataset: {KAGGLE_SPOTIFY_DATASET}")
            
            # Download dataset
            path = kagglehub.dataset_download(KAGGLE_SPOTIFY_DATASET)
            logger.info(f"Dataset downloaded to: {path}")
            
            # Find the CSV file
            csv_path = Path(path) / KAGGLE_SPOTIFY_FILE
            if not csv_path.exists():
                # Try to find it dynamically
                csv_files = list(Path(path).glob("*.csv"))
                if csv_files:
                    csv_path = csv_files[0]
                    logger.info(f"Found CSV: {csv_path}")
                else:
                    raise FileNotFoundError(f"No CSV found in {path}")
            
            logger.info(f"Loading CSV from: {csv_path}")
            self.dataset = pd.read_csv(csv_path, low_memory=False)
            logger.info(f"Loaded dataset shape: {self.dataset.shape}")
            
            return self.dataset
            
        except ImportError:
            logger.error("kagglehub not installed. Install with: pip install kagglehub")
            raise
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def load_local_csv(self, csv_path):
        """Load from local CSV file"""
        logger.info(f"Loading from local CSV: {csv_path}")
        self.dataset = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Loaded dataset shape: {self.dataset.shape}")
        return self.dataset
    
    def create_phase_1_sample(self, size=PHASE_1_SAMPLE_SIZE, seed=PHASE_1_SEED):
        """Create Phase 1 sample (50K rows with stratified sampling)"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download_from_kaggle() first.")
        
        # Stratify by emotion if available
        if "emotion" in self.dataset.columns:
            logger.info(f"Stratified sampling by emotion...")
            self.sample = self.dataset.groupby("emotion", group_keys=False).apply(
                lambda x: x.sample(frac=size/len(self.dataset), random_state=seed)
            )
            self.sample = self.sample.sample(n=min(size, len(self.sample)), random_state=seed)
        else:
            logger.info(f"Random sampling {size} rows...")
            self.sample = self.dataset.sample(n=min(size, len(self.dataset)), random_state=seed)
        
        logger.info(f"Phase 1 sample shape: {self.sample.shape}")
        return self.sample
    
    def save_sample(self, output_path=SPOTIFY_RAW_PATH):
        """Save sample to CSV"""
        if self.sample is None:
            raise ValueError("Sample not created. Call create_phase_1_sample() first.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample.to_csv(output_path, index=False)
        logger.info(f"Sample saved to: {output_path}")
        return output_path
    
    def get_column_info(self):
        """Print dataset column information"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        
        logger.info("\n" + "="*80)
        logger.info("DATASET COLUMN INFORMATION")
        logger.info("="*80)
        logger.info(f"Total rows: {len(self.dataset)}")
        logger.info(f"Total columns: {len(self.dataset.columns)}")
        logger.info("\nColumns:")
        for col in self.dataset.columns:
            dtype = self.dataset[col].dtype
            missing = self.dataset[col].isna().sum()
            missing_pct = (missing / len(self.dataset)) * 100
            logger.info(f"  {col:30s} | dtype: {str(dtype):15s} | missing: {missing:7d} ({missing_pct:5.2f}%)")
        
        return self.dataset.dtypes, self.dataset.isnull().sum()


def main():
    """Main execution"""
    loader = SpotifyDataLoader()
    
    # Step 1: Download dataset
    logger.info("Pipeline Setup - PHASE 1: Data Loading")
    logger.info("=" * 80)
    
    try:
        loader.download_from_kaggle()
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Attempting to use cached dataset...")
        try:
            # Try to find cached dataset
            cache_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "devdope" / "900k-spotify" / "versions" / "3" / "spotify_dataset.csv"
            if cache_path.exists():
                loader.load_local_csv(cache_path)
            else:
                raise FileNotFoundError("No cached dataset found")
        except Exception as e2:
            logger.error(f"Failed to load cached dataset: {e2}")
            logger.error("Please download manually from Kaggle and provide path")
            return None
    
    # Step 2: Get column info
    loader.get_column_info()
    
    # Step 3: Create Phase 1 sample
    logger.info("\nCreating Phase 1 sample (50K rows)...")
    loader.create_phase_1_sample(size=PHASE_1_SAMPLE_SIZE, seed=PHASE_1_SEED)
    
    # Step 4: Save sample
    logger.info("\nSaving Phase 1 sample...")
    output_path = loader.save_sample()
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 1 DATA LOADING COMPLETE")
    logger.info("="*80)
    logger.info(f"Sample saved to: {output_path}")
    logger.info(f"Sample shape: {loader.sample.shape}")
    logger.info(f"Next step: Run data_processor.py to deduplicate and split data")
    
    return loader.sample


if __name__ == "__main__":
    main()

