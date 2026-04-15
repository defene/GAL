#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend RAG Index Builder

Builds trend RAG index from existing fire analysis JSON files with GT data.
Supports loading Ground Truth data from CSV files and associating with analysis results.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import json
import glob
import numpy as np

from rag_trend_retriever import TrendRAGRetriever
from config import RAG_SPLIT
from utils.llm_utils import safe_extract_number
from utils.date_utils import extract_mmdd_from_date


class TrendRAGIndexBuilder:
    """Trend RAG Index Builder"""
    
    def __init__(self, output_dir: str = "rag_data"):
        self.output_dir = output_dir
        self.retriever = TrendRAGRetriever()
        
    def build_index_with_gt(self, 
                           analysis_glob: str = "trend_analysis_data/fire_analysis_*.json",
                           gt_dir: str = "final_data_cleaned",
                           min_valid_features: int = 5) -> bool:
        """
        Build trend RAG index with Ground Truth data.
        
        Args:
            analysis_glob: Glob pattern for analysis result JSON files
            gt_dir: Ground Truth CSV files directory
            min_valid_features: Minimum number of valid features threshold
            
        Returns:
            Whether index was built successfully
        """
        print("Building Trend RAG Index with Ground Truth data...")
        
        # 1. Load GT data
        print("Loading Ground Truth data...")
        gt_data_map = self._load_gt_data(gt_dir)
        if not gt_data_map:
            print("No Ground Truth data loaded")
            return False
        
        print(f"Loaded GT data for {len(gt_data_map)} records")
        
        # 2. Build vector index (with GT data)
        print("Building vector index...")
        success = self._build_vector_index_with_gt(
            analysis_glob, gt_data_map, min_valid_features
        )
        
        if success:
            # 3. Save index
            print("Saving trend index...")
            save_success = self.retriever.save_index(self.output_dir)
            if save_success:
                print(f"Trend RAG index built successfully!")
                print(f"   Output directory: {self.output_dir}")
                return True
            else:
                print("Failed to save trend index")
                return False
        else:
            print("Failed to build vector index")
            return False
    
    def _load_gt_data(self, gt_dir: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Load Ground Truth data from CSV files.
        
        Args:
            gt_dir: GT CSV files directory
            
        Returns:
            GT data mapping {(fire_name, mmdd): {GT_columns}}
        """
        gt_data_map = {}
        gt_dir_path = Path(gt_dir)
        
        if not gt_dir_path.exists():
            print(f"GT directory not found: {gt_dir}")
            return gt_data_map
        
        # Iterate through all GT CSV files
        for csv_file in gt_dir_path.glob("*_gt.csv"):
            file_gt_data = self._safe_process_gt_file(csv_file)
            gt_data_map.update(file_gt_data)
        
        return gt_data_map
    
    def _safe_process_gt_file(self, csv_file: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Safely process a single Ground Truth CSV file.
        
        Args:
            csv_file: CSV file path
            
        Returns:
            GT data mapping dictionary
        """
        gt_data = {}
        
        try:
            # Extract fire name from filename
            fire_name = csv_file.stem.replace("_gt", "")
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            print(f"   Processing {fire_name}: {len(df)} records")
            
            # Process each row of GT data
            success_count = 0
            for _, row in df.iterrows():
                result = self._safe_process_gt_row(row, fire_name)
                if result:
                    gt_key, gt_record = result
                    gt_data[gt_key] = gt_record
                    success_count += 1
            
            if success_count < len(df):
                print(f"      Warning: {len(df) - success_count} rows skipped due to errors")
                
        except Exception as e:
            print(f"   Error reading {csv_file}: {e}")
        
        return gt_data
    
    def _safe_process_gt_row(self, row: pd.Series, fire_name: str) -> Optional[Tuple[Tuple[str, str], Dict[str, Any]]]:
        """
        Safely process a single Ground Truth data row.
        
        Args:
            row: DataFrame row data
            fire_name: Fire name
            
        Returns:
            ((fire_name, mmdd), gt_record) on success, None on failure
        """
        try:
            # Extract date
            report_date = str(row.get('REPORT_FROM_DATE', row.get('report_date', '')))
            mmdd = extract_mmdd_from_date(report_date)
            if not mmdd:
                return None
            
            # Build GT record
            gt_record = {
                'TOTAL_PERSONNEL': safe_extract_number(row.get('TOTAL_PERSONNEL'), default=None, as_int=True),
                'EST_IM_COST_TO_DATE_FIXED_DAILY': safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED_DAILY'), default=None),
                'EST_IM_COST_TO_DATE_FIXED': safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED'), default=None)
            }
            
            gt_key = (fire_name, mmdd)
            return (gt_key, gt_record)
            
        except Exception as e:
            return None
    
    def _build_vector_index_with_gt(self, 
                                   analysis_glob: str,
                                   gt_data_map: Dict[Tuple[str, str], Dict[str, Any]],
                                   min_valid_features: int) -> bool:
        """
        Build vector index with GT data.
        
        Args:
            analysis_glob: Analysis files glob pattern
            gt_data_map: Ground Truth data mapping
            min_valid_features: Minimum valid features threshold
            
        Returns:
            Whether build was successful
        """
        # Scan analysis result files
        json_files = glob.glob(analysis_glob)
        if not json_files:
            print(f"No analysis files found matching: {analysis_glob}")
            return False
        
        print(f"Found {len(json_files)} analysis files")
        
        # Extract features and metadata
        features, meta_data = self._extract_features_and_metadata(
            json_files, gt_data_map, min_valid_features
        )
        
        if not features:
            print("No valid trend features extracted")
            return False
        
        # Build and normalize index matrix
        self._build_index_matrix(features, meta_data)
        
        return True
    
    def _extract_features_and_metadata(self,
                                      json_files: List[str],
                                      gt_data_map: Dict[Tuple[str, str], Dict[str, Any]],
                                      min_valid_features: int) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Extract features and metadata from JSON files.
        
        Args:
            json_files: List of JSON file paths
            gt_data_map: Ground Truth data mapping
            min_valid_features: Minimum valid features threshold
            
        Returns:
            Tuple of (features list, metadata list)
        """
        features = []
        meta_data = []
        
        rag_fires = set(RAG_SPLIT.get("rag_fires", []))
        test_fires = set(RAG_SPLIT.get("test_fires", []))
        
        for json_file in json_files:
            result = self._process_analysis_file(
                json_file, gt_data_map, test_fires, min_valid_features
            )
            
            if result:
                feature_vector, meta_entry = result
                features.append(feature_vector)
                meta_data.append(meta_entry)
        
        return features, meta_data
    
    def _process_analysis_file(self,
                               json_file: str,
                               gt_data_map: Dict[Tuple[str, str], Dict[str, Any]],
                               test_fires: set,
                               min_valid_features: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process a single analysis JSON file.
        
        Args:
            json_file: JSON file path
            gt_data_map: Ground Truth data mapping
            test_fires: Set of test fire names to exclude
            min_valid_features: Minimum valid features threshold
            
        Returns:
            (feature_vector, meta_entry) on success, None on failure or skip
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            if not summary:
                return None
            
            fire_name = summary.get('fire_name', 'Unknown')
            analysis_date = summary.get('analysis_date', 'Unknown')
            analysis_mmdd = summary.get('analysis_mmdd', analysis_date[5:] if len(analysis_date) >= 10 else 'Unknown')
            
            # Check if fire is in RAG training set
            if fire_name in test_fires:
                print(f"   Skipping test fire: {fire_name}")
                return None
            
            # Trend RAG includes all data (with fire points + no fire points)
            # Vectorize (using trend vectorizer)
            feature_vector = self.retriever.vectorizer.vectorize_summary(summary)
            if feature_vector is None:
                return None
            
            # Check valid features count
            valid_features = np.sum(feature_vector != 0)
            if valid_features < min_valid_features:
                print(f"   Skipping {fire_name} {analysis_mmdd}: insufficient features ({valid_features})")
                return None
            
            # Extract basic metadata
            no_fire_today = summary.get('no_fire_points_today', False)
            fire_overview = summary.get('fire_overview', {})
            total_fire_points = fire_overview.get('total_fire_points', 0)
            
            # Find corresponding GT data
            gt_key = (fire_name, analysis_mmdd)
            gt_data = gt_data_map.get(gt_key, {})
            
            meta_entry = {
                'fire_name': fire_name,
                'date': analysis_date,
                'mmdd': analysis_mmdd,
                'source_file': json_file,
                'no_fire_today': no_fire_today,
                'total_fire_points': total_fire_points,
                'TOTAL_PERSONNEL': gt_data.get('TOTAL_PERSONNEL'),
                'EST_IM_COST_TO_DATE_FIXED_DAILY': gt_data.get('EST_IM_COST_TO_DATE_FIXED_DAILY'),
                'EST_IM_COST_TO_DATE_FIXED': gt_data.get('EST_IM_COST_TO_DATE_FIXED')
            }
            
            return (feature_vector, meta_entry)
            
        except Exception as e:
            print(f"Failed to process {json_file}: {e}")
            return None
    
    def _build_index_matrix(self, features: List[np.ndarray], meta_data: List[Dict[str, Any]]) -> None:
        """
        Build and normalize index matrix from features.
        
        Args:
            features: List of feature vectors
            meta_data: List of metadata entries
        """
        # Convert to matrix and standardize
        feature_matrix = np.array(features)
        self.retriever.feature_mean = np.mean(feature_matrix, axis=0)
        self.retriever.feature_std = np.std(feature_matrix, axis=0)
        
        # Avoid division by zero
        self.retriever.feature_std = np.where(self.retriever.feature_std == 0, 1, self.retriever.feature_std)
        
        # Standardize
        self.retriever.index_matrix = (feature_matrix - self.retriever.feature_mean) / self.retriever.feature_std
        self.retriever.meta_data = meta_data
        
        # Statistics
        no_fire_count = sum(1 for m in meta_data if m.get('no_fire_today', False))
        fire_count = len(meta_data) - no_fire_count
        gt_count = sum(1 for m in meta_data if m.get('TOTAL_PERSONNEL') is not None)
        
        print(f"Built trend index with {len(meta_data)} samples and {feature_matrix.shape[1]} features")
        print(f"   - Fire samples: {fire_count}")
        print(f"   - No-fire samples: {no_fire_count}")
        print(f"   - With GT data: {gt_count}")


def main():
    """Main function"""
    print("Trend RAG Index Builder")
    print("=" * 40)
    
    # Create index builder
    builder = TrendRAGIndexBuilder()
    
    # Ensure output directory exists
    Path(builder.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build index
    success = builder.build_index_with_gt(
        analysis_glob="trend_analysis_data/fire_analysis_*.json",
        gt_dir="final_data_cleaned",
        min_valid_features=5
    )
    
    if success:
        print("\nTrend RAG index building completed successfully!")
        print(f"Index files saved in: {builder.output_dir}/")
        print("   - rag_trend_index.npz (vector index)")
        print("   - rag_trend_meta.json (metadata)")
    else:
        print("\nFailed to build trend RAG index")
        return 1
    
    # Test loading
    print("\nTesting index loading...")
    from rag_trend_retriever import TrendRAGRetriever
    test_retriever = TrendRAGRetriever()
    if test_retriever.build_corpus_index():
        print("Index loading test passed")
        
        # Display statistics
        status = {
            "total_samples": len(test_retriever.meta_data),
            "no_fire_samples": sum(1 for m in test_retriever.meta_data if m.get('no_fire_today', False)),
            "fire_samples": sum(1 for m in test_retriever.meta_data if not m.get('no_fire_today', False)),
            "with_gt": sum(1 for m in test_retriever.meta_data if m.get('TOTAL_PERSONNEL') is not None)
        }
        
        print("Index Statistics:")
        print(f"   Total samples: {status['total_samples']}")
        print(f"   Fire samples: {status['fire_samples']}")
        print(f"   No-fire samples: {status['no_fire_samples']}")
        print(f"   With GT data: {status['with_gt']}")
        
    else:
        print("Index loading test failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())