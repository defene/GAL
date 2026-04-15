#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Corpus Builder

Batch process analysis.FireAnalysis from final_data_cleaned/*.csv to generate
analysis results, build standardized vector index and metadata, support data
splitting (rag_fires vs test_fires).
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import argparse
from datetime import datetime

from config import RAG_CONFIG, RAG_SPLIT, FIRE_NAMES
from analysis import FireAnalysis
from utils.llm_utils import safe_extract_number
from utils.fire_name_utils import normalize_fire_name
from utils.date_utils import extract_mmdd_from_date
from rag_retriever import RAGRetriever
from rag_trend_retriever import TrendRAGRetriever


class RAGCorpusBuilder:
    """RAG corpus builder for standard fire analysis index."""
    
    def __init__(self):
        self.analyzer = FireAnalysis()
        self.retriever = RAGRetriever()  # Reuse vectorization logic
    
    def build_corpus(self, 
                    fires: Optional[List[str]] = None,
                    output_dir: str = None,
                    skip_existing_json: bool = True,
                    min_valid_features: int = 0) -> bool:
        """
        Build RAG corpus from ground truth data.
        
        Args:
            fires: List of fire names to process, None means use RAG_SPLIT.rag_fires
            output_dir: Output directory path
            skip_existing_json: Whether to skip existing JSON files
            min_valid_features: Minimum valid feature count threshold
            
        Returns:
            True if corpus built successfully, False otherwise
        """
        if fires is None:
            fires = RAG_SPLIT.get('rag_fires', [])
        
        if output_dir is None:
            output_dir = RAG_CONFIG.get('output_dir', 'rag_data/')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Building RAG corpus for {len(fires)} fires")
        print(f"Output directory: {output_dir}")
        
        # Validate data split
        if not self._validate_data_split():
            return False
        
        # Collect all (fire_name, date) pairs to process
        fire_date_pairs = []
        gt_data_map = {}  # (fire_name, mmdd) -> GT data
        
        for fire_csv_name in fires:
            csv_file = f"final_data_cleaned/{fire_csv_name}_gt.csv"
            if not os.path.exists(csv_file):
                print(f"Warning: GT file not found: {csv_file}")
                continue
            
            # Read GT data
            try:
                df = pd.read_csv(csv_file)
                print(f"Processing {csv_file}: {len(df)} records")
                
                for _, row in df.iterrows():
                    report_date = row['REPORT_FROM_DATE']
                    if pd.isna(report_date):
                        continue
                    
                    # Normalize fire name
                    standard_fire_name = normalize_fire_name(fire_csv_name)
                    
                    # Extract GT data
                    gt_data = {
                        'TOTAL_PERSONNEL': safe_extract_number(row.get('TOTAL_PERSONNEL'), default=None, as_int=True),
                        'EST_IM_COST_TO_DATE_FIXED_DAILY': safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED_DAILY'), default=None),
                        'EST_IM_COST_TO_DATE_FIXED': safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED'), default=None)
                    }
                    
                    # Calculate mmdd
                    try:
                        date_obj = pd.to_datetime(report_date)
                        mmdd = date_obj.strftime('%m-%d')
                        date_str = date_obj.strftime('%Y-%m-%d')
                    except:
                        print(f"Warning: Invalid date format: {report_date}")
                        continue
                    
                    fire_date_pairs.append((standard_fire_name, date_str))
                    gt_data_map[(standard_fire_name, mmdd)] = gt_data
                    
            except Exception as e:
                print(f"Failed to process {csv_file}: {e}")
                continue
        
        print(f"Found {len(fire_date_pairs)} fire-date pairs to process")
        
        # Generate analysis results
        analysis_results = []
        for fire_name, date_str in fire_date_pairs:
            json_filename = f"fire_analysis_{fire_name}_{date_str}.json"
            
            if skip_existing_json and os.path.exists(json_filename):
                print(f"Skipping existing: {json_filename}")
                # Load existing file
                try:
                    with open(json_filename, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    analysis_results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to load existing {json_filename}: {e}")
                continue
            
            print(f"Analyzing {fire_name} on {date_str}")
            
            # Run analysis
            try:
                result = self.analyzer.analyze_fire(fire_name, date_str)
                
                if "error" in result or result.get("errors"):
                    print(f"Analysis failed for {fire_name} {date_str}: {result.get('error', result.get('errors'))}")
                    continue
                
                analysis_results.append(result)
                print(f"Analyzed: {fire_name} {date_str}")
                
            except Exception as e:
                print(f"Failed to analyze {fire_name} {date_str}: {e}")
                continue
        
        print(f"Successfully processed {len(analysis_results)} analyses")
        
        # Build vector index
        return self._build_vector_index(analysis_results, gt_data_map, output_dir, min_valid_features)
    
    def _validate_data_split(self) -> bool:
        """
        Validate data split configuration.
        
        Returns:
            True if validation passed, False if overlap found in strict mode
        """
        rag_fires = set(RAG_SPLIT.get('rag_fires', []))
        test_fires = set(RAG_SPLIT.get('test_fires', []))
        
        if RAG_SPLIT.get('strict_no_overlap', True):
            overlap = rag_fires & test_fires
            if overlap:
                print(f"Data split validation failed: overlap found: {overlap}")
                return False
        
        print(f"Data split validation passed: {len(rag_fires)} RAG fires, {len(test_fires)} test fires")
        return True
    
    def _build_vector_index(self, 
                           analysis_results: List[Dict[str, Any]], 
                           gt_data_map: Dict[Tuple[str, str], Dict[str, Any]],
                           output_dir: str,
                           min_valid_features: int) -> bool:
        """
        Build vector index from analysis results.
        
        Args:
            analysis_results: List of analysis result dictionaries
            gt_data_map: Mapping from (fire_name, mmdd) to GT data
            output_dir: Output directory path
            min_valid_features: Minimum valid feature count threshold
            
        Returns:
            True if index built successfully, False otherwise
        """
        print("Building vector index...")
        
        features = []
        meta_data = []
        
        for result in analysis_results:
            summary = result.get('summary', {})
            if not summary:
                continue
            
            # Skip dates with no fire points (total_fire_points <= 0)
            fire_overview = summary.get('fire_overview', {})
            if float(fire_overview.get('total_fire_points', 0) or 0) <= 0:
                continue

            # Vectorize
            feature_vector = self.retriever.vectorize_summary(summary)
            if feature_vector is None:
                continue
            
            # Check valid feature count
            valid_features = np.sum(feature_vector != 0)
            if valid_features < min_valid_features:
                continue
            
            features.append(feature_vector)
            
            # Build metadata
            fire_name = summary.get('fire_name', 'Unknown')
            analysis_date = summary.get('analysis_date', 'Unknown')
            analysis_mmdd = summary.get('analysis_mmdd', analysis_date[5:] if len(analysis_date) >= 10 else 'Unknown')
            
            # Find corresponding GT data
            gt_key = (fire_name, analysis_mmdd)
            gt_data = gt_data_map.get(gt_key, {})
            
            meta_entry = {
                'fire_name': fire_name,
                'date': analysis_date,
                'mmdd': analysis_mmdd,
                'source_file': f"fire_analysis_{fire_name}_{analysis_date}.json",
                'TOTAL_PERSONNEL': gt_data.get('TOTAL_PERSONNEL'),
                'EST_IM_COST_TO_DATE_FIXED_DAILY': gt_data.get('EST_IM_COST_TO_DATE_FIXED_DAILY'),
                'EST_IM_COST_TO_DATE_FIXED': gt_data.get('EST_IM_COST_TO_DATE_FIXED')
            }
            
            meta_data.append(meta_entry)
        
        if not features:
            print("No valid features extracted")
            return False
        
        # Convert to matrix and standardize
        feature_matrix = np.array(features)
        feature_mean = np.mean(feature_matrix, axis=0)
        feature_std = np.std(feature_matrix, axis=0)
        
        # Avoid division by zero
        feature_std = np.where(feature_std == 0, 1, feature_std)
        
        # Standardize
        normalized_matrix = (feature_matrix - feature_mean) / feature_std
        
        # Save index
        index_file = os.path.join(output_dir, "rag_index.npz")
        np.savez_compressed(
            index_file,
            X=normalized_matrix,
            mean=feature_mean,
            std=feature_std,
            feature_names=self.retriever.get_feature_names()
        )
        
        # Save metadata
        meta_file = os.path.join(output_dir, "rag_meta.json")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Built vector index:")
        print(f"   {len(meta_data)} samples")
        print(f"   {feature_matrix.shape[1]} features")
        print(f"   Saved to {index_file}")
        print(f"   Metadata saved to {meta_file}")
        
        return True
    
    def list_available_fires(self) -> Dict[str, List[str]]:
        """
        List available fires and their dates from GT CSV files.
        
        Returns:
            Dictionary mapping fire names to list of available dates
        """
        available = {}
        
        csv_files = glob.glob("final_data_cleaned/*_gt.csv")
        for csv_file in csv_files:
            basename = os.path.basename(csv_file)
            fire_name = normalize_fire_name(basename)
            
            try:
                df = pd.read_csv(csv_file)
                dates = []
                for _, row in df.iterrows():
                    report_date = row['REPORT_FROM_DATE']
                    if not pd.isna(report_date):
                        try:
                            date_obj = pd.to_datetime(report_date)
                            dates.append(date_obj.strftime('%Y-%m-%d'))
                        except:
                            continue
                
                available[fire_name] = sorted(dates)
                
            except Exception as e:
                print(f"Warning: Failed to read {csv_file}: {e}")
        
        return available


class TrendRAGCorpusBuilder:
    """
    Trend RAG corpus builder for building rag_trend_index.

    - Only indexes training set fires (RAG_SPLIT.rag_fires)
    - Supports building directly from generated trend JSONs (e.g., trend_analysis_data/fire_analysis_*.json)
    - Includes no-fire days (relies on trend features for similarity retrieval)
    - Uses TrendRAGRetriever's trend vectorizer
    """

    def __init__(self):
        self.retriever = TrendRAGRetriever()

    def _load_gt_map(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Load GT mapping from CSV files.
        
        Returns:
            Dictionary mapping (fire_name, mmdd) to GT data columns
        """
        gt_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        csv_files = glob.glob("final_data_cleaned/*_gt.csv")

        for csv_file in csv_files:
            try:
                fire_name = normalize_fire_name(os.path.basename(csv_file))
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    report_date = str(row.get("REPORT_FROM_DATE", ""))
                    if not report_date or len(report_date) < 10:
                        continue
                    mmdd = report_date[5:]
                    gt_map[(fire_name, mmdd)] = {
                        'TOTAL_PERSONNEL': safe_extract_number(row.get('TOTAL_PERSONNEL'), default=None, as_int=True),
                        'EST_IM_COST_TO_DATE_FIXED_DAILY': safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED_DAILY'), default=None),
                        'EST_IM_COST_TO_DATE_FIXED': safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED'), default=None),
                    }
            except Exception as e:
                print(f"Warning: Failed to read GT CSV {csv_file}: {e}")
                continue
        return gt_map

    def build_trend_index_from_jsons(
        self,
        corpus_glob: str = "trend_analysis_data/fire_analysis_*.json",
        output_dir: str = None,
        min_valid_features: int = 5,
    ) -> bool:
        """
        Build rag_trend_index from trend JSON files.

        Only includes RAG_SPLIT.rag_fires (training set fires); includes no-fire days;
        feature vectorization uses TrendRAGRetriever's TrendVectorizer.
        
        Args:
            corpus_glob: Glob pattern for trend JSON corpus files
            output_dir: Output directory path
            min_valid_features: Minimum non-zero feature count threshold
            
        Returns:
            True if index built successfully, False otherwise
        """
        if output_dir is None:
            output_dir = RAG_CONFIG.get('output_dir', 'rag_data/')

        os.makedirs(output_dir, exist_ok=True)

        # Validate data split
        rag_fires = set(RAG_SPLIT.get('rag_fires', []))
        test_fires = set(RAG_SPLIT.get('test_fires', []))
        if RAG_SPLIT.get('strict_no_overlap', True) and (rag_fires & test_fires):
            print(f"Data split invalid: overlap between rag/test fires")
            return False

        print(f"Building TREND RAG index from: {corpus_glob}")
        json_files = glob.glob(corpus_glob)
        if not json_files:
            print(f"No trend JSON files found: {corpus_glob}")
            return False
        print(f"Found {len(json_files)} JSONs")

        gt_map = self._load_gt_map()

        features: List[np.ndarray] = []
        meta_data: List[Dict[str, Any]] = []

        kept, skipped = 0, 0
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                summary = data.get('summary', {})
                if not summary:
                    skipped += 1
                    continue

                fire_name = summary.get('fire_name') or 'Unknown'
                if fire_name not in rag_fires:
                    # Only index training set fires
                    skipped += 1
                    continue

                analysis_date = summary.get('analysis_date', 'Unknown')
                mmdd = summary.get('analysis_mmdd', analysis_date[5:] if len(analysis_date) >= 10 else 'Unknown')

                # Trend vectorization (includes no-fire days)
                vec = self.retriever.vectorizer.vectorize_summary(summary)
                if vec is None:
                    skipped += 1
                    continue

                valid_features = int(np.sum(vec != 0))
                if valid_features < min_valid_features:
                    skipped += 1
                    continue

                features.append(vec)

                gt = gt_map.get((fire_name, mmdd), {})
                meta_data.append({
                    'fire_name': fire_name,
                    'date': analysis_date,
                    'mmdd': mmdd,
                    'source_file': jf,
                    'no_fire_today': bool(summary.get('no_fire_points_today', False)),
                    'total_fire_points': summary.get('fire_overview', {}).get('total_fire_points', 0),
                    'TOTAL_PERSONNEL': gt.get('TOTAL_PERSONNEL'),
                    'EST_IM_COST_TO_DATE_FIXED_DAILY': gt.get('EST_IM_COST_TO_DATE_FIXED_DAILY'),
                    'EST_IM_COST_TO_DATE_FIXED': gt.get('EST_IM_COST_TO_DATE_FIXED'),
                })
                kept += 1

            except Exception as e:
                print(f"Warning: Failed to process {jf}: {e}")
                skipped += 1
                continue

        if not features:
            print("No valid trend features extracted")
            return False

        X = np.vstack(features).astype(np.float32)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1, std)
        Xn = (X - mean) / std

        # Save index
        index_file = os.path.join(output_dir, "rag_trend_index.npz")
        np.savez_compressed(index_file, X=Xn, mean=mean, std=std)

        meta_file = os.path.join(output_dir, "rag_trend_meta.json")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        print(f"Built TREND index: {kept} kept, {skipped} skipped, dim={X.shape[1]}")
        print(f"   {index_file}")
        print(f"   {meta_file}")
        return True

def main():
    """Command-line entry point for RAG corpus building."""
    parser = argparse.ArgumentParser(description="Build RAG corpus from GT data")
    parser.add_argument("--fires", nargs="*", help="Fire names to process (default: use RAG_SPLIT.rag_fires)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip existing JSON files")
    parser.add_argument("--min-features", type=int, default=0, help="Minimum valid features threshold")
    parser.add_argument("--list-fires", action="store_true", help="List available fires and exit")
    # Trend index options
    parser.add_argument("--build-trend", action="store_true", help="Build TREND RAG index from trend JSONs")
    parser.add_argument("--trend-glob", default="trend_analysis_data/fire_analysis_*.json", help="Glob pattern for trend JSON corpus")
    parser.add_argument("--trend-min-features", type=int, default=5, help="Minimum non-zero features for trend vectors")
    
    args = parser.parse_args()
    
    builder = RAGCorpusBuilder()
    
    if args.list_fires:
        available = builder.list_available_fires()
        print("Available fires:")
        for fire_name, dates in available.items():
            print(f"   {fire_name}: {len(dates)} dates ({dates[0]} to {dates[-1]})")
        return
    
    if args.build_trend:
        trend_builder = TrendRAGCorpusBuilder()
        success = trend_builder.build_trend_index_from_jsons(
            corpus_glob=args.trend_glob,
            output_dir=args.output_dir or RAG_CONFIG.get('output_dir', 'rag_data/'),
            min_valid_features=args.trend_min_features,
        )
        if success:
            print("TREND RAG index built successfully!")
            return
        else:
            print("Failed to build TREND RAG index")
            exit(1)

    # Default: build standard corpus
    success = builder.build_corpus(
        fires=args.fires,
        output_dir=args.output_dir,
        skip_existing_json=args.skip_existing,
        min_valid_features=args.min_features
    )
    if success:
        print("RAG corpus built successfully!")
    else:
        print("Failed to build RAG corpus")
        exit(1)


if __name__ == "__main__":
    main()
