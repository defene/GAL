#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Trend Retriever - Rolling Trend-based RAG Retriever

Specialized RAG retrieval for no-fire scenarios, using rolling trend features for similarity calculation.
Contains all historical data (fire + no-fire), but focuses on trend feature similarity.
"""

import os
from typing import Dict, List, Any
from pathlib import Path

from rag_base_retriever import BaseRAGRetriever
from rag_trend_vectorizer import TrendVectorizer


class TrendRAGRetriever(BaseRAGRetriever):
    """Trend RAG Retriever"""
    
    def __init__(self):
        super().__init__()
        self.vectorizer = TrendVectorizer()
        
    def get_index_files(self, output_dir: str) -> tuple:
        """Get trend RAG index file paths."""
        index_file = os.path.join(output_dir, "rag_trend_index.npz")
        meta_file = os.path.join(output_dir, "rag_trend_meta.json")
        return index_file, meta_file
    
    
    def format_rag_context(self, results: List[Dict[str, Any]], current_mmdd: str = None, 
                          query_no_fire: bool = False) -> str:
        """Format trend RAG context into concise range format."""
        from utils.rag_format_utils import format_rag_range_context
        return format_rag_range_context(results)
    
    def save_index(self, output_dir: str = None) -> bool:
        """Save trend index to files."""
        import numpy as np
        import json
        
        if self.index_matrix is None or self.meta_data is None:
            print("No trend index to save")
            return False
        
        if output_dir is None:
            from config import RAG_CONFIG
            output_dir = RAG_CONFIG.get("output_dir", "rag_data/")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            index_file, meta_file = self.get_index_files(output_dir)
            
            save_data = {
                'X': self.index_matrix,
                'mean': self.feature_mean,
                'std': self.feature_std
            }
            if self.feature_names:
                save_data['feature_names'] = np.array(self.feature_names)
            
            np.savez_compressed(index_file, **save_data)
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.meta_data, f, indent=2, ensure_ascii=False)
            
            print(f"Trend index saved to: {output_dir}")
            return True
            
        except Exception as e:
            print(f"Failed to save trend index: {e}")
            return False
