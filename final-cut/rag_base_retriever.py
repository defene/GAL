#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base RAG Retriever - Common functionality for RAG retrievers

Provides shared implementation for cosine similarity computation,
fire name deduplication, and index loading/management.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from config import RAG_CONFIG, PROMPT_PLUGINS


class BaseRAGRetriever(ABC):
    """Base class for RAG retrievers with common functionality"""
    
    def __init__(self):
        self.index_matrix = None
        self.meta_data = None
        self.feature_mean = None
        self.feature_std = None
        self.feature_names = None
        self.vectorizer = None  # Should be set by subclass
    
    @abstractmethod
    def get_index_files(self, output_dir: str) -> tuple:
        """
        Get index and metadata file paths.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Tuple of (index_file_path, meta_file_path)
        """
        pass
    
    def build_corpus_index(self, corpus_path: str = None) -> bool:
        """
        Build or load corpus index.
        
        Args:
            corpus_path: Corpus path (for future use)
            
        Returns:
            True if successful
        """
        output_dir = RAG_CONFIG.get("output_dir", "rag_data/")
        index_file, meta_file = self.get_index_files(output_dir)
        
        if os.path.exists(index_file) and os.path.exists(meta_file):
            print(f"Loading pre-built index from {output_dir}")
            return self._load_index(index_file, meta_file)
        
        print(f"Pre-built index not found in {output_dir}")
        return False
    
    def _load_index(self, index_file: str, meta_file: str) -> bool:
        """
        Load pre-built index with NaN handling.
        
        Args:
            index_file: Index file path
            meta_file: Metadata file path
            
        Returns:
            True if successful
        """
        try:
            # Load index
            data = np.load(index_file)
            X = data['X']
            mean = data['mean']
            std = data['std']
            
            # Clean NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = np.where(std == 0, 1, std)
            
            self.index_matrix = X
            self.feature_mean = mean
            self.feature_std = std
            
            if 'feature_names' in data:
                self.feature_names = data['feature_names'].tolist()
            
            # Load metadata
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.meta_data = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def retrieve_topk(self, query_summary: Dict[str, Any], k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K similar samples.
        
        Args:
            query_summary: Query summary data
            k: Number of samples to return
            
        Returns:
            List of similar samples
        """
        if k is None:
            k = PROMPT_PLUGINS.get("rag", {}).get("top_k", 3)
        
        if self.index_matrix is None or self.meta_data is None:
            print("Index not loaded")
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.vectorize_summary(query_summary)
        if query_vector is None:
            return []
        
        # Clean and standardize
        query_vector = np.nan_to_num(query_vector, nan=0.0, posinf=0.0, neginf=0.0)
        mean = np.nan_to_num(self.feature_mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(self.feature_std, nan=1.0, posinf=1.0, neginf=1.0)
        std = np.where(std == 0, 1, std)
        
        query_normalized = (query_vector - mean) / std
        query_normalized = np.nan_to_num(query_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate similarity
        similarities = self._cosine_similarity(query_normalized, self.index_matrix)
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get sorted indices
        all_indices = np.argsort(similarities)[::-1]
        
        # Deduplicate if enabled
        unique_by_fire = RAG_CONFIG.get('unique_by_fire', True)
        
        if unique_by_fire:
            return self._deduplicate_by_fire_name(all_indices, similarities, k)
        else:
            results = []
            for idx in all_indices[:k]:
                meta = self.meta_data[idx].copy()
                meta['similarity'] = float(similarities[idx])
                results.append(meta)
            return results
    
    def _cosine_similarity(self, query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity with robust NaN handling.
        
        Args:
            query_vec: Query feature vector
            corpus_matrix: Corpus feature matrix
            
        Returns:
            Array of similarity scores
        """
        # Clean inputs
        query_vec = np.nan_to_num(query_vec, nan=0.0, posinf=0.0, neginf=0.0)
        corpus_matrix = np.nan_to_num(corpus_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize query
        query_norm = np.linalg.norm(query_vec)
        query_norm = max(query_norm, 1e-8)  # Avoid division by zero
        query_normalized = query_vec / query_norm
        
        # Normalize corpus
        corpus_norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        corpus_norms = np.where(corpus_norms < 1e-8, 1e-8, corpus_norms)
        corpus_normalized = corpus_matrix / corpus_norms
        
        # Calculate similarity
        similarities = np.dot(corpus_normalized, query_normalized)
        
        # Ensure valid range
        similarities = np.clip(similarities, -1.0, 1.0)
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
        
        return similarities
    
    def _deduplicate_by_fire_name(self, all_indices: np.ndarray, 
                                   similarities: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """
        Deduplicate by fire name, ensuring top-k from different fires.
        
        Args:
            all_indices: Sorted indices
            similarities: Similarity scores
            k: Target count
            
        Returns:
            Deduplicated results
        """
        seen_fires = set()
        results = []
        
        for idx in all_indices:
            if len(results) >= k:
                break
            
            meta = self.meta_data[idx]
            fire_name = meta.get('fire_name', 'Unknown')
            
            if fire_name in seen_fires:
                continue
            
            result_meta = meta.copy()
            result_meta['similarity'] = float(similarities[idx])
            results.append(result_meta)
            seen_fires.add(fire_name)
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns:
            List of feature names
        """
        if self.feature_names is not None:
            return self.feature_names
        if self.vectorizer and hasattr(self.vectorizer, 'get_feature_names'):
            return self.vectorizer.get_feature_names()
        return []
