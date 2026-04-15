#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Retriever - Numeric feature-based similarity retrieval

Implements cosine similarity retrieval based on summary numeric features, supporting:
- Feature vectorization (including date cyclic features)
- z-score standardization
- Cosine similarity computation
- Top-K retrieval
"""

import os
from typing import Dict, List, Any

from rag_base_retriever import BaseRAGRetriever
from rag_standard_vectorizer import StandardVectorizer


class RAGRetriever(BaseRAGRetriever):
    """RAG retriever for standard fire analysis similarity search."""
    
    def __init__(self):
        super().__init__()
        self.vectorizer = StandardVectorizer()
        
    def get_index_files(self, output_dir: str) -> tuple:
        """Get standard RAG index file paths."""
        index_file = os.path.join(output_dir, "rag_index.npz")
        meta_file = os.path.join(output_dir, "rag_meta.json")
        return index_file, meta_file
    
    
    def format_rag_context(self, results: List[Dict[str, Any]], current_mmdd: str = None) -> str:
        """Format RAG context into concise range format."""
        from utils.rag_format_utils import format_rag_range_context
        return format_rag_range_context(results)
