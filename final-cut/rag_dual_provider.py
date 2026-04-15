#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Provider - Unified RAG system provider

Smart router that selects appropriate RAG strategy based on configuration and fire point status:
- With fire points: Use standard RAG (based on instant features)
- No fire points: Use trend RAG (based on rolling trend features)
"""

from typing import Dict, Any, Optional

from rag_standard_retriever import RAGRetriever
from rag_trend_retriever import TrendRAGRetriever
from config import PROMPT_PLUGINS


class RagProvider:
    """Unified RAG context provider supporting both standard and trend strategies"""
    
    def __init__(self, enable_dual: bool = True):
        """
        Initialize RAG provider
        
        Args:
            enable_dual: Enable dual RAG mode (standard + trend). False means standard RAG only.
        """
        self.enable_dual = enable_dual
        self.standard_retriever = None
        self.trend_retriever = None
        self._standard_initialized = False
        self._trend_initialized = False
    
    def _ensure_standard_initialized(self) -> bool:
        """
        Ensure standard RAG retriever is initialized.
        
        Lazy initialization pattern - only initializes on first use.
        
        Returns:
            True if retriever is available, False if initialization failed
        """
        if self._standard_initialized:
            return self.standard_retriever is not None
        
        try:
            self.standard_retriever = RAGRetriever()
            success = self.standard_retriever.build_corpus_index()
            self._standard_initialized = True
            return success
        except Exception as e:
            print(f"Failed to initialize standard RAG retriever: {e}")
            self._standard_initialized = True
            self.standard_retriever = None
            return False
    
    def _ensure_trend_initialized(self) -> bool:
        """
        Ensure trend RAG retriever is initialized.
        
        Lazy initialization pattern - only initializes on first use.
        
        Returns:
            True if retriever is available, False if initialization failed
        """
        if self._trend_initialized:
            return self.trend_retriever is not None
        
        try:
            self.trend_retriever = TrendRAGRetriever()
            success = self.trend_retriever.build_corpus_index()
            self._trend_initialized = True
            return success
        except Exception as e:
            print(f"Failed to initialize trend RAG retriever: {e}")
            self._trend_initialized = True
            self.trend_retriever = None
            return False
    
    def get_context(self, summary: Dict[str, Any], k: Optional[int] = None) -> Optional[str]:
        """
        Intelligently get RAG context based on configuration and fire point status
        
        Args:
            summary: Fire analysis summary data
            k: Number of similar samples to retrieve, None uses default
            
        Returns:
            Formatted RAG context string, None if failed
        """
        try:
            # Use default value from config
            if k is None:
                k = PROMPT_PLUGINS.get("rag", {}).get("top_k", 3)
            
            # Select RAG strategy
            use_trend = self.enable_dual and summary.get("no_fire_points_today", False)
            
            if use_trend:
                return self._get_trend_context(summary, k)
            else:
                return self._get_standard_context(summary, k)
            
        except Exception as e:
            print(f"RAG context generation failed: {e}")
            return None
    
    def _get_standard_context(self, summary: Dict[str, Any], k: int) -> Optional[str]:
        """
        Use standard RAG to get context (instant features-based).
        
        Args:
            summary: Fire analysis summary data
            k: Number of similar samples to retrieve
            
        Returns:
            Formatted RAG context string, None if failed
        """
        if not self._ensure_standard_initialized():
            return None
        
        try:
            results = self.standard_retriever.retrieve_topk(summary, k=k)
            if not results:
                return None
            
            current_mmdd = summary.get('analysis_mmdd')
            rag_context = self.standard_retriever.format_rag_context(results, current_mmdd=current_mmdd)
            
            return rag_context if rag_context.strip() else None
            
        except Exception as e:
            print(f"Standard RAG context generation failed: {e}")
            return None
    
    def _get_trend_context(self, summary: Dict[str, Any], k: int) -> Optional[str]:
        """
        Use trend RAG to get context (rolling trend features-based).
        
        Args:
            summary: Fire analysis summary data
            k: Number of similar samples to retrieve
            
        Returns:
            Formatted RAG context string, None if failed
        """
        if not self._ensure_trend_initialized():
            return None
        
        try:
            results = self.trend_retriever.retrieve_topk(summary, k=k)
            
            if not results:
                return None
            
            current_mmdd = summary.get('analysis_mmdd')
            no_fire_today = summary.get("no_fire_points_today", False)
            
            rag_context = self.trend_retriever.format_rag_context(
                results, 
                current_mmdd=current_mmdd,
                query_no_fire=no_fire_today
            )
            
            return rag_context if rag_context and rag_context.strip() else None
            
        except Exception as e:
            print(f"Trend RAG context generation failed: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get RAG system status information.
        
        Returns:
            Dictionary containing:
                - dual_mode_enabled: Whether dual RAG mode is enabled
                - standard_rag: Standard RAG status (initialized, available, samples)
                - trend_rag: Trend RAG status (initialized, available, samples)
        """
        return {
            "dual_mode_enabled": self.enable_dual,
            "standard_rag": {
                "initialized": self._standard_initialized,
                "available": self.standard_retriever is not None,
                "samples": len(self.standard_retriever.meta_data) if self.standard_retriever and self.standard_retriever.meta_data else 0
            },
            "trend_rag": {
                "initialized": self._trend_initialized,
                "available": self.trend_retriever is not None,
                "samples": len(self.trend_retriever.meta_data) if self.trend_retriever and self.trend_retriever.meta_data else 0
            }
        }


def get_auto_rag_context(summary: Dict[str, Any]) -> Optional[str]:
    """
    Get RAG context based on configuration (for prompt_builder use)
    
    Create provider and get context if RAG is enabled in configuration.
    
    Args:
        summary: Fire analysis summary data
        
    Returns:
        RAG context string, None if failed or disabled
    """
    # Check configuration
    rag_config = PROMPT_PLUGINS.get("rag", {})
    
    # Check if enabled
    if not rag_config.get("enabled", False):
        return None
    
    # Create provider and get context
    provider_type = rag_config.get("provider", "dual")
    enable_dual = (provider_type == "dual")
    
    provider = RagProvider(enable_dual=enable_dual)
    
    try:
        top_k = rag_config.get("top_k", 3)
        return provider.get_context(summary, k=top_k)
    except Exception as e:
        fail_mode = rag_config.get("fail_mode", "silent")
        if fail_mode == "error":
            raise RuntimeError(f"RAG context generation failed: {e}")
        else:
            print(f"RAG context generation failed: {e}")
            return None