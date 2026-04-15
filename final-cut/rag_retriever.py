#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility shim for older imports.

The standard retriever implementation lives in `rag_standard_retriever.py`,
but some scripts still import `RAGRetriever` from `rag_retriever.py`.
Re-exporting the class here keeps those entry points working without
changing the rest of the codebase.
"""

from rag_standard_retriever import RAGRetriever

__all__ = ["RAGRetriever"]
