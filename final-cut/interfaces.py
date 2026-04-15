#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Definition Module - Defines abstract base classes for core components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    @abstractmethod
    def build_prompt(
        self,
        summary: Dict[str, Any],
        clusters_data: List[Dict[str, Any]],
        previous_analysis: Optional[Dict[str, Any]] = None,
        previous_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Build analysis prompt."""
        pass


class FireAgent(ABC):
    """Abstract base class for fire agents."""
    pass

