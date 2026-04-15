#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire Analysis Prompt Builder - Modular Implementation

A modular, configurable prompt builder that consumes pre-calculated analysis results.
Supports section-based customization, automatic RAG injection, and template switching.
"""

from typing import Any, Dict, List, Optional, Tuple

from interfaces import PromptBuilder
from config import PROMPT_SECTION_CONFIG
from prompt_templates import assemble_system_prompt
from prompt_sections import Context, build_user_prompt_sections
from rag_dual_provider import get_auto_rag_context


class FireAnalysisPromptBuilder(PromptBuilder):
    """
    Main prompt builder for fire analysis with modular architecture.
    
    Features:
    - Template-based prompt configuration
    - Automatic RAG context injection
    - Section-based user prompt assembly
    - Configurable historical context inclusion
    - Support for no-fire-point scenarios
    """
    
    def __init__(self):
        """Initialize the prompt builder with section configuration."""
        # Load section configuration
        self.section_config = PROMPT_SECTION_CONFIG
    
    def build_prompt(
        self,
        summary: Dict[str, Any],
        clusters_data: List[Dict[str, Any]],
        previous_analysis: Optional[Dict[str, Any]] = None,
        previous_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Build prompt for fire analysis using modular sections.
        
        Constructs both system and user prompts based on configurable templates.
        Automatically injects RAG context based on configuration.
        
        Args:
            summary: Summary data from fire analysis (with pre-calculated statistics)
            clusters_data: List of cluster analysis data
            previous_analysis: Previous analysis result (for historical context)
            previous_summary: Previous analysis summary (for daily comparison)
            
        Returns:
            Dictionary with "system" and "user" prompt strings
        """
        # Validate inputs - allow empty clusters if no fire points today
        if not self._validate_inputs(summary, clusters_data):
            return self._build_error_prompt()
        
        # Get RAG context based on configuration
        rag_context = get_auto_rag_context(summary)
        
        # Create context for section builders
        context = self._build_context(
            summary, clusters_data, previous_analysis, 
            previous_summary, rag_context
        )
        
        # Get template configuration
        system_config, user_config = self._get_template_config()
        
        # Build system prompt
        system_prompt = self._build_system_prompt(rag_context, system_config)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(context, user_config)
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _validate_inputs(self, summary: Dict[str, Any], 
                        clusters_data: List[Dict[str, Any]]) -> bool:
        """
        Validate input data for prompt building.
        
        Args:
            summary: Summary data
            clusters_data: Cluster data list
            
        Returns:
            True if inputs are valid, False otherwise
        """
        no_fire_points_today = summary.get("no_fire_points_today", False)
        return bool(clusters_data) or no_fire_points_today
    
    def _build_error_prompt(self) -> Dict[str, str]:
        """
        Build error prompt for invalid inputs.
        
        Returns:
            Dictionary with error messages for system and user prompts
        """
        return {
            "system": "No valid cluster data found.",
            "user": "Please provide valid fire analysis data."
        }
    
    def _build_context(self, summary: Dict[str, Any],
                      clusters_data: List[Dict[str, Any]],
                      previous_analysis: Optional[Dict[str, Any]],
                      previous_summary: Optional[Dict[str, Any]],
                      rag_context: Optional[str]) -> Context:
        """
        Create context object for section builders.
        
        Args:
            summary: Summary data
            clusters_data: Cluster data list
            previous_analysis: Previous analysis
            previous_summary: Previous summary
            rag_context: RAG context
            
        Returns:
            Context object for prompt building
        """
        return Context(
            summary=summary,
            clusters_data=clusters_data,
            previous_analysis=previous_analysis,
            previous_summary=previous_summary,
            rag_context=rag_context
        )
    
    def _get_template_config(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get system and user configuration from current template.
        
        Retrieves the active template configuration and extracts
        system and user prompt settings.
        
        Returns:
            Tuple of (system_config, user_config) dictionaries
        """
        current_template = self.section_config.get("current_template", "template1")
        template_config = self.section_config.get("templates", {}).get(current_template, {})
        fire_analysis_config = template_config.get("fire_analysis", {})
        
        system_config = fire_analysis_config.get("system", {})
        user_config = fire_analysis_config.get("user", {})
        
        return system_config, user_config
    
    def _build_system_prompt(self, rag_context: Optional[str], 
                            system_config: Dict[str, Any]) -> str:
        """
        Build system prompt with optional historical context.
        
        Args:
            rag_context: RAG context (if available)
            system_config: System configuration from template
            
        Returns:
            Assembled system prompt string
        """
        include_historical = self._should_include_historical_context(
            rag_context, 
            system_config.get("toggles", {})
        )
        
        system_order = system_config.get("order", None)
        return assemble_system_prompt("fire_analysis", include_historical, system_order)
    
    def _build_user_prompt(self, context: Context, 
                          user_config: Dict[str, Any]) -> str:
        """
        Build user prompt from configured sections.
        
        Args:
            context: Context object with all data
            user_config: User configuration from template
            
        Returns:
            Assembled user prompt string
        """
        user_sections = build_user_prompt_sections(
            context,
            user_config.get("order", []),
            user_config.get("toggles", {})
        )
        
        return "\n\n".join(user_sections)
    
    def _should_include_historical_context(self, rag_context: Optional[str], 
                                          toggles: Dict[str, Any]) -> bool:
        """
        Determine if historical context should be included in system prompt.
        
        Supports three modes:
        - True/False: Explicitly include/exclude
        - "auto": Include only if RAG context is available
        - Other: Treat as boolean
        
        Args:
            rag_context: RAG context string (if available)
            toggles: Toggle configuration dictionary
            
        Returns:
            True if historical context should be included
        """
        toggle_value = toggles.get("include_historical_guidelines", "auto")
        
        if isinstance(toggle_value, bool):
            return toggle_value
        elif toggle_value == "auto":
            return bool(rag_context and rag_context.strip())
        else:
            return bool(toggle_value)