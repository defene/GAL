#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Templates - System Prompt Components

Contains all system prompt template blocks that can be independently modified
and combined based on configuration.
"""

from typing import Dict, List
import json


# =============================================================================
# System Prompt Components
# =============================================================================

# System role definition
SYSTEM_ROLE = """You are a wildfire analysis and resource management expert. You must return ONLY a valid JSON object following the exact schema provided below."""

# Global Guidelines
GLOBAL_GUIDELINES = """### Global Guidelines
- The task is to estimate TODAY's required daily_personnel and daily_budget.
- reasoning must explain how terrain, weather, fire intensity, population exposure, and resource accessibility shape your judgment, considering both current conditions and previous analysis context.
- daily_personnel is the total integer headcount assigned today (all crews/engines/aviation modules plus command/overhead/support).
- daily_budget is the **new cost incurred today only**, in USD."""

# Simplified Global Guidelines (for template1)
SIMPLIFIED_GLOBAL_GUIDELINES = """### Global Guidelines
Estimate TODAY's required `daily_personnel` and `daily_budget`."""

# Resource Estimation Principles
RESOURCE_ESTIMATION_PRINCIPLES = """### Resource Estimation Principles
- If the fire surges, remember resources are finite—do not assume cost and personnel can scale proportionally.
- When the fire eases, non-suppression needs persist (patrol, mop-up, rehab, logistics); budget and staffing may still be required.
- In "stable" periods, account for cumulative costs and crew fatigue—budgets and crews are not unlimited.
- No detected hotspots ≠ full extinguishment; avoid indiscriminate cuts and maintain a prudent baseline.
- Weigh these trade-offs and produce a balanced, defensible recommendation for today's personnel and today's spend. Include any key assumptions and risks.
- **Common pitfall**: after you've committed resources and the fire is "under control" but not yet stable, that actually signals under-resourcing—maintain or increase resources until true stability is confirmed."""

# Analysis Approach
ANALYSIS_APPROACH = """### Analysis Approach
- Analyze the fire situation holistically, considering today's conditions and changes from the previous analysis.
- Provide updated estimates for required daily_personnel and daily_budget based on your professional judgment."""

# Core Principles (for template1)
CORE_PRINCIPLES = """### Core Principles
- Integrate **terrain, weather, fire intensity, population exposure, and resource accessibility** jointly; do not rely on a single metric (e.g., FRP).
- Resources are finite; increases in personnel/budget cannot scale proportionally to fire surges.
- Global max indicators: personnel/cost typically rise in following days' rolling averages, not immediately on the same day.
- When fire eases, account for ongoing non-suppression needs (patrol, mop-up, rehab, logistics).
- In stable periods, consider cumulative cost and fatigue—budgets/crews are not unlimited.
- Few/no hotspots ≠ extinguishment; maintain a prudent baseline.
- Common mistake: cutting resources once the fire "appears" controlled but not yet stable; stability must be confirmed before reducing."""

# RAG Guidance (for template1)
RAG_GUIDANCE = """### RAG Guidance
- Use provided RAG examples to set **soft lower/upper bounds** for personnel and budget."""

# Historical Context Analysis (for template2)
HISTORICAL_CONTEXT = """### Historical Context Analysis
- The concrete RAG examples are strong guides for setting soft lower/upper bounds on resource allocation (personnel and today's spend); treat them as flexible ranges and reference them when specifying bounds.
- Do not deviate too far from these examples; if it conflicts with verified facts or higher-quality evidence, discard it and state why."""

# Format Rules
FORMAT_RULES = """### Additional Format Rules
- All numbers must be plain digits with no commas or underscores; units must match exactly. daily_budget is rounded to the nearest USD (integer).
- For "intermediate_indicators", each field value MUST be exactly one of: "minimal", "low", "moderate", "high", "critical" (lowercase; no other text)."""


# =============================================================================
# Output Schema Components
# =============================================================================

# Output schema definition
OUTPUT_SCHEMA_TEMPLATE = {
    "analysis_reasoning": {
        "situation_comparison": "<2-3 sentences comparing today vs yesterday>",
        "personnel_reasoning": "<2-3 sentences explaining daily_personnel changes>",
        "budget_reasoning": "<2-3 sentences explaining daily_budget changes>",
        "overall_reasoning": "<2-3 sentences with overall change assessment>"
    },
    "resource_requirements": {
        "daily_personnel": {"value": "<integer>", "unit": "people"},
        "daily_budget": {"value": "<integer>", "unit": "USD"}
    },
    "confidence": {
        "score": "<1-5 integer>"
    },
    "intermediate_indicators": {
        "spread_containment_difficulty": "<minimal|low|moderate|high|critical>",
        "resource_access_deployment": "<minimal|low|moderate|high|critical>",
        "weather_escalation_risk": "<minimal|low|moderate|high|critical>",
        "terrain_operational_complexity": "<minimal|low|moderate|high|critical>",
        "population_exposure_density": "<minimal|low|moderate|high|critical>",
        "fire_station_coverage": "<minimal|low|moderate|high|critical>"
    }
}


def build_output_schema() -> str:
    """
    Build formatted output schema string for system prompt.
    
    Converts the OUTPUT_SCHEMA_TEMPLATE dictionary to a formatted JSON string
    with markdown header for inclusion in system prompts.
    
    Returns:
        Formatted output schema string with header and JSON template
    """
    schema_str = json.dumps(OUTPUT_SCHEMA_TEMPLATE, indent=2)
    return f"""### Output Schema (STRICT JSON; no extra keys; no comments)
{schema_str}"""


# =============================================================================
# Template Assembly Functions
# =============================================================================

# Component mapping for system prompt assembly
COMPONENT_MAP = {
    "role": SYSTEM_ROLE,
    "global_guidelines": GLOBAL_GUIDELINES,
    "simplified_global_guidelines": SIMPLIFIED_GLOBAL_GUIDELINES,
    "resource_principles": RESOURCE_ESTIMATION_PRINCIPLES,
    "analysis_approach": ANALYSIS_APPROACH,
    "core_principles": CORE_PRINCIPLES,
    "rag_guidance": RAG_GUIDANCE,
    "historical_context": HISTORICAL_CONTEXT,
    "format_rules": FORMAT_RULES,
}


def assemble_system_prompt(task_mode: str = "fire_analysis", include_historical: bool = False, order: List[str] = None) -> str:
    """
    Assemble complete system prompt from components
    
    Args:
        task_mode: Task mode (kept for backward compatibility, currently unused)
        include_historical: Whether to include historical context section
        order: List of component keys to assemble in order
    
    Returns:
        Complete system prompt string
    """
    # Get order from config if not provided
    if order is None:
        from config import PROMPT_SECTION_CONFIG
        current_template = PROMPT_SECTION_CONFIG.get("current_template", "template1")
        template_config = PROMPT_SECTION_CONFIG.get("templates", {}).get(current_template, {})
        fire_analysis_config = template_config.get("fire_analysis", {})
        system_config = fire_analysis_config.get("system", {})
        order = system_config.get("order", [])
        
        if not order:
            # Fallback to default order for template2
            order = [
                "role",
                "global_guidelines",
                "resource_principles",
                "analysis_approach",
                "historical_context?",
                "output_schema",
                "format_rules"
            ]
    
    # Assemble components
    sections = []
    for key in order:
        # Remove optional marker if present
        clean_key = key.rstrip('?')
        is_optional = key.endswith('?')
        
        # Skip optional historical context if not included
        if clean_key == "historical_context" and is_optional and not include_historical:
            continue
        
        # Handle output_schema specially
        if clean_key == "output_schema":
            sections.append(build_output_schema())
        elif clean_key in COMPONENT_MAP:
            sections.append(COMPONENT_MAP[clean_key])
    
    return "\n\n".join(sections)
