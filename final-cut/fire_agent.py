#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire Analysis Agent - Complete Automated Workflow

Integrates fire analysis, prompt building, and LLM inference into a single agent.
Supports configurable LLM models and comprehensive result saving.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from interfaces import FireAgent
from analysis import FireAnalysis
from prompt_builder import FireAnalysisPromptBuilder
from utils.llm_utils import call_model
from config import FIRE_NAMES, LLM_CONFIG


@dataclass
class AgentConfig:
    """Configuration for Fire Analysis Agent"""
    model: str = LLM_CONFIG["default_model"]
    temperature: float = LLM_CONFIG["default_temperature"]
    max_retries: int = LLM_CONFIG["default_max_retries"]
    output_dir: str = "agent_outputs"
    verbose: bool = True


class FireAnalysisAgent(FireAgent):
    """
    Complete fire analysis agent with automated workflow.
    
    Orchestrates the end-to-end fire analysis pipeline including data collection,
    prompt generation, LLM inference, and result parsing/validation.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the fire analysis agent.
        
        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or AgentConfig()
        
        # Initialize components
        self.analyzer = FireAnalysis()
        self.prompt_builder = FireAnalysisPromptBuilder()
        
        # Create output directory for IO files
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def analyze_fire(
        self,
        fire_name: str,
        date_str: str,
        previous_analysis: Optional[Dict[str, Any]] = None,
        previous_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete fire analysis workflow.
        
        Executes the full pipeline: data analysis -> prompt building -> LLM inference
        -> response parsing -> result saving.
        
        Args:
            fire_name: Name of the fire (e.g., "CREEK")
            date_str: Date in YYYY-MM-DD format
            previous_analysis: Previous analysis result (for incremental mode)
            previous_summary: Previous analysis summary (for daily comparison)
            
        Returns:
            Dictionary containing workflow results, steps, and final result
        """
        # Initialize result structure
        workflow_result = self._init_workflow_result(fire_name, date_str)
        
        try:
            # Step 1: Fire Data Analysis
            analysis_result = self._execute_data_analysis(fire_name, date_str, workflow_result)
            if self._should_abort(workflow_result):
                return workflow_result
            
            # Calculate consecutive no-fire days
            self._calculate_consecutive_no_fire_days(analysis_result, previous_summary)
            
            # Step 2: Prompt Building
            prompt_result = self._execute_prompt_building(
                analysis_result, previous_analysis, previous_summary, workflow_result
            )
            if self._should_abort(workflow_result):
                return workflow_result
            
            # Step 3: LLM Inference
            llm_response = self._execute_llm_inference(prompt_result, workflow_result)
            if self._should_abort(workflow_result):
                return workflow_result
            
            # Step 4: Parse and Validate JSON Response
            parsed_result = self._execute_response_parsing(llm_response, workflow_result)
            if self._should_abort(workflow_result):
                return workflow_result
            
            workflow_result["final_result"] = parsed_result
            
            # Step 5: Save IO file
            io_filename = self._save_input_output_txt(
                fire_name, date_str, prompt_result, llm_response, parsed_result
            )
            self._record_step_success(workflow_result, "io_txt_saving", {"filename": io_filename})
            
            return workflow_result
            
        except Exception as e:
            error_msg = f"Workflow failed with unexpected error: {str(e)}"
            workflow_result["errors"].append(error_msg)
            return workflow_result
    
    def _should_abort(self, workflow_result: Dict[str, Any]) -> bool:
        """
        Check if workflow should abort due to errors.
        
        Args:
            workflow_result: Workflow result dictionary
            
        Returns:
            True if workflow should abort, False otherwise
        """
        return bool(workflow_result.get("errors"))
    
    def _init_workflow_result(self, fire_name: str, date_str: str) -> Dict[str, Any]:
        """
        Initialize workflow result structure.
        
        Args:
            fire_name: Fire name
            date_str: Date string
            
        Returns:
            Initial workflow result dictionary
        """
        return {
            "fire_name": fire_name,
            "analysis_date": date_str,
            "task_mode": "fire_analysis",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries
            },
            "workflow_steps": {},
            "final_result": None,
            "errors": []
        }
    
    def _execute_data_analysis(self, fire_name: str, date_str: str, 
                               workflow_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute fire data analysis step.
        
        Args:
            fire_name: Fire name
            date_str: Date string
            workflow_result: Workflow result to update
            
        Returns:
            Analysis result dictionary, or None if failed
        """
        try:
            analysis_result = self.analyzer.analyze_fire(fire_name, date_str)
            
            step_status = "completed" if not analysis_result.get("errors") else "failed"
            self._record_step_success(workflow_result, "data_analysis", {
                "status": step_status,
                "result": analysis_result
            })
            
            if "error" in analysis_result or analysis_result.get("errors"):
                error_msg = f"Data analysis failed: {analysis_result.get('error', analysis_result.get('errors'))}"
                self._record_step_error(workflow_result, "data_analysis", error_msg)
                return None
            
            return analysis_result
            
        except Exception as e:
            self._record_step_error(workflow_result, "data_analysis", str(e))
            return None
    
    def _execute_prompt_building(self, analysis_result: Dict[str, Any],
                                 previous_analysis: Optional[Dict[str, Any]],
                                 previous_summary: Optional[Dict[str, Any]],
                                 workflow_result: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Execute prompt building step.
        
        Args:
            analysis_result: Analysis result from previous step
            previous_analysis: Previous analysis (if any)
            previous_summary: Previous summary (if any)
            workflow_result: Workflow result to update
            
        Returns:
            Dictionary with system and user prompts, or None if failed
        """
        try:
            prompt_result = self.prompt_builder.build_prompt(
                summary=analysis_result.get("summary", {}),
                clusters_data=analysis_result.get("cluster_analysis", []),
                previous_analysis=previous_analysis,
                previous_summary=previous_summary
            )
            
            self._record_step_success(workflow_result, "prompt_building", {
                "status": "completed",
                "system_prompt_length": len(prompt_result.get("system", "")),
                "user_prompt_length": len(prompt_result.get("user", ""))
            })
            
            return prompt_result
            
        except Exception as e:
            self._record_step_error(workflow_result, "prompt_building", str(e))
            return None
    
    def _execute_llm_inference(self, prompt_result: Dict[str, str],
                               workflow_result: Dict[str, Any]) -> Optional[str]:
        """
        Execute LLM inference step.
        
        Args:
            prompt_result: Prompt dictionary with system and user messages
            workflow_result: Workflow result to update
            
        Returns:
            LLM response string, or None if failed
        """
        try:
            llm_response = call_model(
                model=self.config.model,
                system=prompt_result.get("system", ""),
                user=prompt_result.get("user", ""),
                temperature=self.config.temperature,
                max_retries=self.config.max_retries
            )
            
            if llm_response is None:
                self._record_step_error(workflow_result, "llm_inference", 
                                       "LLM inference failed - no response received")
                return None
            
            self._record_step_success(workflow_result, "llm_inference", {
                "status": "completed",
                "response_length": len(llm_response)
            })
            
            return llm_response
            
        except Exception as e:
            self._record_step_error(workflow_result, "llm_inference", str(e))
            return None
    
    def _execute_response_parsing(self, llm_response: str,
                                  workflow_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute response parsing and validation step.
        
        Args:
            llm_response: Raw LLM response string
            workflow_result: Workflow result to update
            
        Returns:
            Parsed result dictionary, or None if failed
        """
        try:
            parsed_result = self._parse_and_validate_response(llm_response)
            
            self._record_step_success(workflow_result, "response_parsing", {
                "status": "completed",
                "validation_passed": True
            })
            
            return parsed_result
            
        except Exception as e:
            error_data = {
                "status": "failed",
                "error": str(e),
                "raw_response": llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
            }
            workflow_result["workflow_steps"]["response_parsing"] = error_data
            workflow_result["errors"].append(f"Response parsing failed: {str(e)}")
            return None
    
    def _record_step_success(self, workflow_result: Dict[str, Any], 
                            step_name: str, step_data: Dict[str, Any]) -> None:
        """
        Record successful step execution.
        
        Args:
            workflow_result: Workflow result to update
            step_name: Name of the step
            step_data: Step data to record
        """
        workflow_result["workflow_steps"][step_name] = step_data
    
    def _record_step_error(self, workflow_result: Dict[str, Any], 
                          step_name: str, error_msg: str) -> None:
        """
        Record step execution error.
        
        Args:
            workflow_result: Workflow result to update
            step_name: Name of the step
            error_msg: Error message
        """
        workflow_result["errors"].append(f"{step_name} failed: {error_msg}")
        workflow_result["workflow_steps"][step_name] = {
            "status": "failed",
            "error": error_msg
        }
    
    def _parse_and_validate_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM JSON response.
        
        Performs the following validations:
        - Cleans markdown code blocks
        - Parses JSON
        - Validates required keys
        - Validates resource requirements
        - Validates indicator levels
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed and validated result dictionary
            
        Raises:
            ValueError: If response is invalid or validation fails
        """
        # Clean and parse JSON
        cleaned_response = self._clean_json_response(response)
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
        
        # Validate structure
        self._validate_required_keys(parsed)
        self._validate_resource_requirements(parsed)
        self._validate_indicators(parsed)
        
        return parsed
    
    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response by removing markdown code blocks.
        
        Args:
            response: Raw response string
            
        Returns:
            Cleaned response string
        """
        cleaned = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        return cleaned.strip()
    
    def _validate_required_keys(self, parsed: Dict[str, Any]) -> None:
        """
        Validate that all required keys are present.
        
        Args:
            parsed: Parsed JSON dictionary
            
        Raises:
            ValueError: If any required key is missing
        """
        required_keys = ["analysis_reasoning", "resource_requirements", 
                        "confidence", "intermediate_indicators"]
        
        for key in required_keys:
            if key not in parsed:
                raise ValueError(f"Missing required key: {key}")
    
    def _validate_resource_requirements(self, parsed: Dict[str, Any]) -> None:
        """
        Validate resource requirements structure.
        
        Args:
            parsed: Parsed JSON dictionary
            
        Raises:
            ValueError: If resource requirements are invalid
        """
        resource_req = parsed.get("resource_requirements", {})
        
        if "daily_personnel" not in resource_req:
            raise ValueError("Missing daily_personnel in resource_requirements")
        
        if "daily_budget" not in resource_req:
            raise ValueError("Missing daily_budget in resource_requirements")
    
    def _validate_indicators(self, parsed: Dict[str, Any]) -> None:
        """
        Validate intermediate indicators levels.
        
        Args:
            parsed: Parsed JSON dictionary
            
        Raises:
            ValueError: If any indicator level is invalid
        """
        indicators = parsed.get("intermediate_indicators", {})
        valid_levels = {"minimal", "low", "moderate", "high", "critical"}
        
        for indicator_name, level in indicators.items():
            if level not in valid_levels:
                raise ValueError(
                    f"Invalid indicator level '{level}' for {indicator_name}. "
                    f"Must be one of: {valid_levels}"
                )
    
    def _save_input_output_txt(
        self, 
        fire_name: str, 
        date_str: str, 
        prompt_result: Dict[str, str],
        llm_response: str,
        parsed_result: Dict[str, Any]
    ) -> str:
        """
        Save input/output in readable txt format.
        
        Creates a comprehensive log file containing prompts, raw response,
        parsed result, and summary statistics.
        
        Args:
            fire_name: Fire name
            date_str: Date string
            prompt_result: Dictionary with system and user prompts
            llm_response: Raw LLM response
            parsed_result: Parsed result dictionary
            
        Returns:
            Filename of saved file
        """
        filename = f"io_{fire_name}_{date_str.replace('-', '_')}.txt"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"Fire Analysis Input/Output - {fire_name} - {date_str}\n")
            f.write("=" * 60 + "\n\n")
            
            # System prompt
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(prompt_result.get("system", ""))
            f.write("\n\n")
            
            # User prompt
            f.write("=== USER PROMPT ===\n")
            f.write(prompt_result.get("user", ""))
            f.write("\n\n")
            
            # Raw response
            f.write("=== LLM RAW RESPONSE ===\n")
            f.write(llm_response)
            f.write("\n\n")
            
            # Parsed result
            f.write("=== PARSED RESULT ===\n")
            f.write(json.dumps(parsed_result, indent=2, ensure_ascii=False))
            f.write("\n\n")
            
            # Summary
            f.write("=== SUMMARY ===\n")
            self._write_summary_section(f, parsed_result)
        
        return filename
    
    def _write_summary_section(self, f, parsed_result: Dict[str, Any]) -> None:
        """
        Write summary section to output file.
        
        Args:
            f: File handle
            parsed_result: Parsed result dictionary
        """
        personnel = parsed_result.get("resource_requirements", {}).get("daily_personnel", {}).get("value", "N/A")
        budget = parsed_result.get("resource_requirements", {}).get("daily_budget", {}).get("value", "N/A")
        confidence = parsed_result.get("confidence", {}).get("score", "N/A")
        
        f.write(f"Required Personnel: {personnel} people\n")
        f.write(f"Estimated Budget: ${budget} USD\n")
        f.write(f"Confidence Score: {confidence}/5\n")
        
        # Intermediate indicators summary
        indicators = parsed_result.get("intermediate_indicators", {})
        f.write("\nIntermediate Indicators:\n")
        for indicator, level in indicators.items():
            f.write(f"  - {indicator}: {level}\n")
    
    def _calculate_consecutive_no_fire_days(self, analysis_result: Dict[str, Any], 
                                           previous_summary: Optional[Dict[str, Any]]) -> None:
        """
        Calculate consecutive no-fire days and add to analysis result.
        
        Updates the summary in-place with consecutive no-fire day count.
        
        Args:
            analysis_result: Current analysis result
            previous_summary: Previous day's summary (if available)
        """
        summary = analysis_result.get("summary", {})
        current_no_fire = summary.get("no_fire_points_today", False)
        
        consecutive_days = (
            (previous_summary.get("no_fire_consecutive_days", 0) + 1) 
            if current_no_fire and previous_summary else (1 if current_no_fire else 0)
        )
        
        summary["no_fire_consecutive_days"] = consecutive_days