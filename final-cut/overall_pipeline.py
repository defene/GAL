#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall Pipeline - Batch Fire Analysis Pipeline

Based on fires specified in config.RUN_FIRES, reads corresponding GT CSV files,
executes initial discovery and incremental analysis tasks in chronological order,
generates overall CSV and comparison charts.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import glob

from config import RUN_FIRES, OUTPUT_ROOT, LLM_CONFIG, FIRE_CLUSTERING, RAG_CONFIG, COMPARISON_CONFIG, ROLLING_CONFIG
from fire_agent import FireAnalysisAgent, AgentConfig
from utils.llm_utils import safe_extract_number
from utils.rolling import compute_rolling_metrics, format_rolling_for_llm
from utils.fire_name_utils import normalize_fire_name


class OverallPipeline:
    """Batch Fire Analysis Pipeline"""
    
    def __init__(self):
        # Initialize unit system from config before any processing
        from utils.unit_converter import sync_with_config
        sync_with_config()
        
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_root = os.path.join(OUTPUT_ROOT, self.timestamp)
        
        # Create output directory
        os.makedirs(self.output_root, exist_ok=True)
        
        print(f"Overall Pipeline initialized")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Output root: {self.output_root}")
        print(f"   Fires to process: {RUN_FIRES}")
    
    
    
    def run(self):
        """Run complete pipeline"""
        print(f"\nStarting overall pipeline for {len(RUN_FIRES)} fires...")
        
        # Save run configuration
        self._save_run_config()
        
        # Process each fire
        all_results = []
        
        for fire_name in RUN_FIRES:
            print(f"\nProcessing fire: {fire_name}")
            fire_results = self._process_fire(fire_name)
            if fire_results:
                all_results.extend(fire_results)
        
        # Generate overall CSV
        if all_results:
            overall_df = pd.DataFrame(all_results)
            overall_path = os.path.join(self.output_root, "overall.csv")
            overall_df.to_csv(overall_path, index=False)
            print(f"\nOverall results saved to: {overall_path}")
            print(f"   Total records: {len(overall_df)}")
        else:
            print("\nWarning: No results generated")
        
        print(f"\nPipeline completed! Results in: {self.output_root}")
    
    def _save_run_config(self):
        """Save run configuration snapshot"""
        config_snapshot = {
            "timestamp": self.timestamp,
            "run_fires": RUN_FIRES,
            "output_root": OUTPUT_ROOT,
            "llm_config": {
                "model": LLM_CONFIG["default_model"],
                "temperature": LLM_CONFIG["default_temperature"],
                "max_retries": LLM_CONFIG["default_max_retries"]
            },
            "fire_clustering": FIRE_CLUSTERING,
            "rag_config": RAG_CONFIG
        }
        
        config_path = os.path.join(self.output_root, "run_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        
        print(f"Run config saved to: {config_path}")
    
    def _process_fire(self, fire_name: str) -> List[Dict[str, Any]]:
        """Process single fire"""
        # Read GT data
        csv_file = f"final_data_cleaned/{fire_name}_gt.csv"
        if not os.path.exists(csv_file):
            print(f"GT file not found: {csv_file}")
            return []
        
        try:
            gt_df = pd.read_csv(csv_file)
            print(f"Loaded GT data: {len(gt_df)} records")
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
            return []
        
        # Create fire output directory and images subdirectory
        fire_output_dir = os.path.join(self.output_root, fire_name)
        images_output_dir = os.path.join(fire_output_dir, "images")
        os.makedirs(fire_output_dir, exist_ok=True)
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Sort by date for chronological processing
        gt_df['REPORT_FROM_DATE'] = pd.to_datetime(gt_df['REPORT_FROM_DATE'])
        gt_df = gt_df.sort_values('REPORT_FROM_DATE')
        
        fire_results = []
        previous_analysis = None
        previous_summary = None  # Previous day's summary (for comparison)
        pred_cost_cum = 0.0  # Predicted cumulative cost
        
        # Initialize cumulative variables
        self.pred_personnel_cum = 0.0  # Cumulative personnel-days
        self.fire_start_date = None  # Fire start date
        
        # Initialize rolling calculation history buffer
        fire_history_buffer = []
        
        for idx, row in gt_df.iterrows():
            date_str = row['REPORT_FROM_DATE'].strftime('%Y-%m-%d')
            incident_name = row['INCIDENT_NAME']
            
            # Normalize fire name
            normalized_fire_name = normalize_fire_name(incident_name)
            
            print(f"\nProcessing {incident_name} on {date_str} (normalized: {normalized_fire_name})")
            
            # Use previous day's analysis results (fully chronological)
            
            # Create agent configuration (unified fire analysis mode)
            agent_config = AgentConfig(
                model=LLM_CONFIG["default_model"],
                temperature=LLM_CONFIG["default_temperature"],
                max_retries=LLM_CONFIG["default_max_retries"],
                output_dir=fire_output_dir,
                verbose=True
            )
            
            # Initialize rolling metrics (prevent undefined in exception cases)
            rolling_metrics = {}
            
            # Run analysis
            try:
                agent = FireAnalysisAgent(agent_config)
                result = agent.analyze_fire(
                    fire_name=normalized_fire_name,
                    date_str=date_str,
                    previous_analysis=previous_analysis,
                    previous_summary=previous_summary
                )
                
                # Check if successful
                if result.get("errors") or not result.get("final_result"):
                    print(f"Analysis failed for {date_str}: {result.get('errors', 'No final result')}")
                    continue
                
                final_result = result["final_result"]
                
                # Extract analysis data (simplified path)
                analysis_result = result.get("workflow_steps", {}).get("data_analysis", {}).get("result", {})
                summary = analysis_result.get("summary", {})
                has_fire_points = not summary.get("no_fire_points_today", False)
                
                # Extract predicted values (needed before building cumulative cost info)
                resource_req = final_result.get("resource_requirements", {})
                pred_personnel = safe_extract_number(
                    resource_req.get("daily_personnel", {}).get("value"), 
                    default=0, 
                    as_int=True
                )
                pred_cost_daily = safe_extract_number(
                    resource_req.get("daily_budget", {}).get("value"), 
                    default=0.0, 
                    as_int=False
                )
                
                # Accumulate predicted cost (calculate before building cumulative info)
                previous_day_cumulative = pred_cost_cum
                pred_cost_cum += pred_cost_daily
                
                # Calculate cumulative personnel-days
                self.pred_personnel_cum += pred_personnel
                
                # Calculate days since fire start
                if self.fire_start_date is None:
                    self.fire_start_date = date_str
                from datetime import datetime
                start_date = datetime.strptime(self.fire_start_date, '%Y-%m-%d')
                current_date = datetime.strptime(date_str, '%Y-%m-%d')
                days_since_start = (current_date - start_date).days + 1
                
                # Build cumulative context information
                cumulative_context_info = self._build_cumulative_context_info(
                    fire_results, pred_cost_daily, pred_personnel, 
                    pred_cost_cum, self.pred_personnel_cum, days_since_start
                )
                
                # Extract confidence
                confidence = final_result.get("confidence", {})
                confidence_score = safe_extract_number(confidence.get("score"), default=None)
                
                # Extract actual values
                gt_personnel = safe_extract_number(row.get('TOTAL_PERSONNEL'), default=0, as_int=True)
                gt_cost_daily = safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED_DAILY'), default=0.0)
                gt_cost_cum = safe_extract_number(row.get('EST_IM_COST_TO_DATE_FIXED'), default=0.0)
                
                # Extract intermediate indicators
                indicators = final_result.get("intermediate_indicators", {})
                
                # Extract current day fire metrics for rolling calculation (directly from summary, avoid traversing cluster_analysis)
                fire_overview = summary.get("fire_overview", {})
                current_fire_values = {
                    'fire_points': fire_overview.get('total_fire_points', 0),
                    'area_total': fire_overview.get('total_area_m2', 0.0),
                    'area_max': fire_overview.get('max_cluster_area_m2', 0.0)  # Now directly available in summary
                }
                
                # Calculate rolling metrics
                rolling_metrics = compute_rolling_metrics(fire_history_buffer, current_fire_values)
                
                # Update history buffer
                fire_history_buffer.append(current_fire_values)
                
                # Update previous_analysis and previous_summary to current results (chronological order)
                # Extend previous_analysis to include cumulative cost info and cumulative context info
                enhanced_previous_analysis = final_result.copy()
                enhanced_previous_analysis["cumulative_cost_info"] = {
                    "total_cumulative_cost": pred_cost_cum,
                    "daily_cost": pred_cost_daily,
                    "previous_day_cumulative": previous_day_cumulative
                }
                enhanced_previous_analysis["cumulative_context_info"] = cumulative_context_info
                
                # Add rolling fire metrics to summary for LLM use
                if "fire_overview" not in summary:
                    summary["fire_overview"] = {}
                summary["fire_overview"]["rolling"] = format_rolling_for_llm(rolling_metrics)
                
                previous_analysis = enhanced_previous_analysis
                previous_summary = summary
                
                # Build result record
                record = {
                    'fire_name': incident_name,
                    'date': date_str,
                    'task_mode': 'fire_analysis',
                    'has_fire_points': has_fire_points,
                    'confidence_score': confidence_score,
                    'pred_personnel': pred_personnel,
                    'pred_cost_daily': pred_cost_daily,
                    'pred_cost_cum': pred_cost_cum,
                    'gt_personnel': gt_personnel,
                    'gt_cost_daily': gt_cost_daily,
                    'gt_cost_cum': gt_cost_cum,
                    # Fire point data
                    'fire_points_count': current_fire_values.get('fire_points', 0),
                    'total_area_m2': current_fire_values.get('area_total', 0.0),
                    'max_cluster_area_m2': current_fire_values.get('area_max', 0.0),
                    # six intermediate indicators (categorical labels)
                    'spread_containment_difficulty': indicators.get('spread_containment_difficulty'),
                    'resource_access_deployment': indicators.get('resource_access_deployment'),
                    'weather_escalation_risk': indicators.get('weather_escalation_risk'),
                    'terrain_operational_complexity': indicators.get('terrain_operational_complexity'),
                    'population_exposure_density': indicators.get('population_exposure_density'),
                    'fire_station_coverage': indicators.get('fire_station_coverage')
                }
                
                # Add rolling metrics to record
                record.update(rolling_metrics)
                
                fire_results.append(record)
                print(f"{date_str}: Personnel={pred_personnel}, Daily=${pred_cost_daily}, Cum=${pred_cost_cum:.0f}, HasFire={has_fire_points}")
                
            except Exception as e:
                print(f"Exception processing {date_str}: {e}")
                continue
        
        # Generate fire-level summary and charts
        if fire_results:
            self._save_fire_summary(fire_name, fire_results, fire_output_dir)
            self._generate_fire_plots(fire_name, fire_results, images_output_dir)
        
        return fire_results
    
    def _save_fire_summary(self, fire_name: str, results: List[Dict[str, Any]], output_dir: str):
        """Save fire-level summary CSV."""
        df = pd.DataFrame(results)
        summary_path = os.path.join(output_dir, f"{fire_name}_daily_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Fire summary saved to: {summary_path}")
    
    def _generate_fire_plots(self, fire_name: str, results: List[Dict[str, Any]], output_dir: str):
        """Generate fire-level comparison charts."""
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        
        # Set chart style
        plt.style.use('default')
        fig_size = (12, 6)
        
        # 1. Personnel comparison chart (with fire points count background bar chart)
        fig, ax1 = plt.subplots(figsize=fig_size)
        
        # Fire points count bar chart as background
        ax2 = ax1.twinx()
        bars = ax2.bar(df['date'], df['fire_points_count'], alpha=0.3, color='orange', 
                      label='Fire Points Count', width=0.8)
        ax2.set_ylabel('Fire Points Count', color='orange', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Personnel comparison line chart
        ax1.plot(df['date'], df['pred_personnel'], 'b-o', label='Predicted Personnel', linewidth=2, markersize=4)
        ax1.plot(df['date'], df['gt_personnel'], 'r-s', label='Actual Personnel', linewidth=2, markersize=4)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Personnel Count', fontsize=12)
        ax1.tick_params(axis='y')
        
        # Set title and grid
        plt.title(f'{fire_name} - Personnel Comparison with Fire Points Background', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        personnel_path = os.path.join(output_dir, f"{fire_name}_personnel.png")
        plt.savefig(personnel_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Personnel plot saved to: {personnel_path}")
        
        # 2. Daily cost comparison chart (with fire points count background bar chart)
        fig, ax1 = plt.subplots(figsize=fig_size)
        
        # Fire points count bar chart as background
        ax2 = ax1.twinx()
        bars = ax2.bar(df['date'], df['fire_points_count'], alpha=0.3, color='orange', 
                      label='Fire Points Count', width=0.8)
        ax2.set_ylabel('Fire Points Count', color='orange', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Daily cost comparison line chart
        ax1.plot(df['date'], df['pred_cost_daily'], 'b-o', label='Predicted Daily Cost', linewidth=2, markersize=4)
        ax1.plot(df['date'], df['gt_cost_daily'], 'r-s', label='Actual Daily Cost', linewidth=2, markersize=4)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Daily Cost (USD)', fontsize=12)
        ax1.tick_params(axis='y')
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Set title and grid
        plt.title(f'{fire_name} - Daily Cost Comparison with Fire Points Background', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        daily_cost_path = os.path.join(output_dir, f"{fire_name}_cost_daily.png")
        plt.savefig(daily_cost_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Daily cost plot saved to: {daily_cost_path}")
        
        # 3. Cumulative cost comparison chart (with fire points count background bar chart)
        fig, ax1 = plt.subplots(figsize=fig_size)
        
        # Fire points count bar chart as background
        ax2 = ax1.twinx()
        bars = ax2.bar(df['date'], df['fire_points_count'], alpha=0.3, color='orange', 
                      label='Fire Points Count', width=0.8)
        ax2.set_ylabel('Fire Points Count', color='orange', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Cumulative cost comparison line chart
        ax1.plot(df['date'], df['pred_cost_cum'], 'b-o', label='Predicted Cumulative Cost', linewidth=2, markersize=4)
        ax1.plot(df['date'], df['gt_cost_cum'], 'r-s', label='Actual Cumulative Cost', linewidth=2, markersize=4)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Cost (USD)', fontsize=12)
        ax1.tick_params(axis='y')
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Set title and grid
        plt.title(f'{fire_name} - Cumulative Cost Comparison with Fire Points Background', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        cum_cost_path = os.path.join(output_dir, f"{fire_name}_cost_cum.png")
        plt.savefig(cum_cost_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cumulative cost plot saved to: {cum_cost_path}")
    
    def _build_cumulative_context_info(self, fire_results: List[Dict[str, Any]], 
                                     current_daily_cost: float, current_daily_personnel: int,
                                     total_cumulative_cost: float, total_cumulative_personnel_days: float,
                                     days_since_start: int) -> Dict[str, Any]:
        """Build cumulative context information."""
        from config import CUMULATIVE_CONFIG
        
        if not CUMULATIVE_CONFIG.get("enabled", True):
            return {}
        
        cumulative_info = {
            "total_cumulative_cost": total_cumulative_cost,
            "total_cumulative_personnel_days": total_cumulative_personnel_days,
            "days_since_fire_start": days_since_start
        }
        
        # Calculate rolling averages
        rolling_windows = CUMULATIVE_CONFIG.get("rolling_windows", [3, 7])
        rolling_stats = {}
        
        for window in rolling_windows:
            window_key = f"{window}day"
            
            # Get most recent N days of data
            recent_results = fire_results[-window:] if len(fire_results) >= window else fire_results
            
            if recent_results:
                # Calculate averages
                avg_daily_cost = sum(r.get('pred_cost_daily', 0) for r in recent_results) / len(recent_results)
                avg_daily_personnel = sum(r.get('pred_personnel', 0) for r in recent_results) / len(recent_results)
                
                rolling_stats[window_key] = {
                    "avg_daily_cost": avg_daily_cost,
                    "avg_daily_personnel": avg_daily_personnel,
                    "sample_size": len(recent_results)
                }
        
        cumulative_info["rolling_stats"] = rolling_stats
        
        # Calculate trends (simple increase/decrease/stable determination)
        if len(fire_results) >= 3:
            recent_costs = [r.get('pred_cost_daily', 0) for r in fire_results[-3:]]
            recent_personnel = [r.get('pred_personnel', 0) for r in fire_results[-3:]]
            
            # Simple trend determination: compare average of recent 3 days with earlier average
            if len(fire_results) >= 6:
                earlier_costs = [r.get('pred_cost_daily', 0) for r in fire_results[-6:-3]]
                earlier_personnel = [r.get('pred_personnel', 0) for r in fire_results[-6:-3]]
                
                recent_avg_cost = sum(recent_costs) / len(recent_costs)
                earlier_avg_cost = sum(earlier_costs) / len(earlier_costs)
                recent_avg_personnel = sum(recent_personnel) / len(recent_personnel)
                earlier_avg_personnel = sum(earlier_personnel) / len(earlier_personnel)
                
                cost_trend = "increasing" if recent_avg_cost > earlier_avg_cost * 1.1 else \
                           "decreasing" if recent_avg_cost < earlier_avg_cost * 0.9 else "stable"
                personnel_trend = "increasing" if recent_avg_personnel > earlier_avg_personnel * 1.1 else \
                                "decreasing" if recent_avg_personnel < earlier_avg_personnel * 0.9 else "stable"
                
                cumulative_info["trends"] = {
                    "cost_trend": cost_trend,
                    "personnel_trend": personnel_trend
                }
        
        return cumulative_info


def main():
    """Main function."""
    pipeline = OverallPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
