#!/usr/bin/env python3
"""
Comprehensive analysis script for MOA training results and performance comparison.

This script analyzes:
- Model parameter (weights) progression throughout training
- Score progression and convergence patterns
- Performance comparison between MOA and single LLM
- Time efficiency analysis
- Different evaluation metrics comparison
- Detailed test case performance analysis
- Problem-level performance breakdown
- Failure pattern analysis
- Visualization of training dynamics and final results

Usage:
    python analyze_training_results.py [options]
    
Options:
    --base-dir PATH              Base directory containing the project (default: current directory)
    --training-output PATH       Path to training_output directory
    --checkpoint-dir PATH        Path to specific checkpoint directory
    --final-model PATH           Path to final_model.json file
    --training-history PATH      Path to training_history.json file
    --comparison-results PATH    Path to comparison_results.json file
    --test-cases-dir PATH        Path to test_cases directory
    --output-dir PATH            Directory to save output plots (default: current directory)
    --output-prefix PREFIX       Prefix for output filenames (default: empty)
    --separate-plots             Create separate files for each plot instead of dashboard style
    --no-display                 Do not display plots (useful for headless environments)
    --quiet                      Reduce verbosity of output
    --help, -h                   Show this help message
    
Examples:
    # Use default paths relative to current directory
    python analyze_training_results.py
    
    # Specify custom base directory
    python analyze_training_results.py --base-dir /path/to/project
    
    # Specify individual file paths
    python analyze_training_results.py --final-model /path/to/final_model.json \\
                                       --comparison-results /path/to/comparison_results.json
    
    # Save outputs with custom prefix and directory
    python analyze_training_results.py --output-dir ./results --output-prefix experiment1_
    
    # Create separate files for each plot instead of dashboard style
    python analyze_training_results.py --separate-plots
    
    # Use corrected cumulative analysis for better training insights
    python analyze_training_results.py --separate-plots --output-dir ./analysis_results
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class MOATrainingAnalyzer:
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 training_output_dir: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 final_model_path: Optional[str] = None,
                 training_history_path: Optional[str] = None,
                 comparison_results_path: Optional[str] = None,
                 test_cases_dir: Optional[str] = None,
                 output_dir: str = ".",
                 output_prefix: str = "",
                 show_plots: bool = True,
                 quiet: bool = False,
                 separate_plots: bool = False):
        
        # Set up directory paths
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.output_dir = Path(output_dir)
        self.output_prefix = output_prefix
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up data file paths with fallbacks to default locations
        if training_output_dir:
            self.training_output_dir = Path(training_output_dir)
        else:
            self.training_output_dir = self.base_dir / "training_output"
        
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            # Try to find the most recent checkpoint directory
            checkpoint_pattern = self.training_output_dir / "checkpoint_*"
            checkpoint_dirs = list(glob.glob(str(checkpoint_pattern)))
            if checkpoint_dirs:
                self.checkpoint_dir = Path(sorted(checkpoint_dirs)[-1])  # Use most recent
            else:
                self.checkpoint_dir = self.training_output_dir / "checkpoint_20250611_205320"
        
        if test_cases_dir:
            self.test_cases_dir = Path(test_cases_dir)
        else:
            self.test_cases_dir = self.base_dir / "test_cases"
        
        # Set up individual file paths
        if final_model_path:
            self.final_model_path = Path(final_model_path)
        else:
            self.final_model_path = self.training_output_dir / "final_model.json"
        
        if training_history_path:
            self.training_history_path = Path(training_history_path)
        else:
            self.training_history_path = self.checkpoint_dir / "training_history.json"
        
        if comparison_results_path:
            self.comparison_results_path = Path(comparison_results_path)
        else:
            self.comparison_results_path = self.base_dir / "comparison_results.json"
        
        # Load all data
        self.final_model = self._load_json(self.final_model_path)
        self.training_history = self._load_json(self.training_history_path)
        self.comparison_results = self._load_json(self.comparison_results_path)
        self.checkpoints = self._load_checkpoints()
        
        # Initialize report content
        self.report_content = []
        self.show_plots = show_plots
        self.quiet = quiet
        self.separate_plots = separate_plots
        
        # Print loading summary (only if not quiet)
        if not self.quiet:
            print(f"MOA Training Analyzer Configuration:")
            print(f"  Base directory: {self.base_dir}")
            print(f"  Output directory: {self.output_dir}")
            print(f"  Output prefix: '{self.output_prefix}'")
            print(f"  Final model: {self.final_model_path} ({'✓' if self.final_model_path.exists() else '✗'})")
            print(f"  Training history: {self.training_history_path} ({'✓' if self.training_history_path.exists() else '✗'})")
            print(f"  Comparison results: {self.comparison_results_path} ({'✓' if self.comparison_results_path.exists() else '✗'})")
            print(f"  Checkpoint directory: {self.checkpoint_dir} ({'✓' if self.checkpoint_dir.exists() else '✗'})")
            print(f"  Test cases directory: {self.test_cases_dir} ({'✓' if self.test_cases_dir.exists() else '✗'})")
            print(f"\nData Loading Summary:")
            print(f"  Found {len(self.checkpoints)} checkpoint files")
            print(f"  Training history contains {len(self.training_history.get('history', []))} entries")
            print(f"  Comparison results contain {len(self.comparison_results.get('comparison_results', []))} problem comparisons")

    def _load_json(self, filepath: Path) -> Dict:
        """Load JSON file with error handling."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if not self.quiet:
                print(f"Warning: Could not load {filepath}: {e}")
            return {}

    def _load_checkpoints(self) -> Dict[int, Dict]:
        """Load all checkpoint files."""
        checkpoints = {}
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "model_checkpoint_*.json"))
        
        for filepath in checkpoint_files:
            try:
                checkpoint_num = int(Path(filepath).stem.split('_')[-1])
                checkpoints[checkpoint_num] = self._load_json(Path(filepath))
            except (ValueError, Exception) as e:
                if not self.quiet:
                    print(f"Warning: Error processing checkpoint {filepath}: {e}")
        
        return checkpoints

    def _add_to_report(self, content: str, level: int = 0):
        """Add content to the markdown report."""
        if level > 0:
            content = "#" * level + " " + content
        self.report_content.append(content)

    def _extract_test_case_data(self):
        """Extract detailed test case performance data."""
        if not self.comparison_results or 'comparison_results' not in self.comparison_results:
            return None
        
        results = self.comparison_results['comparison_results']
        test_case_data = []
        
        for problem_idx, result in enumerate(results):
            problem_text = result.get('problem', f'Problem {problem_idx}')
            difficulty = result.get('difficulty', 'Unknown')
            
            # Extract MOA test results
            moa_data = result.get('moa', {})
            moa_test_results = moa_data.get('test_results', {})
            moa_total_tests = moa_data.get('total', 10)
            
            # Extract Single LLM test results
            single_data = result.get('single_llm', {})
            single_test_results = single_data.get('test_results', {})
            single_total_tests = single_data.get('total', 10)
            
            # Process each test case (limit to actual test cases, typically 10)
            max_tests = min(max(moa_total_tests, single_total_tests), 10)  # Limit to 10 test cases
            for test_case_id in range(max_tests):
                test_case_str = str(test_case_id)
                
                # MOA results for this test case
                moa_test_result = moa_test_results.get(test_case_str, {})
                test_case_data.append({
                    'problem_idx': problem_idx,
                    'problem_text': problem_text[:100] + '...' if len(problem_text) > 100 else problem_text,
                    'difficulty': difficulty,
                    'test_case_id': test_case_id,
                    'model': 'MOA',
                    'text_similarity': moa_test_result.get('text_similarity', 0),
                    'exact_match': moa_test_result.get('exact_match', 0),
                    'time_seconds': moa_data.get('time_seconds', 0),
                    'overall_text_sim': moa_data.get('scores', {}).get('text_similarity', 0),
                    'overall_exact_match': moa_data.get('scores', {}).get('exact_match', 0)
                })
                
                # Single LLM results for this test case
                single_test_result = single_test_results.get(test_case_str, {})
                test_case_data.append({
                    'problem_idx': problem_idx,
                    'problem_text': problem_text[:100] + '...' if len(problem_text) > 100 else problem_text,
                    'difficulty': difficulty,
                    'test_case_id': test_case_id,
                    'model': 'Single LLM',
                    'text_similarity': single_test_result.get('text_similarity', 0),
                    'exact_match': single_test_result.get('exact_match', 0),
                    'time_seconds': single_data.get('time_seconds', 0),
                    'overall_text_sim': single_data.get('scores', {}).get('text_similarity', 0),
                    'overall_exact_match': single_data.get('scores', {}).get('exact_match', 0)
                })
        
        return pd.DataFrame(test_case_data)

    def analyze_test_case_performance(self):
        """Analyze detailed test case performance patterns."""
        self._add_to_report("Test Case Performance Analysis", 2)
        
        test_df = self._extract_test_case_data()
        if test_df is None or len(test_df) == 0:
            self._add_to_report("❌ No test case data found")
            return
        
        # Create comprehensive test case analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Detailed Test Case Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Test case success rate by position
        ax1 = axes[0, 0]
        test_case_success = test_df.groupby(['test_case_id', 'model'])['exact_match'].mean().unstack()
        test_case_success.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Success Rate by Test Case Position', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Case ID')
        ax1.set_ylabel('Success Rate')
        ax1.legend(title='Model')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Problem difficulty vs success rate
        ax2 = axes[0, 1]
        if 'difficulty' in test_df.columns and len(test_df['difficulty'].unique()) > 1:
            difficulty_success = test_df.groupby(['difficulty', 'model'])['exact_match'].mean().unstack()
            if len(difficulty_success) > 0:
                difficulty_success.plot(kind='bar', ax=ax2, color=['skyblue', 'lightcoral'])
                ax2.set_title('Success Rate by Problem Difficulty', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Difficulty')
                ax2.set_ylabel('Success Rate')
                ax2.legend(title='Model')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'No difficulty data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Success Rate by Problem Difficulty', fontsize=14, fontweight='bold')
        else:
            # Show overall success rate instead
            overall_success = test_df.groupby('model')['exact_match'].mean()
            bars = ax2.bar(range(len(overall_success)), overall_success.values, color=['skyblue', 'lightcoral'])
            ax2.set_title('Overall Success Rate by Model', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Success Rate')
            ax2.set_xticks(range(len(overall_success)))
            ax2.set_xticklabels(overall_success.index)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, overall_success.values):
                ax2.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        # Plot 3: Test case consistency analysis
        ax3 = axes[1, 0]
        # Calculate per-problem test case pass rates
        problem_consistency = test_df.groupby(['problem_idx', 'model'])['exact_match'].agg(['mean', 'std']).reset_index()
        
        for model in ['MOA', 'Single LLM']:
            model_data = problem_consistency[problem_consistency['model'] == model]
            ax3.scatter(model_data['mean'], model_data['std'], 
                       label=model, alpha=0.6, s=50)
        
        ax3.set_title('Problem Consistency: Mean vs Std of Test Case Success', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Mean Success Rate')
        ax3.set_ylabel('Standard Deviation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Failure pattern heatmap
        ax4 = axes[1, 1]
        # Create failure matrix (problems x test cases)
        moa_failures = test_df[test_df['model'] == 'MOA'].pivot_table(
            values='exact_match', index='problem_idx', columns='test_case_id', 
            aggfunc='mean', fill_value=0
        )
        
        if len(moa_failures) > 0:
            # Show only first 20 problems for readability
            display_failures = moa_failures.head(20) if len(moa_failures) > 20 else moa_failures
            sns.heatmap(display_failures, ax=ax4, cmap='RdYlGn', cbar_kws={'label': 'Success Rate'})
            ax4.set_title('MOA Test Case Success Heatmap (First 20 Problems)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Test Case ID')
            ax4.set_ylabel('Problem Index')
        
        # Plot 5: Performance distribution by test case
        ax5 = axes[2, 0]
        test_case_scores = []
        test_case_labels = []
        
        for test_id in sorted(test_df['test_case_id'].unique()):
            moa_scores = test_df[(test_df['test_case_id'] == test_id) & 
                                (test_df['model'] == 'MOA')]['text_similarity']
            single_scores = test_df[(test_df['test_case_id'] == test_id) & 
                                   (test_df['model'] == 'Single LLM')]['text_similarity']
            
            if len(moa_scores) > 0 and len(single_scores) > 0:
                test_case_scores.extend([moa_scores.values, single_scores.values])
                test_case_labels.extend([f'MOA-T{test_id}', f'Single-T{test_id}'])
        
        if test_case_scores:
            bp = ax5.boxplot(test_case_scores, labels=test_case_labels, patch_artist=True)
            # Color MOA boxes blue, Single LLM boxes red
            for i, box in enumerate(bp['boxes']):
                if 'MOA' in test_case_labels[i]:
                    box.set_facecolor('skyblue')
                else:
                    box.set_facecolor('lightcoral')
            
            ax5.set_title('Score Distribution by Test Case', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Text Similarity Score')
            ax5.tick_params(axis='x', rotation=90)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Hardest test cases identification
        ax6 = axes[2, 1]
        test_case_difficulty = test_df.groupby('test_case_id')['exact_match'].mean().sort_values()
        
        bars = ax6.bar(range(len(test_case_difficulty)), test_case_difficulty.values, 
                      color='lightsteelblue', edgecolor='navy')
        ax6.set_title('Test Case Difficulty Ranking', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Test Case ID (sorted by difficulty)')
        ax6.set_ylabel('Overall Success Rate')
        ax6.set_xticks(range(len(test_case_difficulty)))
        ax6.set_xticklabels([f'T{idx}' for idx in test_case_difficulty.index], rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Highlight hardest test cases
        hardest_indices = test_case_difficulty.head(3).index
        for i, test_id in enumerate(test_case_difficulty.index):
            if test_id in hardest_indices:
                bars[i].set_color('lightcoral')
        
        plt.tight_layout()
        
        # Save the plot
        if self.separate_plots:
            saved_files = self._create_test_case_individual_plots(test_df)
        else:
            output_path = self.output_dir / f"{self.output_prefix}test_case_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files = [output_path]
            if not self.quiet:
                print(f"Saved test case analysis to: {output_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Add detailed statistics to report
        self._add_to_report("### Test Case Performance Summary")
        
        # Overall test case statistics
        total_test_cases = len(test_df['test_case_id'].unique())
        total_problems = len(test_df['problem_idx'].unique())
        self._add_to_report(f"- **Total Test Cases Analyzed**: {total_test_cases}")
        self._add_to_report(f"- **Total Problems**: {total_problems}")
        self._add_to_report(f"- **Total Test Executions**: {len(test_df)}")
        
        # Test case difficulty analysis
        hardest_test_cases = test_case_difficulty.head(3)
        easiest_test_cases = test_case_difficulty.tail(3)
        
        self._add_to_report("#### Hardest Test Cases")
        for test_id, success_rate in hardest_test_cases.items():
            self._add_to_report(f"- **Test Case {test_id}**: {success_rate:.1%} success rate")
        
        self._add_to_report("#### Easiest Test Cases")
        for test_id, success_rate in easiest_test_cases.items():
            self._add_to_report(f"- **Test Case {test_id}**: {success_rate:.1%} success rate")
        
        # Model comparison by test case
        self._add_to_report("#### Model Performance by Test Case")
        for model in ['MOA', 'Single LLM']:
            model_data = test_df[test_df['model'] == model]
            avg_success = model_data['exact_match'].mean()
            consistency = 1 - model_data.groupby('problem_idx')['exact_match'].std().mean()
            self._add_to_report(f"- **{model}**: {avg_success:.1%} average success, {consistency:.1%} consistency")
        
        self._add_to_report("")
        
        return saved_files if 'saved_files' in locals() else []

    def _create_test_case_individual_plots(self, test_df):
        """Create individual plots for test case analysis."""
        saved_files = []
        
        # Plot 1: Test case success rate by position
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        test_case_success = test_df.groupby(['test_case_id', 'model'])['exact_match'].mean().unstack()
        test_case_success.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Success Rate by Test Case Position', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Case ID')
        ax1.set_ylabel('Success Rate')
        ax1.legend(title='Model')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.output_prefix}testcase_01_success_by_position.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved test case success by position to: {output_path}")
        plt.close()
        
        # Plot 2: Problem difficulty analysis or overall success rate
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        if 'difficulty' in test_df.columns and len(test_df['difficulty'].unique()) > 1:
            difficulty_success = test_df.groupby(['difficulty', 'model'])['exact_match'].mean().unstack()
            if len(difficulty_success) > 0:
                difficulty_success.plot(kind='bar', ax=ax2, color=['skyblue', 'lightcoral'])
                ax2.set_title('Success Rate by Problem Difficulty', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Difficulty')
                ax2.set_ylabel('Success Rate')
                ax2.legend(title='Model')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'No difficulty data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Success Rate by Problem Difficulty', fontsize=14, fontweight='bold')
        else:
            # Show overall success rate instead
            overall_success = test_df.groupby('model')['exact_match'].mean()
            bars = ax2.bar(range(len(overall_success)), overall_success.values, color=['skyblue', 'lightcoral'])
            ax2.set_title('Overall Success Rate by Model', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Success Rate')
            ax2.set_xticks(range(len(overall_success)))
            ax2.set_xticklabels(overall_success.index)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, overall_success.values):
                ax2.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.output_prefix}testcase_02_overall_success_rate.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved overall success rate analysis to: {output_path}")
        plt.close()
        
        # Plot 3: Failure pattern heatmap
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
        moa_failures = test_df[test_df['model'] == 'MOA'].pivot_table(
            values='exact_match', index='problem_idx', columns='test_case_id', 
            aggfunc='mean', fill_value=0
        )
        
        if len(moa_failures) > 0:
            display_failures = moa_failures.head(30) if len(moa_failures) > 30 else moa_failures
            sns.heatmap(display_failures, ax=ax3, cmap='RdYlGn', cbar_kws={'label': 'Success Rate'})
            ax3.set_title('MOA Test Case Success Heatmap', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Test Case ID')
            ax3.set_ylabel('Problem Index')
            plt.tight_layout()
            
            output_path = self.output_dir / f"{self.output_prefix}testcase_03_failure_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files.append(output_path)
            if not self.quiet:
                print(f"Saved failure heatmap to: {output_path}")
        plt.close()
        
        # Plot 4: Test case difficulty ranking
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        test_case_difficulty = test_df.groupby('test_case_id')['exact_match'].mean().sort_values()
        
        bars = ax4.bar(range(len(test_case_difficulty)), test_case_difficulty.values, 
                      color='lightsteelblue', edgecolor='navy')
        ax4.set_title('Test Case Difficulty Ranking', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Test Case ID (sorted by difficulty)')
        ax4.set_ylabel('Overall Success Rate')
        ax4.set_xticks(range(len(test_case_difficulty)))
        ax4.set_xticklabels([f'T{idx}' for idx in test_case_difficulty.index], rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Highlight hardest test cases
        hardest_indices = test_case_difficulty.head(3).index
        for i, test_id in enumerate(test_case_difficulty.index):
            if test_id in hardest_indices:
                bars[i].set_color('lightcoral')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.output_prefix}testcase_04_difficulty_ranking.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved test case difficulty ranking to: {output_path}")
        plt.close()
        
        return saved_files

    def analyze_problem_level_performance(self):
        """Analyze performance at the individual problem level."""
        self._add_to_report("Problem-Level Performance Analysis", 2)
        
        if not self.comparison_results or 'comparison_results' not in self.comparison_results:
            self._add_to_report("❌ No comparison results found")
            return
        
        results = self.comparison_results['comparison_results']
        problem_data = []
        
        for problem_idx, result in enumerate(results):
            problem_text = result.get('problem', f'Problem {problem_idx}')
            difficulty = result.get('difficulty', 'Unknown')
            
            moa_data = result.get('moa', {})
            single_data = result.get('single_llm', {})
            
            # Calculate test case statistics for each problem
            moa_test_results = moa_data.get('test_results', {})
            single_test_results = single_data.get('test_results', {})
            
            moa_test_scores = [res.get('exact_match', 0) for res in moa_test_results.values()]
            single_test_scores = [res.get('exact_match', 0) for res in single_test_results.values()]
            
            problem_data.append({
                'problem_idx': problem_idx,
                'problem_text': problem_text[:50] + '...' if len(problem_text) > 50 else problem_text,
                'difficulty': difficulty,
                'moa_overall_score': moa_data.get('scores', {}).get('exact_match', 0),
                'single_overall_score': single_data.get('scores', {}).get('exact_match', 0),
                'moa_test_case_mean': np.mean(moa_test_scores) if moa_test_scores else 0,
                'single_test_case_mean': np.mean(single_test_scores) if single_test_scores else 0,
                'moa_test_case_std': np.std(moa_test_scores) if moa_test_scores else 0,
                'single_test_case_std': np.std(single_test_scores) if single_test_scores else 0,
                'moa_time': moa_data.get('time_seconds', 0),
                'single_time': single_data.get('time_seconds', 0),
                'moa_passed_tests': sum(moa_test_scores),
                'single_passed_tests': sum(single_test_scores),
                'total_tests': len(moa_test_scores)
            })
        
        problem_df = pd.DataFrame(problem_data)
        
        # Create problem-level analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Problem-Level Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Problem performance scatter
        ax1 = axes[0, 0]
        ax1.scatter(problem_df['moa_overall_score'], problem_df['single_overall_score'], 
                   alpha=0.6, s=50, c='steelblue')
        
        # Add diagonal line for equal performance
        max_score = max(problem_df['moa_overall_score'].max(), problem_df['single_overall_score'].max())
        ax1.plot([0, max_score], [0, max_score], 'r--', alpha=0.7, label='Equal Performance')
        
        ax1.set_title('Problem-by-Problem Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('MOA Score')
        ax1.set_ylabel('Single LLM Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance difference distribution
        ax2 = axes[0, 1]
        performance_diff = problem_df['moa_overall_score'] - problem_df['single_overall_score']
        ax2.hist(performance_diff, bins=20, alpha=0.7, color='lightblue', edgecolor='navy')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Difference')
        ax2.axvline(x=performance_diff.mean(), color='green', linestyle='-', alpha=0.7, 
                   label=f'Mean Diff: {performance_diff.mean():.3f}')
        
        ax2.set_title('Performance Difference Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('MOA Score - Single LLM Score')
        ax2.set_ylabel('Number of Problems')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Difficulty vs performance or overall comparison
        ax3 = axes[0, 2]
        if 'difficulty' in problem_df.columns and len(problem_df['difficulty'].unique()) > 1:
            difficulty_perf = problem_df.groupby('difficulty')[['moa_overall_score', 'single_overall_score']].mean()
            difficulty_perf.plot(kind='bar', ax=ax3, color=['skyblue', 'lightcoral'])
            ax3.set_title('Average Performance by Difficulty', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Average Score')
            ax3.legend(['MOA', 'Single LLM'])
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            # Show overall performance comparison instead
            overall_perf = [problem_df['moa_overall_score'].mean(), problem_df['single_overall_score'].mean()]
            bars = ax3.bar(['MOA', 'Single LLM'], overall_perf, color=['skyblue', 'lightcoral'])
            ax3.set_title('Overall Average Performance', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Average Score')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, overall_perf):
                ax3.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Time vs performance trade-off
        ax4 = axes[1, 0]
        ax4.scatter(problem_df['moa_time'], problem_df['moa_overall_score'], 
                   alpha=0.6, label='MOA', color='skyblue', s=50)
        ax4.scatter(problem_df['single_time'], problem_df['single_overall_score'], 
                   alpha=0.6, label='Single LLM', color='lightcoral', s=50)
        
        ax4.set_title('Time vs Performance Trade-off', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Test case consistency
        ax5 = axes[1, 1]
        ax5.scatter(problem_df['moa_test_case_mean'], problem_df['moa_test_case_std'], 
                   alpha=0.6, label='MOA', color='skyblue', s=50)
        ax5.scatter(problem_df['single_test_case_mean'], problem_df['single_test_case_std'], 
                   alpha=0.6, label='Single LLM', color='lightcoral', s=50)
        
        ax5.set_title('Test Case Consistency Analysis', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Mean Test Case Score')
        ax5.set_ylabel('Standard Deviation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Problem success rate distribution
        ax6 = axes[1, 2]
        moa_success_rates = problem_df['moa_passed_tests'] / problem_df['total_tests']
        single_success_rates = problem_df['single_passed_tests'] / problem_df['total_tests']
        
        ax6.hist(moa_success_rates, bins=10, alpha=0.6, label='MOA', color='skyblue', density=True)
        ax6.hist(single_success_rates, bins=10, alpha=0.6, label='Single LLM', color='lightcoral', density=True)
        
        ax6.set_title('Problem Success Rate Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Success Rate (Passed Tests / Total Tests)')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        if self.separate_plots:
            saved_files = self._create_problem_level_individual_plots(problem_df)
        else:
            output_path = self.output_dir / f"{self.output_prefix}problem_level_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files = [output_path]
            if not self.quiet:
                print(f"Saved problem-level analysis to: {output_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Add detailed statistics to report
        self._add_to_report("### Problem-Level Performance Summary")
        
        # Overall statistics
        moa_wins = (problem_df['moa_overall_score'] > problem_df['single_overall_score']).sum()
        single_wins = (problem_df['single_overall_score'] > problem_df['moa_overall_score']).sum()
        ties = (problem_df['moa_overall_score'] == problem_df['single_overall_score']).sum()
        
        self._add_to_report(f"- **MOA Wins**: {moa_wins} problems ({moa_wins/len(problem_df):.1%})")
        self._add_to_report(f"- **Single LLM Wins**: {single_wins} problems ({single_wins/len(problem_df):.1%})")
        self._add_to_report(f"- **Ties**: {ties} problems ({ties/len(problem_df):.1%})")
        
        # Performance statistics
        avg_moa_score = problem_df['moa_overall_score'].mean()
        avg_single_score = problem_df['single_overall_score'].mean()
        self._add_to_report(f"- **Average MOA Score**: {avg_moa_score:.3f}")
        self._add_to_report(f"- **Average Single LLM Score**: {avg_single_score:.3f}")
        self._add_to_report(f"- **Average Improvement**: {avg_moa_score - avg_single_score:.3f}")
        
        # Identify best and worst performing problems
        performance_diff = problem_df['moa_overall_score'] - problem_df['single_overall_score']
        best_improvements = problem_df.loc[performance_diff.nlargest(3).index]
        worst_regressions = problem_df.loc[performance_diff.nsmallest(3).index]
        
        self._add_to_report("#### Biggest MOA Improvements")
        for _, row in best_improvements.iterrows():
            diff = row['moa_overall_score'] - row['single_overall_score']
            self._add_to_report(f"- **Problem {row['problem_idx']}**: +{diff:.3f} improvement")
        
        self._add_to_report("#### Biggest MOA Regressions")
        for _, row in worst_regressions.iterrows():
            diff = row['moa_overall_score'] - row['single_overall_score']
            self._add_to_report(f"- **Problem {row['problem_idx']}**: {diff:.3f} regression")
        
        self._add_to_report("")
        
        return saved_files if 'saved_files' in locals() else []

    def _create_problem_level_individual_plots(self, problem_df):
        """Create individual plots for problem-level analysis."""
        saved_files = []
        
        # Plot 1: Problem performance scatter
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax1.scatter(problem_df['moa_overall_score'], problem_df['single_overall_score'], 
                   alpha=0.6, s=50, c='steelblue')
        
        max_score = max(problem_df['moa_overall_score'].max(), problem_df['single_overall_score'].max())
        ax1.plot([0, max_score], [0, max_score], 'r--', alpha=0.7, label='Equal Performance')
        
        ax1.set_title('Problem-by-Problem Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('MOA Score')
        ax1.set_ylabel('Single LLM Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.output_prefix}problem_01_performance_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved problem performance scatter to: {output_path}")
        plt.close()
        
        # Plot 2: Performance difference distribution
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        performance_diff = problem_df['moa_overall_score'] - problem_df['single_overall_score']
        ax2.hist(performance_diff, bins=20, alpha=0.7, color='lightblue', edgecolor='navy')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Difference')
        ax2.axvline(x=performance_diff.mean(), color='green', linestyle='-', alpha=0.7, 
                   label=f'Mean Diff: {performance_diff.mean():.3f}')
        
        ax2.set_title('Performance Difference Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('MOA Score - Single LLM Score')
        ax2.set_ylabel('Number of Problems')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.output_prefix}problem_02_difference_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved performance difference distribution to: {output_path}")
        plt.close()
        
        # Plot 3: Time vs performance trade-off
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
        ax3.scatter(problem_df['moa_time'], problem_df['moa_overall_score'], 
                   alpha=0.6, label='MOA', color='skyblue', s=50)
        ax3.scatter(problem_df['single_time'], problem_df['single_overall_score'], 
                   alpha=0.6, label='Single LLM', color='lightcoral', s=50)
        
        ax3.set_title('Time vs Performance Trade-off by Problem', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.output_prefix}problem_03_time_vs_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved time vs performance to: {output_path}")
        plt.close()
        
        return saved_files

    def _create_weight_progression_individual_plots(self, weight_data, checkpoint_nums, 
                                                   weight_stats, checkpoints_for_stats,
                                                   key_checkpoints, weight_changes, avg_changes):
        """Create individual plots for weight progression analysis."""
        saved_files = []
        
        # Plot 1: Individual Weight Trajectories
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        for key, values in weight_data.items():
            if values:
                checkpoints_nums, weights = zip(*values)
                ax1.plot(checkpoints_nums, weights, label=key, alpha=0.7, linewidth=1)
        ax1.set_title('Individual Weight Trajectories', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Checkpoint Number')
        ax1.set_ylabel('Weight Value')
        ax1.grid(True, alpha=0.3)
        if len(weight_data) <= 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        output_path = self.output_dir / f"{self.output_prefix}weight_01_individual_trajectories.png"
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig1)
        
        # Plot 2: Weight Statistics Over Time
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        if checkpoints_for_stats:
            ax2.plot(checkpoints_for_stats, weight_stats['mean'], label='Mean', linewidth=2)
            ax2.fill_between(checkpoints_for_stats, 
                           np.array(weight_stats['mean']) - np.array(weight_stats['std']),
                           np.array(weight_stats['mean']) + np.array(weight_stats['std']),
                           alpha=0.3, label='±1 STD')
            ax2.plot(checkpoints_for_stats, weight_stats['min'], label='Min', linestyle='--')
            ax2.plot(checkpoints_for_stats, weight_stats['max'], label='Max', linestyle='--')
        
        ax2.set_title('Weight Statistics Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Checkpoint Number')
        ax2.set_ylabel('Weight Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}weight_02_statistics_over_time.png"
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig2)
        
        # Plot 3: Weight Distribution at Key Checkpoints
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        for i, checkpoint_num in enumerate(key_checkpoints):
            checkpoint_weights = []
            for key, values in weight_data.items():
                for cp_num, weight in values:
                    if cp_num == checkpoint_num:
                        checkpoint_weights.append(weight)
            
            if checkpoint_weights:
                ax3.hist(checkpoint_weights, bins=20, alpha=0.6, 
                        label=f'Checkpoint {checkpoint_num}', density=True)
        
        ax3.set_title('Weight Distribution at Key Checkpoints', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}weight_03_distribution_checkpoints.png"
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig3)
        
        # Plot 4: Weight Change Velocity
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        if avg_changes:
            checkpoints_list, changes_list = zip(*sorted(avg_changes.items()))
            ax4.plot(checkpoints_list, changes_list, marker='o', linewidth=2)
            ax4.set_title('Average Weight Change Magnitude', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Checkpoint Number')
            ax4.set_ylabel('Average |Weight Change|')
            ax4.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}weight_04_change_magnitude.png"
        fig4.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig4)
        
        return saved_files

    def _create_score_progression_individual_plots(self, df, history, overall_stats):
        """Create individual plots for score progression analysis."""
        saved_files = []
        
        # Plot 1: Individual agent learning progress with moving averages
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        for agent in df['agent'].unique()[:10]:  # Limit to first 10 agents for readability
            agent_data = df[df['agent'] == agent].sort_values('iteration')
            if len(agent_data) > 0:
                # Calculate moving average (last 10 iterations) for better trend visualization
                window_size = min(10, len(agent_data) // 3) if len(agent_data) > 3 else 1
                if window_size > 1:
                    moving_avg = agent_data['score'].rolling(window=window_size, center=True).mean()
                    ax1.plot(agent_data['iteration'], moving_avg, 
                            label=f'Agent {agent}', alpha=0.8, linewidth=2)
                    # Show raw data as faint background
                    ax1.plot(agent_data['iteration'], agent_data['score'], 
                            alpha=0.3, linewidth=0.5, color=ax1.lines[-1].get_color())
                else:
                    ax1.plot(agent_data['iteration'], agent_data['score'], 
                            label=f'Agent {agent}', alpha=0.8, linewidth=1.5)
        ax1.set_title('Individual Agent Learning Progress (Moving Average)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Score (Moving Average)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}score_01_individual_agent_trajectories.png"
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig1)
        
        # Plot 2: Cumulative average scores by layer
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        layer_stats = df.groupby(['iteration', 'layer'])['score'].agg(['mean', 'std']).reset_index()
        
        for layer in layer_stats['layer'].unique():
            layer_data = layer_stats[layer_stats['layer'] == layer].sort_values('iteration')
            if len(layer_data) > 0:
                # Calculate cumulative average for this layer
                cumulative_mean = layer_data['mean'].expanding().mean()
                cumulative_std = layer_data['std'].expanding().mean()
                ax2.plot(layer_data['iteration'], cumulative_mean, 
                        label=f'Layer {layer}', linewidth=2, marker='o', markersize=3)
                ax2.fill_between(layer_data['iteration'],
                               cumulative_mean - cumulative_std,
                               cumulative_mean + cumulative_std,
                               alpha=0.2)
        
        ax2.set_title('Cumulative Average Scores by Layer', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Cumulative Average Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}score_02_average_scores_by_layer.png"
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig2)
        
        # Plot 3: Score distribution evolution
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        iterations_to_show = [0, len(history)//3, 2*len(history)//3, len(history)-1]
        
        for i, iteration in enumerate(iterations_to_show):
            if iteration < len(history):
                iteration_scores = df[df['iteration'] == iteration]['score'].values
                if len(iteration_scores) > 0:
                    ax3.hist(iteration_scores, bins=15, alpha=0.6, 
                            label=f'Iteration {iteration}', density=True)
        
        ax3.set_title('Score Distribution Evolution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}score_03_distribution_evolution.png"
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig3)
        
        # Plot 4: Convergence analysis with cumulative statistics
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate cumulative statistics for better convergence visualization
        cumulative_mean = overall_stats['mean'].expanding().mean()
        cumulative_std = overall_stats['std'].expanding().mean()
        cumulative_max = overall_stats['max'].expanding().max()
        cumulative_min = overall_stats['min'].expanding().min()
        
        ax4.plot(overall_stats['iteration'], cumulative_mean, 
                label='Cumulative Mean Score', linewidth=2, color='blue')
        ax4.fill_between(overall_stats['iteration'],
                        cumulative_mean - cumulative_std,
                        cumulative_mean + cumulative_std,
                        alpha=0.3, color='blue', label='±1 Cumulative STD')
        ax4.plot(overall_stats['iteration'], cumulative_max, 
                label='Best Score Achieved', linestyle='--', color='green')
        ax4.plot(overall_stats['iteration'], cumulative_min, 
                label='Worst Score Seen', linestyle='--', color='red')
        
        ax4.set_title('Overall Convergence Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Training Iteration')
        ax4.set_ylabel('Cumulative Score Statistics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}score_04_convergence_analysis.png"
        fig4.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig4)
        
        # Plot 5: Learning velocity (based on cumulative average improvement)
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
        if len(overall_stats) > 1:
            # Calculate learning velocity based on cumulative average changes
            cumulative_mean = overall_stats['mean'].expanding().mean()
            score_changes = np.diff(cumulative_mean)
            # Smooth the changes with a rolling average for better visualization
            if len(score_changes) > 5:
                smoothed_changes = pd.Series(score_changes).rolling(window=min(5, len(score_changes)//3), center=True).mean()
                ax5.plot(overall_stats['iteration'][1:], smoothed_changes, 
                        marker='o', linewidth=2, markersize=3, label='Smoothed Learning Velocity')
            ax5.plot(overall_stats['iteration'][1:], score_changes, 
                    alpha=0.5, linewidth=1, markersize=2, label='Raw Changes')
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
            ax5.set_title('Learning Velocity (Cumulative Average Change)', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Training Iteration')
            ax5.set_ylabel('Change in Cumulative Average Score')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}score_05_improvement_rate.png"
        fig5.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig5)
        
        # Plot 6: Final score comparison
        fig6, ax6 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get final scores - use the last few iterations to get a more stable average
        max_iteration = df['iteration'].max()
        final_iterations = max(1, min(3, len(df['iteration'].unique())))
        final_scores = df[df['iteration'] >= max_iteration - final_iterations + 1]
        
        # Group by layer for better visualization
        layer_final_scores = final_scores.groupby('layer')['score'].mean()
        
        if len(layer_final_scores) > 0 and layer_final_scores.sum() > 0:
            bars = ax6.bar(range(len(layer_final_scores)), layer_final_scores.values, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(layer_final_scores))))
            ax6.set_title('Final Average Scores by Layer', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Layer')
            ax6.set_ylabel('Final Average Score')
            ax6.set_xticks(range(len(layer_final_scores)))
            ax6.set_xticklabels([f'Layer {i}' for i in layer_final_scores.index])
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        else:
            # If no valid scores, show overall score distribution instead
            overall_final_scores = final_scores['score']
            if len(overall_final_scores) > 0:
                ax6.hist(overall_final_scores, bins=min(10, len(overall_final_scores)), 
                        alpha=0.7, color='skyblue')
                ax6.set_title('Final Score Distribution', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Score')
                ax6.set_ylabel('Frequency')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No final scores available', ha='center', va='center', 
                        transform=ax6.transAxes, fontsize=12)
                ax6.set_title('Final Scores - No Data Available')
        
        output_path = self.output_dir / f"{self.output_prefix}score_06_final_comparison.png"
        fig6.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig6)
        
        return saved_files

    def _create_performance_comparison_individual_plots(self, df, moa_scores, single_scores, 
                                                       moa_times, single_times, 
                                                       moa_success_text, moa_success_exact, 
                                                       single_success_text, single_success_exact):
        """Create individual plots for performance comparison analysis."""
        saved_files = []
        
        # Plot 1: Accuracy comparison
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        metrics = ['text_similarity', 'exact_match']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, moa_scores, width, label='MOA', color='skyblue')
        bars2 = ax1.bar(x + width/2, single_scores, width, label='Single LLM', color='lightcoral')
        
        ax1.set_title('Average Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Text Similarity', 'Exact Match'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        output_path = self.output_dir / f"{self.output_prefix}performance_01_accuracy_comparison.png"
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig1)
        
        # Plot 2: Time comparison
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        box_data = [moa_times.dropna(), single_times.dropna()]
        box_labels = ['MOA', 'Single LLM']
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        if len(bp['boxes']) >= 1:
            bp['boxes'][0].set_facecolor('skyblue')
        if len(bp['boxes']) >= 2:
            bp['boxes'][1].set_facecolor('lightcoral')
        
        ax2.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}performance_02_time_comparison.png"
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig2)
        
        # Plot 3: Success rate comparison
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        
        success_data = {
            'MOA': [moa_success_text, moa_success_exact],
            'Single LLM': [single_success_text, single_success_exact]
        }
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, success_data['MOA'], width, label='MOA', color='skyblue')
        bars2 = ax3.bar(x + width/2, success_data['Single LLM'], width, label='Single LLM', color='lightcoral')
        
        ax3.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Success Rate')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Text Similarity', 'Exact Match'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2%}', ha='center', va='bottom')
        
        output_path = self.output_dir / f"{self.output_prefix}performance_03_success_rate_comparison.png"
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig3)
        
        # Plot 4: Score distribution comparison
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        
        moa_text_scores = df[df['model'] == 'MOA']['text_similarity'].dropna()
        single_text_scores = df[df['model'] == 'Single LLM']['text_similarity'].dropna()
        
        ax4.hist(moa_text_scores, bins=20, alpha=0.6, label='MOA', density=True, color='skyblue')
        ax4.hist(single_text_scores, bins=20, alpha=0.6, label='Single LLM', density=True, color='lightcoral')
        
        ax4.set_title('Text Similarity Score Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Text Similarity Score')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}performance_04_score_distribution.png"
        fig4.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig4)
        
        # Plot 5: Time vs Performance scatter
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
        
        moa_df = df[df['model'] == 'MOA'].dropna(subset=['time_seconds', 'text_similarity'])
        single_df = df[df['model'] == 'Single LLM'].dropna(subset=['time_seconds', 'text_similarity'])
        
        ax5.scatter(moa_df['time_seconds'], moa_df['text_similarity'], 
                   label='MOA', alpha=0.6, color='skyblue', s=50)
        ax5.scatter(single_df['time_seconds'], single_df['text_similarity'], 
                   label='Single LLM', alpha=0.6, color='lightcoral', s=50)
        
        ax5.set_title('Performance vs Time Trade-off', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Text Similarity Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}performance_05_time_vs_performance.png"
        fig5.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig5)
        
        # Plot 6: Efficiency comparison (score per second)
        fig6, ax6 = plt.subplots(1, 1, figsize=(10, 6))
        
        moa_efficiency = moa_df['text_similarity'] / (moa_df['time_seconds'] + 1e-6)
        single_efficiency = single_df['text_similarity'] / (single_df['time_seconds'] + 1e-6)
        
        efficiency_data = [moa_efficiency.dropna(), single_efficiency.dropna()]
        efficiency_labels = ['MOA', 'Single LLM']
        
        # Filter out extreme outliers for better visualization
        filtered_efficiency_data = []
        for data in efficiency_data:
            if len(data) > 0:
                q99 = data.quantile(0.99)
                filtered_data = data[data <= q99]
                filtered_efficiency_data.append(filtered_data)
            else:
                filtered_efficiency_data.append(data)
        
        if all(len(data) > 0 for data in filtered_efficiency_data):
            bp = ax6.boxplot(filtered_efficiency_data, labels=efficiency_labels, patch_artist=True)
            if len(bp['boxes']) >= 1:
                bp['boxes'][0].set_facecolor('skyblue')
            if len(bp['boxes']) >= 2:
                bp['boxes'][1].set_facecolor('lightcoral')
        
        ax6.set_title('Efficiency (Score/Second)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score per Second')
        ax6.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}performance_06_efficiency.png"
        fig6.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        
        return saved_files

    def _create_metrics_comparison_individual_plots(self, df, correlation, agreement_df, thresholds, threshold_performance):
        """Create individual plots for metrics comparison analysis."""
        saved_files = []
        
        # Plot 1: Correlation between metrics
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax1.scatter(model_data['text_similarity'], model_data['exact_match'], 
                       label=model, alpha=0.6, s=50)
        
        # Add correlation line
        if not np.isnan(correlation):
            z = np.polyfit(df['text_similarity'], df['exact_match'], 1)
            p = np.poly1d(z)
            ax1.plot(sorted(df['text_similarity']), p(sorted(df['text_similarity'])), 
                    "r--", alpha=0.8, label=f'Correlation: {correlation:.3f}')
        
        ax1.set_title('Text Similarity vs Exact Match Correlation', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Text Similarity Score')
        ax1.set_ylabel('Exact Match Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}metrics_01_correlation.png"
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig1)
        
        # Plot 2: Metric agreement analysis
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        agreement_counts = agreement_df.groupby(['model', 'category']).size().unstack(fill_value=0)
        
        if len(agreement_counts) > 0:
            agreement_counts.plot(kind='bar', ax=ax2, stacked=True)
            ax2.set_title('Metric Agreement Patterns', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Count')
            ax2.legend(title='Agreement Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
        
        output_path = self.output_dir / f"{self.output_prefix}metrics_02_agreement_patterns.png"
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig2)
        
        # Plot 3: Metric sensitivity analysis
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate differences between metrics for each model
        df_copy = df.copy()
        df_copy['metric_diff'] = df_copy['text_similarity'] - df_copy['exact_match']
        
        for model in df_copy['model'].unique():
            model_data = df_copy[df_copy['model'] == model]
            if len(model_data) > 0:
                ax3.hist(model_data['metric_diff'], bins=20, alpha=0.6, 
                        label=model, density=True)
        
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Agreement')
        ax3.set_title('Metric Difference Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Text Similarity - Exact Match')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}metrics_03_sensitivity.png"
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig3)
        
        # Plot 4: Performance by metric threshold
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax4.plot(thresholds, threshold_performance['MOA']['text'], 
                label='MOA Text Similarity', linewidth=2, marker='o')
        ax4.plot(thresholds, threshold_performance['MOA']['exact'], 
                label='MOA Exact Match', linewidth=2, marker='s')
        ax4.plot(thresholds, threshold_performance['Single LLM']['text'], 
                label='Single LLM Text Similarity', linewidth=2, marker='^', linestyle='--')
        ax4.plot(thresholds, threshold_performance['Single LLM']['exact'], 
                label='Single LLM Exact Match', linewidth=2, marker='d', linestyle='--')
        
        ax4.set_title('Performance by Score Threshold', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Score Threshold')
        ax4.set_ylabel('Fraction Above Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.output_prefix}metrics_04_threshold_performance.png"
        fig4.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        if not self.quiet:
            print(f"Saved individual plot to: {output_path}")
        if not self.show_plots:
            plt.close(fig4)
        
        return saved_files

    def _save_report(self):
        """Save the accumulated report content to a markdown file."""
        report_path = self.output_dir / f"{self.output_prefix}analysis_report.md"
        
        # Add header
        full_report = [
            "# MOA Training Analysis Report",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
            "",
            "## Configuration",
            f"- **Base directory**: `{self.base_dir}`",
            f"- **Output directory**: `{self.output_dir}`",
            f"- **Output prefix**: `{self.output_prefix}`",
            f"- **Final model**: `{self.final_model_path}` {'✅' if self.final_model_path.exists() else '❌'}",
            f"- **Training history**: `{self.training_history_path}` {'✅' if self.training_history_path.exists() else '❌'}",
            f"- **Comparison results**: `{self.comparison_results_path}` {'✅' if self.comparison_results_path.exists() else '❌'}",
            f"- **Checkpoint directory**: `{self.checkpoint_dir}` {'✅' if self.checkpoint_dir.exists() else '❌'}",
            "",
            "---",
            ""
        ]
        
        # Add accumulated content
        full_report.extend(self.report_content)
        
        # Add footer
        generated_files_section = [
            "",
            "---",
            "",
            "## Generated Files",
        ]
        
        if self.separate_plots:
            generated_files_section.extend([
                "### Weight Progression Analysis",
                f"- `{self.output_prefix}weight_01_individual_trajectories.png`",
                f"- `{self.output_prefix}weight_02_statistics_over_time.png`",
                f"- `{self.output_prefix}weight_03_distribution_checkpoints.png`",
                f"- `{self.output_prefix}weight_04_change_magnitude.png`",
                "",
                "### Score Progression Analysis",
                f"- `{self.output_prefix}score_01_individual_agent_trajectories.png`",
                f"- `{self.output_prefix}score_02_average_scores_by_layer.png`",
                f"- `{self.output_prefix}score_03_distribution_evolution.png`",
                f"- `{self.output_prefix}score_04_convergence_analysis.png`",
                f"- `{self.output_prefix}score_05_improvement_rate.png`",
                f"- `{self.output_prefix}score_06_final_comparison.png`",
                "",
                "### Performance Comparison Analysis",
                f"- `{self.output_prefix}performance_01_accuracy_comparison.png`",
                f"- `{self.output_prefix}performance_02_time_comparison.png`",
                f"- `{self.output_prefix}performance_03_success_rate_comparison.png`",
                f"- `{self.output_prefix}performance_04_score_distribution.png`",
                f"- `{self.output_prefix}performance_05_time_vs_performance.png`",
                f"- `{self.output_prefix}performance_06_efficiency.png`",
                "",
                "### Metrics Comparison Analysis",
                f"- `{self.output_prefix}metrics_01_correlation.png`",
                f"- `{self.output_prefix}metrics_02_agreement_patterns.png`",
                f"- `{self.output_prefix}metrics_03_sensitivity.png`",
                f"- `{self.output_prefix}metrics_04_threshold_performance.png`",
                "",
                "### Report",
                f"- `{self.output_prefix}analysis_report.md`",
            ])
        else:
            generated_files_section.extend([
                "### Dashboard Style Plots",
                f"- `{self.output_prefix}weight_progression_analysis.png`",
                f"- `{self.output_prefix}score_progression_analysis.png`",
                f"- `{self.output_prefix}performance_comparison_analysis.png`",
                f"- `{self.output_prefix}metrics_comparison_analysis.png`",
                f"- `{self.output_prefix}analysis_report.md`",
            ])
        
        generated_files_section.extend([
            "",
            "*End of Report*"
        ])
        
        full_report.extend(generated_files_section)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_report))
        
        return report_path

    def analyze_weight_progression(self):
        """Analyze how model weights change throughout training."""
        self._add_to_report("Weight Progression Analysis", 2)
        
        if not self.checkpoints:
            self._add_to_report("❌ No checkpoints found for weight progression analysis")
            return
        
        # Extract weights for each checkpoint
        checkpoint_nums = sorted(self.checkpoints.keys())
        weight_data = {}
        
        for checkpoint_num in checkpoint_nums:
            checkpoint = self.checkpoints[checkpoint_num]
            if 'layers' not in checkpoint:
                continue
                
            weights = []
            for layer_idx, layer in enumerate(checkpoint['layers']):
                for neuron_idx, neuron in enumerate(layer['neurons']):
                    if 'w' in neuron:
                        for weight_idx, weight in enumerate(neuron['w']):
                            key = f"L{layer_idx}_N{neuron_idx}_W{weight_idx}"
                            if key not in weight_data:
                                weight_data[key] = []
                            weight_data[key].append((checkpoint_num, weight))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Weight Progression Throughout Training', fontsize=16, fontweight='bold')
        
        # Plot 1: All weights over time
        ax1 = axes[0, 0]
        for key, values in weight_data.items():
            if values:
                checkpoints_nums, weights = zip(*values)
                ax1.plot(checkpoints_nums, weights, label=key, alpha=0.7, linewidth=1)
        ax1.set_title('Individual Weight Trajectories')
        ax1.set_xlabel('Checkpoint Number')
        ax1.set_ylabel('Weight Value')
        ax1.grid(True, alpha=0.3)
        if len(weight_data) <= 10:  # Only show legend if not too many weights
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Weight statistics over time with convergence trends
        ax2 = axes[0, 1]
        weight_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        checkpoints_for_stats = []
        
        for checkpoint_num in checkpoint_nums:
            checkpoint_weights = []
            for key, values in weight_data.items():
                for cp_num, weight in values:
                    if cp_num == checkpoint_num:
                        checkpoint_weights.append(weight)
            
            if checkpoint_weights:
                weight_stats['mean'].append(np.mean(checkpoint_weights))
                weight_stats['std'].append(np.std(checkpoint_weights))
                weight_stats['min'].append(np.min(checkpoint_weights))
                weight_stats['max'].append(np.max(checkpoint_weights))
                checkpoints_for_stats.append(checkpoint_num)
        
        if checkpoints_for_stats:
            # Use moving averages for smoother weight progression visualization
            mean_series = pd.Series(weight_stats['mean'])
            std_series = pd.Series(weight_stats['std'])
            
            # Apply light smoothing to reduce noise
            window_size = min(5, len(mean_series) // 3) if len(mean_series) > 3 else 1
            if window_size > 1:
                smoothed_mean = mean_series.rolling(window=window_size, center=True).mean()
                smoothed_std = std_series.rolling(window=window_size, center=True).mean()
                ax2.plot(checkpoints_for_stats, smoothed_mean, label='Smoothed Mean', linewidth=2, color='blue')
                ax2.plot(checkpoints_for_stats, mean_series, label='Raw Mean', linewidth=1, alpha=0.5, color='lightblue')
                ax2.fill_between(checkpoints_for_stats, 
                               smoothed_mean - smoothed_std,
                               smoothed_mean + smoothed_std,
                               alpha=0.3, label='±1 Smoothed STD', color='blue')
            else:
                ax2.plot(checkpoints_for_stats, weight_stats['mean'], label='Mean', linewidth=2)
                ax2.fill_between(checkpoints_for_stats, 
                               np.array(weight_stats['mean']) - np.array(weight_stats['std']),
                               np.array(weight_stats['mean']) + np.array(weight_stats['std']),
                               alpha=0.3, label='±1 STD')
            
            # Show extremes with cumulative bounds
            cumulative_max = np.maximum.accumulate(weight_stats['max'])
            cumulative_min = np.minimum.accumulate(weight_stats['min'])
            ax2.plot(checkpoints_for_stats, cumulative_max, label='Best Weight Seen', linestyle='--', color='green')
            ax2.plot(checkpoints_for_stats, cumulative_min, label='Worst Weight Seen', linestyle='--', color='red')
        
        ax2.set_title('Weight Convergence Analysis')
        ax2.set_xlabel('Checkpoint Number')
        ax2.set_ylabel('Weight Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Weight distribution at key checkpoints
        ax3 = axes[1, 0]
        key_checkpoints = [checkpoint_nums[0], checkpoint_nums[len(checkpoint_nums)//2], checkpoint_nums[-1]]
        
        for i, checkpoint_num in enumerate(key_checkpoints):
            checkpoint_weights = []
            for key, values in weight_data.items():
                for cp_num, weight in values:
                    if cp_num == checkpoint_num:
                        checkpoint_weights.append(weight)
            
            if checkpoint_weights:
                ax3.hist(checkpoint_weights, bins=20, alpha=0.6, 
                        label=f'Checkpoint {checkpoint_num}', density=True)
        
        ax3.set_title('Weight Distribution at Key Checkpoints')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Weight stability analysis (learning rate decay visualization)
        ax4 = axes[1, 1]
        weight_changes = {}
        
        for key, values in weight_data.items():
            if len(values) > 1:
                changes = []
                for i in range(1, len(values)):
                    change = abs(values[i][1] - values[i-1][1])
                    changes.append((values[i][0], change))
                weight_changes[key] = changes
        
        # Average change per checkpoint with trend analysis
        avg_changes = {}
        for checkpoint_num in checkpoint_nums[1:]:
            changes_at_checkpoint = []
            for key, changes in weight_changes.items():
                for cp_num, change in changes:
                    if cp_num == checkpoint_num:
                        changes_at_checkpoint.append(change)
            if changes_at_checkpoint:
                avg_changes[checkpoint_num] = np.mean(changes_at_checkpoint)
        
        if avg_changes:
            checkpoints_list, changes_list = zip(*sorted(avg_changes.items()))
            
            # Plot both raw and smoothed changes
            ax4.plot(checkpoints_list, changes_list, marker='o', linewidth=1, alpha=0.6, label='Raw Change Magnitude')
            
            # Add smoothed trend line to show convergence
            if len(changes_list) > 5:
                smoothed_changes = pd.Series(changes_list).rolling(window=min(5, len(changes_list)//3), center=True).mean()
                ax4.plot(checkpoints_list, smoothed_changes, linewidth=2, color='red', label='Smoothed Trend')
            
            # Add exponential decay fit to show if weights are stabilizing
            if len(changes_list) > 10:
                try:
                    # Fit exponential decay: y = a * exp(-bx) + c
                    from scipy.optimize import curve_fit
                    def exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    
                    x_norm = np.array(checkpoints_list) - checkpoints_list[0]  # Normalize to start from 0
                    popt, _ = curve_fit(exp_decay, x_norm, changes_list, maxfev=1000)
                    if popt[1] > 0:  # Only plot if decay rate is positive
                        fitted_curve = exp_decay(x_norm, *popt)
                        ax4.plot(checkpoints_list, fitted_curve, '--', linewidth=2, color='green', 
                                label=f'Exponential Decay (λ={popt[1]:.4f})')
                except:
                    pass  # If fitting fails, just continue without it
            
            ax4.set_title('Weight Stability Analysis (Learning Convergence)')
            ax4.set_xlabel('Checkpoint Number')
            ax4.set_ylabel('Average |Weight Change|')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')  # Log scale often shows weight change trends better
        
        plt.tight_layout()
        
        # Save plots using the new method
        subplot_titles = [
            "Individual Weight Trajectories",
            "Weight Statistics Over Time", 
            "Weight Distribution at Key Checkpoints",
            "Average Weight Change Magnitude"
        ]
        
        if self.separate_plots:
            # Create individual plots
            saved_files = self._create_weight_progression_individual_plots(
                weight_data, checkpoint_nums, weight_stats, checkpoints_for_stats,
                key_checkpoints, weight_changes, avg_changes
            )
        else:
            # Save dashboard
            output_path = self.output_dir / f"{self.output_prefix}weight_progression_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files = [output_path]
            if not self.quiet:
                print(f"Saved weight progression analysis to: {output_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Add summary statistics to report
        self._add_to_report("### Weight Progression Summary")
        self._add_to_report(f"- **Total unique weights tracked**: {len(weight_data)}")
        self._add_to_report(f"- **Checkpoints analyzed**: {len(checkpoint_nums)}")
        if weight_stats['mean']:
            self._add_to_report(f"- **Final mean weight**: {weight_stats['mean'][-1]:.4f}")
            self._add_to_report(f"- **Final weight std**: {weight_stats['std'][-1]:.4f}")
            self._add_to_report(f"- **Weight range**: [{weight_stats['min'][-1]:.4f}, {weight_stats['max'][-1]:.4f}]")
            
            # Calculate convergence metrics
            if len(weight_stats['mean']) > 10:
                early_mean = np.mean(weight_stats['mean'][:5])
                late_mean = np.mean(weight_stats['mean'][-5:])
                weight_stability = abs(late_mean - early_mean) / abs(early_mean) if early_mean != 0 else 0
                self._add_to_report(f"- **Weight stability**: {weight_stability:.2%} change from early to late training")
                
                if avg_changes:
                    early_changes = np.mean(list(avg_changes.values())[:5]) if len(avg_changes) > 5 else np.mean(list(avg_changes.values()))
                    late_changes = np.mean(list(avg_changes.values())[-5:]) if len(avg_changes) > 5 else np.mean(list(avg_changes.values()))
                    if early_changes > 0:
                        change_reduction = (early_changes - late_changes) / early_changes
                        self._add_to_report(f"- **Learning convergence**: {change_reduction:.2%} reduction in weight changes")
        self._add_to_report("")

    def analyze_score_progression(self):
        """Analyze training score progression."""
        self._add_to_report("Score Progression Analysis", 2)
        
        if not self.training_history or 'history' not in self.training_history:
            self._add_to_report("❌ No training history found")
            return
        
        history = self.training_history['history']
        
        # Extract score data
        training_data = []
        for i, entry in enumerate(history):
            timestamp = entry.get('timestamp', '')
            scores = entry.get('scores', {})
            
            for agent_key, score in scores.items():
                try:
                    layer, neuron = map(int, agent_key.split('_'))
                    training_data.append({
                        'iteration': i,
                        'layer': layer,
                        'neuron': neuron,
                        'agent': agent_key,
                        'score': score,
                        'timestamp': timestamp
                    })
                except ValueError:
                    continue
        
        if not training_data:
            self._add_to_report("❌ No valid score data found in training history")
            return
        
        df = pd.DataFrame(training_data)
        
        # Create comprehensive score analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Score Progression Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Individual agent learning progress with moving averages
        ax1 = axes[0, 0]
        for agent in df['agent'].unique()[:10]:  # Limit to first 10 agents for readability
            agent_data = df[df['agent'] == agent].sort_values('iteration')
            if len(agent_data) > 0:
                # Calculate moving average (last 10 iterations) for better trend visualization
                window_size = min(10, len(agent_data) // 3) if len(agent_data) > 3 else 1
                if window_size > 1:
                    moving_avg = agent_data['score'].rolling(window=window_size, center=True).mean()
                    ax1.plot(agent_data['iteration'], moving_avg, 
                            label=f'Agent {agent}', alpha=0.8, linewidth=2)
                    # Show raw data as faint background
                    ax1.plot(agent_data['iteration'], agent_data['score'], 
                            alpha=0.3, linewidth=0.5, color=ax1.lines[-1].get_color())
                else:
                    ax1.plot(agent_data['iteration'], agent_data['score'], 
                            label=f'Agent {agent}', alpha=0.8, linewidth=1.5)
        ax1.set_title('Individual Agent Learning Progress (Moving Average)')
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Score (Moving Average)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Layer performance trends with moving averages
        ax2 = axes[0, 1]
        layer_stats = df.groupby(['iteration', 'layer'])['score'].agg(['mean', 'std']).reset_index()
        
        for layer in layer_stats['layer'].unique():
            layer_data = layer_stats[layer_stats['layer'] == layer].sort_values('iteration')
            if len(layer_data) > 0:
                # Calculate moving average for this layer (better shows recent trends)
                window_size = min(8, len(layer_data) // 3) if len(layer_data) > 5 else 1
                if window_size > 1:
                    moving_mean = layer_data['mean'].rolling(window=window_size, center=True).mean()
                    moving_std = layer_data['std'].rolling(window=window_size, center=True).mean()
                    ax2.plot(layer_data['iteration'], moving_mean, 
                            label=f'Layer {layer}', linewidth=2, marker='o', markersize=2)
                    ax2.fill_between(layer_data['iteration'],
                                   moving_mean - moving_std,
                                   moving_mean + moving_std,
                                   alpha=0.2)
                    # Add trend line
                    valid_mask = moving_mean.notna()
                    if valid_mask.sum() > 2:  # Need at least 3 points for trend line
                        valid_iterations = layer_data['iteration'][valid_mask]
                        valid_moving_mean = moving_mean[valid_mask]
                        z = np.polyfit(valid_iterations, valid_moving_mean, 1)
                        trend_line = np.poly1d(z)
                        ax2.plot(layer_data['iteration'], trend_line(layer_data['iteration']), 
                                '--', alpha=0.7, linewidth=1, color=ax2.lines[-1].get_color())
                else:
                    ax2.plot(layer_data['iteration'], layer_data['mean'], 
                            label=f'Layer {layer}', linewidth=2, marker='o', markersize=3)
        
        ax2.set_title('Layer Performance Trends (Moving Average + Trend Lines)')
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Score (Moving Average)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Score distribution evolution
        ax3 = axes[0, 2]
        iterations_to_show = [0, len(history)//3, 2*len(history)//3, len(history)-1]
        
        for i, iteration in enumerate(iterations_to_show):
            if iteration < len(history):
                iteration_scores = df[df['iteration'] == iteration]['score'].values
                if len(iteration_scores) > 0:
                    ax3.hist(iteration_scores, bins=15, alpha=0.6, 
                            label=f'Iteration {iteration}', density=True)
        
        ax3.set_title('Score Distribution Evolution')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning progress analysis with trend visualization
        ax4 = axes[1, 0]
        overall_stats = df.groupby('iteration')['score'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Use moving averages and trend analysis for clearer learning progress
        window_size = min(10, len(overall_stats) // 4) if len(overall_stats) > 8 else 1
        if window_size > 1:
            moving_mean = overall_stats['mean'].rolling(window=window_size, center=True).mean()
            moving_std = overall_stats['std'].rolling(window=window_size, center=True).mean()
            
            # Plot moving average with confidence band
            ax4.plot(overall_stats['iteration'], moving_mean, 
                    label='Moving Average Score', linewidth=3, color='blue')
            ax4.fill_between(overall_stats['iteration'],
                            moving_mean - moving_std,
                            moving_mean + moving_std,
                            alpha=0.3, color='blue', label='±1 STD')
            
            # Add overall trend line
            valid_mask = moving_mean.notna()
            if valid_mask.sum() > 3:
                valid_iterations = overall_stats['iteration'][valid_mask]
                valid_moving_mean = moving_mean[valid_mask]
                z = np.polyfit(valid_iterations, valid_moving_mean, 1)
                trend_line = np.poly1d(z)
                ax4.plot(overall_stats['iteration'], trend_line(overall_stats['iteration']), 
                        'r--', linewidth=2, alpha=0.8, 
                        label=f'Overall Trend (slope: {z[0]:.6f})')
            
            # Show raw data faintly
            ax4.plot(overall_stats['iteration'], overall_stats['mean'], 
                    alpha=0.3, linewidth=1, color='lightblue', label='Raw Mean')
        else:
            ax4.plot(overall_stats['iteration'], overall_stats['mean'], 
                    label='Mean Score', linewidth=2, color='blue')
        
        # Always show best performance envelope
        ax4.plot(overall_stats['iteration'], overall_stats['max'], 
                label='Best Performance', linestyle=':', color='green', alpha=0.7)
        
        ax4.set_title('Learning Progress Analysis')
        ax4.set_xlabel('Training Iteration')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Learning rate and improvement analysis
        ax5 = axes[1, 1]
        if len(overall_stats) > 1:
            # Calculate improvement using moving average differences for better trend visibility
            window_size = min(5, len(overall_stats) // 5) if len(overall_stats) > 10 else 1
            if window_size > 1:
                moving_mean = overall_stats['mean'].rolling(window=window_size, center=True).mean()
                score_changes = moving_mean.diff()
            else:
                score_changes = overall_stats['mean'].diff()
            
            # Plot both raw changes and smoothed trend
            ax5.plot(overall_stats['iteration'][1:], score_changes[1:], 
                    alpha=0.6, linewidth=1, marker='o', markersize=2, label='Score Changes')
            
            # Add smoothed trend line
            if len(score_changes) > 5:
                smoothed_changes = score_changes.rolling(window=min(8, len(score_changes)//3), center=True).mean()
                ax5.plot(overall_stats['iteration'][1:], smoothed_changes[1:], 
                        linewidth=3, label='Learning Trend', color='red')
            
            # Add reference lines
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Change')
            
            # Calculate and show overall learning rate
            if len(overall_stats) > 10:
                first_half_mean = overall_stats['mean'][:len(overall_stats)//2].mean()
                second_half_mean = overall_stats['mean'][len(overall_stats)//2:].mean()
                improvement = second_half_mean - first_half_mean
                ax5.axhline(y=improvement/len(overall_stats), color='green', linestyle=':', 
                           alpha=0.7, label=f'Avg Improvement: {improvement:.4f}')
            
            ax5.set_title('Learning Rate Analysis')
            ax5.set_xlabel('Training Iteration')
            ax5.set_ylabel('Score Change per Iteration')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Final score comparison
        ax6 = axes[1, 2]
        
        # Get final scores - use the last few iterations to get a more stable average
        max_iteration = df['iteration'].max()
        # Use last 3 iterations or all if less than 3
        final_iterations = max(1, min(3, len(df['iteration'].unique())))
        final_scores = df[df['iteration'] >= max_iteration - final_iterations + 1]
        
        # Group by layer for better visualization
        layer_final_scores = final_scores.groupby('layer')['score'].mean()
        
        # Debug information (only if needed for troubleshooting)
        # print(f"Debug - Final scores analysis:")
        # print(f"  Max iteration: {max_iteration}")
        # print(f"  Final iterations used: {final_iterations}")
        # print(f"  Final scores shape: {final_scores.shape}")
        # print(f"  Layer final scores:\n{layer_final_scores}")
        
        if len(layer_final_scores) > 0 and layer_final_scores.sum() > 0:
            bars = ax6.bar(range(len(layer_final_scores)), layer_final_scores.values, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(layer_final_scores))))
            ax6.set_title('Final Average Scores by Layer')
            ax6.set_xlabel('Layer')
            ax6.set_ylabel('Final Average Score')
            ax6.set_xticks(range(len(layer_final_scores)))
            ax6.set_xticklabels([f'Layer {i}' for i in layer_final_scores.index])
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        else:
            # If no valid scores, show overall score distribution instead
            overall_final_scores = final_scores['score']
            if len(overall_final_scores) > 0:
                ax6.hist(overall_final_scores, bins=min(10, len(overall_final_scores)), 
                        alpha=0.7, color='skyblue')
                ax6.set_title('Final Score Distribution')
                ax6.set_xlabel('Score')
                ax6.set_ylabel('Frequency')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No final scores available', ha='center', va='center', 
                        transform=ax6.transAxes, fontsize=12)
                ax6.set_title('Final Scores - No Data Available')
        
        plt.tight_layout()
        
        # Save plots using the new method
        if self.separate_plots:
            # Create individual plots
            saved_files = self._create_score_progression_individual_plots(df, history, overall_stats)
        else:
            # Save dashboard
            output_path = self.output_dir / f"{self.output_prefix}score_progression_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files = [output_path]
            if not self.quiet:
                print(f"Saved score progression analysis to: {output_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Add summary statistics to report
        self._add_to_report("### Score Progression Summary")
        self._add_to_report(f"- **Total training iterations**: {len(history)}")
        self._add_to_report(f"- **Number of agents**: {len(df['agent'].unique())}")
        if len(overall_stats) > 0:
            # Calculate cumulative metrics for more meaningful reporting
            cumulative_mean = overall_stats['mean'].expanding().mean()
            cumulative_std = overall_stats['std'].expanding().mean()
            cumulative_max = overall_stats['max'].expanding().max()
            
            self._add_to_report(f"- **Final cumulative mean score**: {cumulative_mean.iloc[-1]:.4f}")
            self._add_to_report(f"- **Final cumulative std**: {cumulative_std.iloc[-1]:.4f}")
            self._add_to_report(f"- **Best score achieved**: {cumulative_max.iloc[-1]:.4f}")
            self._add_to_report(f"- **Cumulative improvement**: {cumulative_mean.iloc[-1] - cumulative_mean.iloc[0]:.4f}")
            
            # Calculate learning efficiency
            if len(cumulative_mean) > 10:
                early_performance = cumulative_mean.iloc[:10].mean()
                late_performance = cumulative_mean.iloc[-10:].mean()
                improvement_rate = (late_performance - early_performance) / len(history) if len(history) > 0 else 0
                self._add_to_report(f"- **Learning efficiency**: {improvement_rate:.6f} score improvement per iteration")
                
                # Convergence analysis
                final_changes = np.diff(cumulative_mean.iloc[-10:]) if len(cumulative_mean) > 10 else np.diff(cumulative_mean)
                convergence_stability = np.std(final_changes) if len(final_changes) > 0 else 0
                self._add_to_report(f"- **Convergence stability**: {convergence_stability:.6f} (lower = more stable)")
        self._add_to_report("")

    def analyze_performance_comparison(self):
        """Analyze MOA vs Single LLM performance."""
        self._add_to_report("Performance Comparison Analysis", 2)
        
        if not self.comparison_results or 'comparison_results' not in self.comparison_results:
            self._add_to_report("❌ No comparison results found")
            return
        
        results = self.comparison_results['comparison_results']
        
        # Extract performance data
        performance_data = []
        for result in results:
            timestamp = result.get('timestamp', '')
            
            moa_data = result.get('moa', {})
            single_llm_data = result.get('single_llm', {})
            
            performance_data.append({
                'timestamp': timestamp,
                'model': 'MOA',
                'text_similarity': moa_data.get('scores', {}).get('text_similarity', 0),
                'exact_match': moa_data.get('scores', {}).get('exact_match', 0),
                'passed_text_similarity': moa_data.get('passed', {}).get('text_similarity', 0),
                'passed_exact_match': moa_data.get('passed', {}).get('exact_match', 0),
                'total_tests': moa_data.get('total', 10),
                'time_seconds': moa_data.get('time_seconds', 0)
            })
            
            performance_data.append({
                'timestamp': timestamp,
                'model': 'Single LLM',
                'text_similarity': single_llm_data.get('scores', {}).get('text_similarity', 0),
                'exact_match': single_llm_data.get('scores', {}).get('exact_match', 0),
                'passed_text_similarity': single_llm_data.get('passed', {}).get('text_similarity', 0),
                'passed_exact_match': single_llm_data.get('passed', {}).get('exact_match', 0),
                'total_tests': single_llm_data.get('total', 10),
                'time_seconds': single_llm_data.get('time_seconds', 0)
            })
        
        df = pd.DataFrame(performance_data)
        
        # Create comparison visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MOA vs Single LLM Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        metrics = ['text_similarity', 'exact_match']
        moa_scores = [df[df['model'] == 'MOA'][metric].mean() for metric in metrics]
        single_scores = [df[df['model'] == 'Single LLM'][metric].mean() for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, moa_scores, width, label='MOA', color='skyblue')
        bars2 = ax1.bar(x + width/2, single_scores, width, label='Single LLM', color='lightcoral')
        
        ax1.set_title('Average Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Text Similarity', 'Exact Match'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 2: Time comparison
        ax2 = axes[0, 1]
        moa_times = df[df['model'] == 'MOA']['time_seconds']
        single_times = df[df['model'] == 'Single LLM']['time_seconds']
        
        box_data = [moa_times.dropna(), single_times.dropna()]
        box_labels = ['MOA', 'Single LLM']
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        if len(bp['boxes']) >= 1:
            bp['boxes'][0].set_facecolor('skyblue')
        if len(bp['boxes']) >= 2:
            bp['boxes'][1].set_facecolor('lightcoral')
        
        ax2.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success rate comparison
        ax3 = axes[0, 2]
        moa_success_text = (df[df['model'] == 'MOA']['passed_text_similarity'] / 
                           df[df['model'] == 'MOA']['total_tests']).mean()
        moa_success_exact = (df[df['model'] == 'MOA']['passed_exact_match'] / 
                            df[df['model'] == 'MOA']['total_tests']).mean()
        
        single_success_text = (df[df['model'] == 'Single LLM']['passed_text_similarity'] / 
                              df[df['model'] == 'Single LLM']['total_tests']).mean()
        single_success_exact = (df[df['model'] == 'Single LLM']['passed_exact_match'] / 
                               df[df['model'] == 'Single LLM']['total_tests']).mean()
        
        success_data = {
            'MOA': [moa_success_text, moa_success_exact],
            'Single LLM': [single_success_text, single_success_exact]
        }
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, success_data['MOA'], width, label='MOA', color='skyblue')
        bars2 = ax3.bar(x + width/2, success_data['Single LLM'], width, label='Single LLM', color='lightcoral')
        
        ax3.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Success Rate')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Text Similarity', 'Exact Match'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2%}', ha='center', va='bottom')
        
        # Plot 4: Score distribution comparison
        ax4 = axes[1, 0]
        
        moa_text_scores = df[df['model'] == 'MOA']['text_similarity'].dropna()
        single_text_scores = df[df['model'] == 'Single LLM']['text_similarity'].dropna()
        
        ax4.hist(moa_text_scores, bins=20, alpha=0.6, label='MOA', density=True, color='skyblue')
        ax4.hist(single_text_scores, bins=20, alpha=0.6, label='Single LLM', density=True, color='lightcoral')
        
        ax4.set_title('Text Similarity Score Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Text Similarity Score')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Time vs Performance scatter
        ax5 = axes[1, 1]
        
        moa_df = df[df['model'] == 'MOA'].dropna(subset=['time_seconds', 'text_similarity'])
        single_df = df[df['model'] == 'Single LLM'].dropna(subset=['time_seconds', 'text_similarity'])
        
        ax5.scatter(moa_df['time_seconds'], moa_df['text_similarity'], 
                   label='MOA', alpha=0.6, color='skyblue', s=50)
        ax5.scatter(single_df['time_seconds'], single_df['text_similarity'], 
                   label='Single LLM', alpha=0.6, color='lightcoral', s=50)
        
        ax5.set_title('Performance vs Time Trade-off', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Text Similarity Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Efficiency comparison (score per second)
        ax6 = axes[1, 2]
        
        moa_efficiency = moa_df['text_similarity'] / (moa_df['time_seconds'] + 1e-6)
        single_efficiency = single_df['text_similarity'] / (single_df['time_seconds'] + 1e-6)
        
        efficiency_data = [moa_efficiency.dropna(), single_efficiency.dropna()]
        efficiency_labels = ['MOA', 'Single LLM']
        
        # Filter out extreme outliers for better visualization
        filtered_efficiency_data = []
        for data in efficiency_data:
            if len(data) > 0:
                q99 = data.quantile(0.99)
                filtered_data = data[data <= q99]
                filtered_efficiency_data.append(filtered_data)
            else:
                filtered_efficiency_data.append(data)
        
        if all(len(data) > 0 for data in filtered_efficiency_data):
            bp = ax6.boxplot(filtered_efficiency_data, labels=efficiency_labels, patch_artist=True)
            if len(bp['boxes']) >= 1:
                bp['boxes'][0].set_facecolor('skyblue')
            if len(bp['boxes']) >= 2:
                bp['boxes'][1].set_facecolor('lightcoral')
        
        ax6.set_title('Efficiency (Score/Second)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score per Second')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots using the new method
        if self.separate_plots:
            # Create individual plots
            saved_files = self._create_performance_comparison_individual_plots(df, moa_scores, single_scores, 
                                                                              moa_times, single_times, 
                                                                              moa_success_text, moa_success_exact, 
                                                                              single_success_text, single_success_exact)
        else:
            # Save dashboard
            output_path = self.output_dir / f"{self.output_prefix}performance_comparison_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files = [output_path]
            if not self.quiet:
                print(f"Saved performance comparison analysis to: {output_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Add summary statistics to report
        self._add_to_report("### Performance Comparison Summary")
        
        self._add_to_report("#### MOA Average Scores")
        self._add_to_report(f"- **Text Similarity**: {moa_scores[0]:.4f}")
        self._add_to_report(f"- **Exact Match**: {moa_scores[1]:.4f}")
        self._add_to_report(f"- **Average Time**: {moa_times.mean():.2f}s")
        self._add_to_report(f"- **Success Rate (Text Sim)**: {moa_success_text:.2%}")
        self._add_to_report(f"- **Success Rate (Exact)**: {moa_success_exact:.2%}")
        
        self._add_to_report("#### Single LLM Average Scores")
        self._add_to_report(f"- **Text Similarity**: {single_scores[0]:.4f}")
        self._add_to_report(f"- **Exact Match**: {single_scores[1]:.4f}")
        self._add_to_report(f"- **Average Time**: {single_times.mean():.2f}s")
        self._add_to_report(f"- **Success Rate (Text Sim)**: {single_success_text:.2%}")
        self._add_to_report(f"- **Success Rate (Exact)**: {single_success_exact:.2%}")
        
        self._add_to_report("#### Comparison (MOA vs Single LLM)")
        text_sim_diff = moa_scores[0] - single_scores[0]
        exact_diff = moa_scores[1] - single_scores[1]
        time_diff = moa_times.mean() - single_times.mean()
        
        self._add_to_report(f"- **Text Similarity Difference**: {text_sim_diff:+.4f} {'🟢' if text_sim_diff > 0 else '🔴' if text_sim_diff < 0 else '🟡'}")
        self._add_to_report(f"- **Exact Match Difference**: {exact_diff:+.4f} {'🟢' if exact_diff > 0 else '🔴' if exact_diff < 0 else '🟡'}")
        self._add_to_report(f"- **Time Difference**: {time_diff:+.2f}s {'🔴' if time_diff > 0 else '🟢' if time_diff < 0 else '🟡'}")
        self._add_to_report("")

    def analyze_metrics_comparison(self):
        """Compare different evaluation metrics."""
        self._add_to_report("Metrics Comparison Analysis", 2)
        
        if not self.comparison_results or 'comparison_results' not in self.comparison_results:
            self._add_to_report("❌ No comparison results found")
            return
        
        results = self.comparison_results['comparison_results']
        
        # Extract metrics data
        metrics_data = []
        for result in results:
            moa_data = result.get('moa', {})
            single_llm_data = result.get('single_llm', {})
            
            for model_name, model_data in [('MOA', moa_data), ('Single LLM', single_llm_data)]:
                scores = model_data.get('scores', {})
                metrics_data.append({
                    'model': model_name,
                    'text_similarity': scores.get('text_similarity', 0),
                    'exact_match': scores.get('exact_match', 0)
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Create metrics comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Evaluation Metrics Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation between metrics
        ax1 = axes[0, 0]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax1.scatter(model_data['text_similarity'], model_data['exact_match'], 
                       label=model, alpha=0.6, s=50)
        
        # Add correlation line
        correlation = np.corrcoef(df['text_similarity'], df['exact_match'])[0, 1]
        if not np.isnan(correlation):
            z = np.polyfit(df['text_similarity'], df['exact_match'], 1)
            p = np.poly1d(z)
            ax1.plot(sorted(df['text_similarity']), p(sorted(df['text_similarity'])), 
                    "r--", alpha=0.8, label=f'Correlation: {correlation:.3f}')
        
        ax1.set_title('Text Similarity vs Exact Match Correlation', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Text Similarity Score')
        ax1.set_ylabel('Exact Match Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Metric agreement analysis
        ax2 = axes[0, 1]
        
        # Calculate agreement categories
        agreement_data = []
        for _, row in df.iterrows():
            text_sim = row['text_similarity']
            exact_match = row['exact_match']
            
            if text_sim > 0.5 and exact_match > 0.5:
                category = 'Both High'
            elif text_sim <= 0.5 and exact_match <= 0.5:
                category = 'Both Low'
            elif text_sim > 0.5 and exact_match <= 0.5:
                category = 'Similarity High, Exact Low'
            else:
                category = 'Similarity Low, Exact High'
            
            agreement_data.append({
                'model': row['model'],
                'category': category
            })
        
        agreement_df = pd.DataFrame(agreement_data)
        agreement_counts = agreement_df.groupby(['model', 'category']).size().unstack(fill_value=0)
        
        if len(agreement_counts) > 0:
            agreement_counts.plot(kind='bar', ax=ax2, stacked=True)
            ax2.set_title('Metric Agreement Patterns', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Count')
            ax2.legend(title='Agreement Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Metric sensitivity analysis
        ax3 = axes[1, 0]
        
        # Calculate differences between metrics for each model
        df['metric_diff'] = df['text_similarity'] - df['exact_match']
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            if len(model_data) > 0:
                ax3.hist(model_data['metric_diff'], bins=20, alpha=0.6, 
                        label=model, density=True)
        
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Agreement')
        ax3.set_title('Metric Difference Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Text Similarity - Exact Match')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance by metric threshold
        ax4 = axes[1, 1]
        
        thresholds = np.linspace(0, 1, 11)
        threshold_performance = {'MOA': {'text': [], 'exact': []}, 
                               'Single LLM': {'text': [], 'exact': []}}
        
        for threshold in thresholds:
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                if len(model_data) > 0:
                    text_above = (model_data['text_similarity'] >= threshold).mean()
                    exact_above = (model_data['exact_match'] >= threshold).mean()
                    
                    if model == 'MOA':
                        threshold_performance['MOA']['text'].append(text_above)
                        threshold_performance['MOA']['exact'].append(exact_above)
                    else:
                        threshold_performance['Single LLM']['text'].append(text_above)
                        threshold_performance['Single LLM']['exact'].append(exact_above)
        
        ax4.plot(thresholds, threshold_performance['MOA']['text'], 
                label='MOA Text Similarity', linewidth=2, marker='o')
        ax4.plot(thresholds, threshold_performance['MOA']['exact'], 
                label='MOA Exact Match', linewidth=2, marker='s')
        ax4.plot(thresholds, threshold_performance['Single LLM']['text'], 
                label='Single LLM Text Similarity', linewidth=2, marker='^', linestyle='--')
        ax4.plot(thresholds, threshold_performance['Single LLM']['exact'], 
                label='Single LLM Exact Match', linewidth=2, marker='d', linestyle='--')
        
        ax4.set_title('Performance by Score Threshold', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Score Threshold')
        ax4.set_ylabel('Fraction Above Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots using the new method
        if self.separate_plots:
            # Create individual plots
            saved_files = self._create_metrics_comparison_individual_plots(df, correlation, agreement_df, thresholds, threshold_performance)
        else:
            # Save dashboard
            output_path = self.output_dir / f"{self.output_prefix}metrics_comparison_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            saved_files = [output_path]
            if not self.quiet:
                print(f"Saved metrics comparison analysis to: {output_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Add summary statistics to report
        self._add_to_report("### Metrics Comparison Summary")
        if not np.isnan(correlation):
            self._add_to_report(f"- **Correlation between Text Similarity and Exact Match**: {correlation:.4f}")
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            if len(model_data) > 0:
                self._add_to_report(f"#### {model}")
                self._add_to_report(f"- **Text Similarity**: μ={model_data['text_similarity'].mean():.4f}, σ={model_data['text_similarity'].std():.4f}")
                self._add_to_report(f"- **Exact Match**: μ={model_data['exact_match'].mean():.4f}, σ={model_data['exact_match'].std():.4f}")
                self._add_to_report(f"- **Metric Difference**: μ={model_data['metric_diff'].mean():.4f}, σ={model_data['metric_diff'].std():.4f}")
        self._add_to_report("")

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        # Add main title to report
        self._add_to_report("Comprehensive MOA Training Analysis Report", 1)
        self._add_to_report(f"*Analysis performed on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")
        self._add_to_report("")
        
        # Run all analyses
        self.analyze_weight_progression()
        self.analyze_score_progression()
        self.analyze_performance_comparison()
        self.analyze_metrics_comparison()
        
        # Run new detailed analyses
        self.analyze_test_case_performance()
        self.analyze_problem_level_performance()
        
        # Generate summary insights
        self._add_to_report("Key Insights and Recommendations", 2)
        
        insights = []
        recommendations = []
        
        # Training convergence insights
        if self.training_history and 'history' in self.training_history:
            history = self.training_history['history']
            if len(history) > 10:
                early_scores = [entry.get('scores', {}) for entry in history[:5]]
                late_scores = [entry.get('scores', {}) for entry in history[-5:]]
                
                early_avg = np.mean([score for entry in early_scores for score in entry.values() if score is not None])
                late_avg = np.mean([score for entry in late_scores for score in entry.values() if score is not None])
                
                if late_avg > early_avg * 1.1:
                    insights.append("✅ Training shows positive convergence - model is learning effectively")
                elif late_avg < early_avg * 0.9:
                    insights.append("⚠️ Training shows negative trend - consider adjusting hyperparameters")
                    recommendations.append("Investigate hyperparameter settings and consider reducing learning rate")
                else:
                    insights.append("ℹ️ Training shows stable performance - model may have converged")
                    recommendations.append("Consider early stopping to prevent overfitting")
        
        # Performance comparison insights
        if self.comparison_results and 'comparison_results' in self.comparison_results:
            results = self.comparison_results['comparison_results']
            if results:
                moa_scores = [r.get('moa', {}).get('scores', {}) for r in results]
                single_scores = [r.get('single_llm', {}).get('scores', {}) for r in results]
                
                moa_text_avg = np.mean([s.get('text_similarity', 0) for s in moa_scores])
                single_text_avg = np.mean([s.get('text_similarity', 0) for s in single_scores])
                
                moa_times = [r.get('moa', {}).get('time_seconds', 0) for r in results]
                single_times = [r.get('single_llm', {}).get('time_seconds', 0) for r in results]
                
                moa_time_avg = np.mean([t for t in moa_times if t > 0])
                single_time_avg = np.mean([t for t in single_times if t > 0])
                
                if moa_text_avg > single_text_avg * 1.05:
                    insights.append("✅ MOA shows superior accuracy compared to single LLM")
                
                if moa_time_avg > single_time_avg * 2:
                    insights.append("⚠️ MOA has significantly higher latency - consider optimization")
                    recommendations.append("Optimize inference pipeline or reduce ensemble size for better latency")
                elif moa_time_avg > single_time_avg * 1.2:
                    insights.append("ℹ️ MOA has moderately higher latency - acceptable trade-off for accuracy")
        
        # Add insights to report
        if insights:
            self._add_to_report("### Key Insights")
            for insight in insights:
                self._add_to_report(f"- {insight}")
            self._add_to_report("")
        
        # Add recommendations to report
        self._add_to_report("### Recommendations")
        default_recommendations = [
            "Monitor weight convergence to prevent overfitting",
            "Consider ensemble size vs. latency trade-offs",
            "Evaluate metric choice based on use case requirements",
            "Implement early stopping if training plateaus",
            "Consider adaptive learning rates for better convergence"
        ]
        
        all_recommendations = recommendations + default_recommendations
        for i, rec in enumerate(all_recommendations, 1):
            self._add_to_report(f"{i}. {rec}")
        
        self._add_to_report("")
        
        # Save the markdown report
        report_path = self._save_report()
        
        return report_path

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis script for MOA training results and performance comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths relative to current directory
  python analyze_training_results.py
  
  # Specify custom base directory
  python analyze_training_results.py --base-dir /path/to/project
  
  # Specify individual file paths
  python analyze_training_results.py --final-model /path/to/final_model.json \\
                                     --comparison-results /path/to/comparison_results.json
  
  # Save outputs with custom prefix and directory
  python analyze_training_results.py --output-dir ./results --output-prefix experiment1_
        """
    )
    
    parser.add_argument('--base-dir', '-bd', type=str, default=None,
                        help='Base directory containing the project (default: current directory)')
    
    parser.add_argument('--training-output', '-to', type=str, default=None,
                        help='Path to training_output directory')
    
    parser.add_argument('--checkpoint-dir', '-cd', type=str, default=None,
                        help='Path to specific checkpoint directory (default: auto-detect most recent)')
    
    parser.add_argument('--final-model', '-fm', type=str, default=None,
                        help='Path to final_model.json file')
    
    parser.add_argument('--training-history', '-th', type=str, default=None,
                        help='Path to training_history.json file')
    
    parser.add_argument('--comparison-results', '-cr', type=str, default=None,
                        help='Path to comparison_results.json file')
    
    parser.add_argument('--test-cases-dir', '-tc', type=str, default=None,
                        help='Path to test_cases directory')
    
    parser.add_argument('--output-dir', '-od', type=str, default='.',
                        help='Directory to save output plots (default: current directory)')
    
    parser.add_argument('--output-prefix', '-op', type=str, default='',
                        help='Prefix for output filenames (default: empty)')
    
    parser.add_argument('--no-display', '-nd', action='store_true',
                        help='Do not display plots (useful for headless environments)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce verbosity of output')
    
    parser.add_argument('--separate-plots', '-sp', action='store_true',
                        help='Create separate files for each plot instead of dashboard style')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Configure matplotlib for headless environments if needed
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')
    
    try:
        analyzer = MOATrainingAnalyzer(
            base_dir=args.base_dir,
            training_output_dir=args.training_output,
            checkpoint_dir=args.checkpoint_dir,
            final_model_path=args.final_model,
            training_history_path=args.training_history,
            comparison_results_path=args.comparison_results,
            test_cases_dir=args.test_cases_dir,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            show_plots=not args.no_display,
            quiet=args.quiet,
            separate_plots=args.separate_plots
        )
        
        if not args.quiet:
            print("\n🚀 Starting comprehensive MOA training analysis...")
        
        report_path = analyzer.generate_comprehensive_report()
        
        if not args.quiet:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Visualizations saved to: {analyzer.output_dir}")
            print(f"📄 Report saved to: {report_path}")
            if args.output_prefix:
                print(f"🏷️  File prefix: {args.output_prefix}")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)
