#!/usr/bin/env python3
"""
Comprehensive analysis script for MOA training results and performance comparison.

This script analyzes:
- Model parameter (weights) progression throughout training
- Score progression and convergence patterns
- Performance comparison between MOA and single LLM
- Time efficiency analysis
- Different evaluation metrics comparison
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
                 quiet: bool = False):
        
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
        full_report.extend([
            "",
            "---",
            "",
            "## Generated Files",
            f"- `{self.output_prefix}weight_progression_analysis.png`",
            f"- `{self.output_prefix}score_progression_analysis.png`",
            f"- `{self.output_prefix}performance_comparison_analysis.png`",
            f"- `{self.output_prefix}metrics_comparison_analysis.png`",
            f"- `{self.output_prefix}analysis_report.md`",
            "",
            "*End of Report*"
        ])
        
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
        
        # Plot 2: Weight statistics over time
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
            ax2.plot(checkpoints_for_stats, weight_stats['mean'], label='Mean', linewidth=2)
            ax2.fill_between(checkpoints_for_stats, 
                           np.array(weight_stats['mean']) - np.array(weight_stats['std']),
                           np.array(weight_stats['mean']) + np.array(weight_stats['std']),
                           alpha=0.3, label='±1 STD')
            ax2.plot(checkpoints_for_stats, weight_stats['min'], label='Min', linestyle='--')
            ax2.plot(checkpoints_for_stats, weight_stats['max'], label='Max', linestyle='--')
        
        ax2.set_title('Weight Statistics Over Time')
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
        
        # Plot 4: Weight change velocity
        ax4 = axes[1, 1]
        weight_changes = {}
        
        for key, values in weight_data.items():
            if len(values) > 1:
                changes = []
                for i in range(1, len(values)):
                    change = abs(values[i][1] - values[i-1][1])
                    changes.append((values[i][0], change))
                weight_changes[key] = changes
        
        # Average change per checkpoint
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
            ax4.plot(checkpoints_list, changes_list, marker='o', linewidth=2)
            ax4.set_title('Average Weight Change Magnitude')
            ax4.set_xlabel('Checkpoint Number')
            ax4.set_ylabel('Average |Weight Change|')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.output_prefix}weight_progression_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        
        # Plot 1: Individual agent scores over time
        ax1 = axes[0, 0]
        for agent in df['agent'].unique()[:10]:  # Limit to first 10 agents for readability
            agent_data = df[df['agent'] == agent]
            ax1.plot(agent_data['iteration'], agent_data['score'], 
                    label=f'Agent {agent}', alpha=0.8, linewidth=1.5)
        ax1.set_title('Individual Agent Score Trajectories')
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average scores by layer
        ax2 = axes[0, 1]
        layer_stats = df.groupby(['iteration', 'layer'])['score'].agg(['mean', 'std']).reset_index()
        
        for layer in layer_stats['layer'].unique():
            layer_data = layer_stats[layer_stats['layer'] == layer]
            ax2.plot(layer_data['iteration'], layer_data['mean'], 
                    label=f'Layer {layer}', linewidth=2, marker='o', markersize=3)
            ax2.fill_between(layer_data['iteration'],
                           layer_data['mean'] - layer_data['std'],
                           layer_data['mean'] + layer_data['std'],
                           alpha=0.2)
        
        ax2.set_title('Average Scores by Layer')
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Average Score')
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
        
        # Plot 4: Convergence analysis
        ax4 = axes[1, 0]
        overall_stats = df.groupby('iteration')['score'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        ax4.plot(overall_stats['iteration'], overall_stats['mean'], 
                label='Mean Score', linewidth=2, color='blue')
        ax4.fill_between(overall_stats['iteration'],
                        overall_stats['mean'] - overall_stats['std'],
                        overall_stats['mean'] + overall_stats['std'],
                        alpha=0.3, color='blue', label='±1 STD')
        ax4.plot(overall_stats['iteration'], overall_stats['max'], 
                label='Max Score', linestyle='--', color='green')
        ax4.plot(overall_stats['iteration'], overall_stats['min'], 
                label='Min Score', linestyle='--', color='red')
        
        ax4.set_title('Overall Score Statistics')
        ax4.set_xlabel('Training Iteration')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Score improvement rate
        ax5 = axes[1, 1]
        if len(overall_stats) > 1:
            score_changes = np.diff(overall_stats['mean'])
            ax5.plot(overall_stats['iteration'][1:], score_changes, 
                    marker='o', linewidth=2, markersize=4)
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax5.set_title('Score Improvement Rate')
            ax5.set_xlabel('Training Iteration')
            ax5.set_ylabel('Score Change')
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
        output_path = self.output_dir / f"{self.output_prefix}score_progression_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
            self._add_to_report(f"- **Final mean score**: {overall_stats['mean'].iloc[-1]:.4f}")
            self._add_to_report(f"- **Final score std**: {overall_stats['std'].iloc[-1]:.4f}")
            self._add_to_report(f"- **Best final score**: {overall_stats['max'].iloc[-1]:.4f}")
            self._add_to_report(f"- **Score improvement**: {overall_stats['mean'].iloc[-1] - overall_stats['mean'].iloc[0]:.4f}")
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
        
        ax1.set_title('Average Accuracy Comparison')
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
        
        ax2.set_title('Response Time Distribution')
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
        
        ax3.set_title('Success Rate Comparison')
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
        
        ax4.set_title('Text Similarity Score Distribution')
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
        
        ax5.set_title('Performance vs Time Trade-off')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Text Similarity Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Efficiency comparison (score per second)
        ax6 = axes[1, 2]
        
        moa_efficiency = moa_df['text_similarity'] / (moa_df['time_seconds'] + 1e-6)  # Avoid division by zero
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
        
        ax6.set_title('Efficiency (Score/Second)')
        ax6.set_ylabel('Score per Second')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.output_prefix}performance_comparison_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        
        ax1.set_title('Text Similarity vs Exact Match Correlation')
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
            ax2.set_title('Metric Agreement Patterns')
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
        ax3.set_title('Metric Difference Distribution')
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
        
        ax4.set_title('Performance by Score Threshold')
        ax4.set_xlabel('Score Threshold')
        ax4.set_ylabel('Fraction Above Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{self.output_prefix}metrics_comparison_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
            quiet=args.quiet
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
