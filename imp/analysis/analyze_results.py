#!/usr/bin/env python3
"""
Modern Analysis Tool for Mixture-of-Agents Training and Comparison Results
Author: Research Assistant
Version: 2.0

This tool provides comprehensive analysis and modern visualizations for:
- Training progression and model evolution
- Performance comparisons between MOA and Single LLM
- Statistical analysis and insights
- Professional visualizations for academic presentations
"""

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Modern styling
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="husl")

class MOAAnalyzer:
    """Modern analyzer for MOA training and comparison results"""
    
    def __init__(self):
        self.colors = {
            'moa': '#2E8B57',      # Sea Green
            'single_llm': '#B22222', # Fire Brick
            'accent': '#FF6347',    # Tomato
            'neutral': '#708090',   # Slate Gray
            'success': '#32CD32',   # Lime Green
            'warning': '#FFD700',   # Gold
            'error': '#DC143C'      # Crimson
        }
        
        self.style = {
            'figure_size': (14, 10),
            'dpi': 300,
            'font_size': 12,
            'title_size': 16,
            'label_size': 14,
            'line_width': 2.5,
            'marker_size': 8,
            'alpha': 0.8
        }
        
        plt.rcParams.update({
            'font.size': self.style['font_size'],
            'axes.titlesize': self.style['title_size'],
            'axes.labelsize': self.style['label_size'],
            'figure.figsize': self.style['figure_size'],
            'figure.dpi': self.style['dpi'],
            'lines.linewidth': self.style['line_width'],
            'lines.markersize': self.style['marker_size']
        })
    
    def load_comparison_results(self, filepath: str) -> pd.DataFrame:
        """Load and parse comparison results into a structured DataFrame"""
        # try:
        if 1 == 1:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            results = []
            for i, result in enumerate(data.get("comparison_results", [])):
                moa_data = result.get("moa", {})
                sllm_data = result.get("single_llm", {})
                
                # Extract scores
                moa_scores = moa_data.get("scores", {})
                sllm_scores = sllm_data.get("scores", {})
                
                moa_score = np.mean(list(moa_scores.values())) if isinstance(moa_scores, dict) else moa_scores
                sllm_score = np.mean(list(sllm_scores.values())) if isinstance(sllm_scores, dict) else sllm_scores
                
                # Extract test results
                moa_passed = moa_data.get("passed", {})
                sllm_passed = sllm_data.get("passed", {})
                
                moa_passed_count = sum(moa_passed.values()) if isinstance(moa_passed, dict) else moa_passed
                sllm_passed_count = sum(sllm_passed.values()) if isinstance(sllm_passed, dict) else sllm_passed
                
                results.append({
                    'problem_id': i + 1,
                    'moa_score': moa_score,
                    'sllm_score': sllm_score,
                    'moa_time': moa_data.get("time_seconds", 0),
                    'sllm_time': sllm_data.get("time_seconds", 0),
                    'moa_passed': moa_passed_count,
                    'sllm_passed': sllm_passed_count,
                    'total_tests': moa_data.get("total", 0),
                    'performance_gap': (moa_score - sllm_score) * 100,
                    'time_diff': moa_data.get("time_seconds", 0) - sllm_data.get("time_seconds", 0),
                    'moa_efficiency': moa_score / moa_data.get("time_seconds", 1),
                    'sllm_efficiency': sllm_score / sllm_data.get("time_seconds", 1)
                })
            
            return pd.DataFrame(results)
            
        # except Exception as e:
        #     print(f"❌ Error loading comparison results: {e}")
        #     return pd.DataFrame()
    
    def load_training_history(self, filepath: str) -> pd.DataFrame:
        """Load and parse training history into a DataFrame"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            history = []
            for i, entry in enumerate(data.get("history", [])):
                scores = entry.get("scores", {})
                avg_score = np.mean(list(scores.values())) if scores else 0
                
                history.append({
                    'iteration': i + 1,
                    'avg_score': avg_score,
                    'timestamp': entry.get("timestamp", ""),
                    'problem': entry.get("problem", "")[:50] + "..." if len(entry.get("problem", "")) > 50 else entry.get("problem", ""),
                    **{f'neuron_{k}': v for k, v in scores.items()}
                })
            
            return pd.DataFrame(history)
            
        except Exception as e:
            print(f"❌ Error loading training history: {e}")
            return pd.DataFrame()
    
    def load_model_checkpoints(self, checkpoint_dir: str) -> pd.DataFrame:
        """Load model checkpoints and extract weight progression"""
        try:
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_checkpoint_*.json"))
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            checkpoints = []
            for filepath in checkpoint_files:
                with open(filepath, 'r') as f:
                    checkpoint = json.load(f)
                
                checkpoint_num = int(os.path.basename(filepath).split('_')[-1].split('.')[0])
                
                # Extract weights for each layer and neuron
                weights_data = {'checkpoint': checkpoint_num}
                
                for layer_idx, layer in enumerate(checkpoint.get('layers', [])):
                    for neuron_idx, neuron in enumerate(layer.get('neurons', [])):
                        weights = neuron.get('w', [])
                        if weights:
                            key = f'layer_{layer_idx}_neuron_{neuron_idx}'
                            weights_data[f'{key}_weight_0'] = weights[0] if len(weights) > 0 else 0
                            weights_data[f'{key}_weight_avg'] = np.mean(weights)
                            weights_data[f'{key}_weight_std'] = np.std(weights)
                            weights_data[f'{key}_weight_count'] = len(weights)
                
                checkpoints.append(weights_data)
            
            return pd.DataFrame(checkpoints)
            
        except Exception as e:
            print(f"❌ Error loading checkpoints: {e}")
            return pd.DataFrame()
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive summary statistics"""
        if df.empty:
            return {}
        
        stats = {
            'total_problems': len(df),
            'total_tests': df['total_tests'].sum(),
            'moa': {
                'total_passed': df['moa_passed'].sum(),
                'avg_score': df['moa_score'].mean(),
                'median_score': df['moa_score'].median(),
                'std_score': df['moa_score'].std(),
                'avg_time': df['moa_time'].mean(),
                'total_time': df['moa_time'].sum(),
                'wins': len(df[df['moa_score'] > df['sllm_score']]),
                'avg_efficiency': df['moa_efficiency'].mean()
            },
            'single_llm': {
                'total_passed': df['sllm_passed'].sum(),
                'avg_score': df['sllm_score'].mean(),
                'median_score': df['sllm_score'].median(),
                'std_score': df['sllm_score'].std(),
                'avg_time': df['sllm_time'].mean(),
                'total_time': df['sllm_time'].sum(),
                'wins': len(df[df['sllm_score'] > df['moa_score']]),
                'avg_efficiency': df['sllm_efficiency'].mean()
            },
            'ties': len(df[df['moa_score'] == df['sllm_score']]),
            'performance_gap': {
                'mean': df['performance_gap'].mean(),
                'median': df['performance_gap'].median(),
                'std': df['performance_gap'].std()
            },
            'time_difference': {
                'mean': df['time_diff'].mean(),
                'median': df['time_diff'].median(),
                'std': df['time_diff'].std()
            }
        }
        
        return stats
    
    def print_executive_summary(self, stats: Dict, training_df: pd.DataFrame = None):
        """Print a clean, executive-style summary"""
        if not stats:
            print("❌ No statistics available")
            return
        
        print("\n" + "="*80)
        print("🎯 MIXTURE-OF-AGENTS PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Key Performance Indicators
        print(f"\n📊 KEY PERFORMANCE INDICATORS")
        print(f"┌─────────────────────────────────────────────────────────────────┐")
        print(f"│ Total Problems Analyzed: {stats['total_problems']:>4} │ Total Test Cases: {stats['total_tests']:>8} │")
        print(f"├─────────────────────────────────────────────────────────────────┤")
        print(f"│ MOA Success Rate:     {stats['moa']['avg_score']:>7.1%} │ SingleLLM Success Rate: {stats['single_llm']['avg_score']:>7.1%} │")
        print(f"│ MOA Avg Time:         {stats['moa']['avg_time']:>7.1f}s │ SingleLLM Avg Time:     {stats['single_llm']['avg_time']:>7.1f}s │")
        print(f"│ MOA Efficiency:       {stats['moa']['avg_efficiency']:>7.4f} │ SingleLLM Efficiency:   {stats['single_llm']['avg_efficiency']:>7.4f} │")
        print(f"└─────────────────────────────────────────────────────────────────┘")
        
        # Performance Verdict
        performance_leader = "MOA" if stats['performance_gap']['mean'] > 0 else "SingleLLM"
        speed_leader = "MOA" if stats['time_difference']['mean'] < 0 else "SingleLLM"
        efficiency_leader = "MOA" if stats['moa']['avg_efficiency'] > stats['single_llm']['avg_efficiency'] else "SingleLLM"
        
        print(f"\n🏆 PERFORMANCE VERDICT")
        print(f"   • Accuracy Winner:   {performance_leader} ({abs(stats['performance_gap']['mean']):.2f}pp advantage)")
        print(f"   • Speed Winner:      {speed_leader} ({abs(stats['time_difference']['mean']):.2f}s faster)")
        print(f"   • Efficiency Winner: {efficiency_leader}")
        
        # Win/Loss Record
        total_decisive = stats['moa']['wins'] + stats['single_llm']['wins']
        print(f"\n🥊 WIN/LOSS RECORD")
        print(f"   • MOA Wins:        {stats['moa']['wins']:>3}/{total_decisive} ({stats['moa']['wins']/total_decisive*100:.1f}%)")
        print(f"   • SingleLLM Wins:  {stats['single_llm']['wins']:>3}/{total_decisive} ({stats['single_llm']['wins']/total_decisive*100:.1f}%)")
        print(f"   • Ties:            {stats['ties']:>3}/{stats['total_problems']}")
        
        # Training Summary (if available)
        if training_df is not None and not training_df.empty:
            initial_score = training_df['avg_score'].iloc[0]
            final_score = training_df['avg_score'].iloc[-1]
            improvement = final_score - initial_score
            
            print(f"\n📈 TRAINING PERFORMANCE")
            print(f"   • Training Iterations: {len(training_df)}")
            print(f"   • Initial Score:       {initial_score:.2%}")
            print(f"   • Final Score:         {final_score:.2%}")
            print(f"   • Total Improvement:   {improvement:.2%} ({improvement*100:.2f}pp)")
        
        print("="*80)
    
    def create_success_rate_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create success rate comparison plot"""
        if df.empty:
            return None
        
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        models = ['MOA', 'SingleLLM']
        scores = [df['moa_score'].mean() * 100, df['sllm_score'].mean() * 100]
        bars = ax.bar(models, scores, color=[self.colors['moa'], self.colors['single_llm']], 
                     alpha=self.style['alpha'], edgecolor='white', linewidth=2)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_title('Success Rate Comparison', fontweight='bold', pad=20)
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Success rate plot saved: {output_path}")
            return None
        else:
            return fig

    def create_execution_time_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create execution time comparison plot"""
        if df.empty:
            return None
        
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        models = ['MOA', 'SingleLLM']
        times = [df['moa_time'].mean(), df['sllm_time'].mean()]
        bars = ax.bar(models, times, color=[self.colors['moa'], self.colors['single_llm']], 
                     alpha=self.style['alpha'], edgecolor='white', linewidth=2)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{time:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_title('Average Execution Time', fontweight='bold', pad=20)
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Execution time plot saved: {output_path}")
            return None
        else:
            return fig

    def create_win_distribution_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create win distribution pie chart"""
        if df.empty:
            return None
        
        figsize = figsize or (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        wins_data = [
            len(df[df['moa_score'] > df['sllm_score']]),
            len(df[df['sllm_score'] > df['moa_score']]),
            len(df[df['moa_score'] == df['sllm_score']])
        ]
        
        wedges, texts, autotexts = ax.pie(wins_data, 
                                         labels=['MOA Wins', 'SingleLLM Wins', 'Ties'],
                                         colors=[self.colors['moa'], self.colors['single_llm'], self.colors['neutral']],
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontweight': 'bold'})
        ax.set_title('Win Distribution', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Win distribution plot saved: {output_path}")
            return None
        else:
            return fig

    def create_efficiency_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create efficiency comparison plot"""
        if df.empty:
            return None
        
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        models = ['MOA', 'SingleLLM']
        efficiencies = [df['moa_efficiency'].mean(), df['sllm_efficiency'].mean()]
        bars = ax.bar(models, efficiencies, color=[self.colors['moa'], self.colors['single_llm']], 
                     alpha=self.style['alpha'], edgecolor='white', linewidth=2)
        
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{eff:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title('Efficiency (Score/Time)', fontweight='bold', pad=20)
        ax.set_ylabel('Efficiency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Efficiency plot saved: {output_path}")
            return None
        else:
            return fig

    def create_performance_gap_distribution_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create performance gap distribution plot"""
        if df.empty:
            return None
        
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(df['performance_gap'], bins=20, alpha=0.7, color=self.colors['accent'], 
               edgecolor='white', linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(x=df['performance_gap'].mean(), color=self.colors['error'], 
                  linestyle='-', linewidth=3, label=f'Mean: {df["performance_gap"].mean():.2f}pp')
        ax.set_title('Performance Gap Distribution (MOA - SingleLLM)', fontweight='bold', pad=20)
        ax.set_xlabel('Performance Gap (percentage points)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Performance gap distribution plot saved: {output_path}")
            return None
        else:
            return fig

    def create_time_difference_distribution_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create time difference distribution plot"""
        if df.empty:
            return None
        
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(df['time_diff'], bins=20, alpha=0.7, color=self.colors['warning'], 
               edgecolor='white', linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(x=df['time_diff'].mean(), color=self.colors['error'], 
                  linestyle='-', linewidth=3, label=f'Mean: {df["time_diff"].mean():.2f}s')
        ax.set_title('Time Difference Distribution (MOA - SingleLLM)', fontweight='bold', pad=20)
        ax.set_xlabel('Time Difference (seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Time difference distribution plot saved: {output_path}")
            return None
        else:
            return fig

    def create_problem_by_problem_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create problem-by-problem performance comparison plot"""
        if df.empty:
            return None
        
        figsize = figsize or (16, 8)
        fig, ax = plt.subplots(figsize=figsize)
        x = df['problem_id']
        ax.plot(x, df['moa_score'] * 100, 'o-', color=self.colors['moa'], 
               linewidth=2, markersize=6, label='MOA', alpha=self.style['alpha'])
        ax.plot(x, df['sllm_score'] * 100, 's-', color=self.colors['single_llm'], 
               linewidth=2, markersize=6, label='SingleLLM', alpha=self.style['alpha'])
        ax.set_title('Problem-by-Problem Performance Comparison', fontweight='bold', pad=20)
        ax.set_xlabel('Problem Number')
        ax.set_ylabel('Success Rate (%)')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Problem-by-problem plot saved: {output_path}")
            return None
        else:
            return fig

    def create_performance_dashboard(self, df: pd.DataFrame, output_dir: str):
        """Create a comprehensive performance dashboard"""
        if df.empty:
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('MOA vs Single LLM Performance Dashboard', 
                    fontsize=24, fontweight='bold', y=0.96)
        
        # 1. Success Rate Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        models = ['MOA', 'SingleLLM']
        scores = [df['moa_score'].mean() * 100, df['sllm_score'].mean() * 100]
        bars = ax1.bar(models, scores, color=[self.colors['moa'], self.colors['single_llm']], 
                      alpha=self.style['alpha'], edgecolor='white', linewidth=2)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.set_title('Success Rate Comparison', fontweight='bold', pad=20)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # 2. Execution Time Comparison (Top Center-Left)
        ax2 = fig.add_subplot(gs[0, 1])
        times = [df['moa_time'].mean(), df['sllm_time'].mean()]
        bars = ax2.bar(models, times, color=[self.colors['moa'], self.colors['single_llm']], 
                      alpha=self.style['alpha'], edgecolor='white', linewidth=2)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax2.set_title('Average Execution Time', fontweight='bold', pad=20)
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win Distribution (Top Center-Right)
        ax3 = fig.add_subplot(gs[0, 2])
        wins_data = [
            len(df[df['moa_score'] > df['sllm_score']]),
            len(df[df['sllm_score'] > df['moa_score']]),
            len(df[df['moa_score'] == df['sllm_score']])
        ]
        
        wedges, texts, autotexts = ax3.pie(wins_data, 
                                          labels=['MOA Wins', 'SingleLLM Wins', 'Ties'],
                                          colors=[self.colors['moa'], self.colors['single_llm'], self.colors['neutral']],
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontweight': 'bold'})
        ax3.set_title('Win Distribution', fontweight='bold', pad=20)
        
        # 4. Efficiency Comparison (Top Right)
        ax4 = fig.add_subplot(gs[0, 3])
        efficiencies = [df['moa_efficiency'].mean(), df['sllm_efficiency'].mean()]
        bars = ax4.bar(models, efficiencies, color=[self.colors['moa'], self.colors['single_llm']], 
                      alpha=self.style['alpha'], edgecolor='white', linewidth=2)
        
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{eff:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax4.set_title('Efficiency (Score/Time)', fontweight='bold', pad=20)
        ax4.set_ylabel('Efficiency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Gap Distribution (Second Row Left)
        ax5 = fig.add_subplot(gs[1, 0:2])
        ax5.hist(df['performance_gap'], bins=20, alpha=0.7, color=self.colors['accent'], 
                edgecolor='white', linewidth=1)
        ax5.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax5.axvline(x=df['performance_gap'].mean(), color=self.colors['error'], 
                   linestyle='-', linewidth=3, label=f'Mean: {df["performance_gap"].mean():.2f}pp')
        ax5.set_title('Performance Gap Distribution (MOA - SingleLLM)', fontweight='bold', pad=20)
        ax5.set_xlabel('Performance Gap (percentage points)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Time Difference Distribution (Second Row Right)
        ax6 = fig.add_subplot(gs[1, 2:4])
        ax6.hist(df['time_diff'], bins=20, alpha=0.7, color=self.colors['warning'], 
                edgecolor='white', linewidth=1)
        ax6.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax6.axvline(x=df['time_diff'].mean(), color=self.colors['error'], 
                   linestyle='-', linewidth=3, label=f'Mean: {df["time_diff"].mean():.2f}s')
        ax6.set_title('Time Difference Distribution (MOA - SingleLLM)', fontweight='bold', pad=20)
        ax6.set_xlabel('Time Difference (seconds)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Problem-by-Problem Performance (Third Row)
        ax7 = fig.add_subplot(gs[2, :])
        x = df['problem_id']
        ax7.plot(x, df['moa_score'] * 100, 'o-', color=self.colors['moa'], 
                linewidth=2, markersize=6, label='MOA', alpha=self.style['alpha'])
        ax7.plot(x, df['sllm_score'] * 100, 's-', color=self.colors['single_llm'], 
                linewidth=2, markersize=6, label='SingleLLM', alpha=self.style['alpha'])
        ax7.set_title('Problem-by-Problem Performance Comparison', fontweight='bold', pad=20)
        ax7.set_xlabel('Problem Number')
        ax7.set_ylabel('Success Rate (%)')
        ax7.legend(fontsize=12)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, 100)
        
        # 8. Statistical Summary (Bottom)
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create statistical comparison table
        stats_text = f"""
        Statistical Summary:
        
        MOA Performance:          │ SingleLLM Performance:     │ Comparison:
        ├─ Avg Score: {df['moa_score'].mean():.1%}      │ ├─ Avg Score: {df['sllm_score'].mean():.1%}      │ ├─ Score Gap: {df['performance_gap'].mean():.2f}pp
        ├─ Std Dev:   {df['moa_score'].std():.3f}      │ ├─ Std Dev:   {df['sllm_score'].std():.3f}      │ ├─ Time Gap:  {df['time_diff'].mean():.2f}s
        ├─ Avg Time:  {df['moa_time'].mean():.1f}s       │ ├─ Avg Time:  {df['sllm_time'].mean():.1f}s       │ ├─ Wins:      {len(df[df['moa_score'] > df['sllm_score']])} vs {len(df[df['sllm_score'] > df['moa_score']])}
        └─ Total Time: {df['moa_time'].sum():.0f}s       │ └─ Total Time: {df['sllm_time'].sum():.0f}s       │ └─ Ties:      {len(df[df['moa_score'] == df['sllm_score']])}
        """
        
        ax8.text(0.05, 0.5, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['neutral'], alpha=0.1))
        
        # Save the dashboard
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), 
                   dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Performance dashboard saved: {output_dir}/performance_dashboard.png")

    def create_separate_performance_plots(self, df: pd.DataFrame, output_dir: str):
        """Create separate files for each performance plot"""
        if df.empty:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual plots
        self.create_success_rate_plot(df, os.path.join(output_dir, 'success_rate_comparison.png'))
        self.create_execution_time_plot(df, os.path.join(output_dir, 'execution_time_comparison.png'))
        self.create_win_distribution_plot(df, os.path.join(output_dir, 'win_distribution.png'))
        self.create_efficiency_plot(df, os.path.join(output_dir, 'efficiency_comparison.png'))
        self.create_performance_gap_distribution_plot(df, os.path.join(output_dir, 'performance_gap_distribution.png'))
        self.create_time_difference_distribution_plot(df, os.path.join(output_dir, 'time_difference_distribution.png'))
        self.create_problem_by_problem_plot(df, os.path.join(output_dir, 'problem_by_problem_performance.png'))
    
    def create_training_progress_plot(self, training_df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create overall training progress plot"""
        if training_df.empty:
            return None
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        iterations = training_df['iteration']
        scores = training_df['avg_score'] * 100
        
        ax.plot(iterations, scores, 'o-', color=self.colors['moa'], 
               linewidth=3, markersize=8, alpha=self.style['alpha'])
        
        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, scores, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), '--', color=self.colors['error'], 
                   linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.2f}% per iteration')
        
        ax.set_title('Training Score Progression', fontweight='bold', pad=20)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Average Score (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Training progress plot saved: {output_path}")
            return None
        else:
            return fig

    def create_improvement_rate_plot(self, training_df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create training improvement rate plot"""
        if training_df.empty or len(training_df) < 2:
            return None
        
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        iterations = training_df['iteration']
        scores = training_df['avg_score'] * 100
        
        improvements = [scores.iloc[i] - scores.iloc[i-1] for i in range(1, len(scores))]
        ax.plot(iterations[1:], improvements, 'o-', color=self.colors['success'], 
               linewidth=2, markersize=6)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.set_title('Improvement Rate', fontweight='bold', pad=20)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score Change (%)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Improvement rate plot saved: {output_path}")
            return None
        else:
            return fig

    def create_individual_neuron_plot(self, training_df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create individual neuron performance plot"""
        if training_df.empty:
            return None
        
        figsize = figsize or (14, 8)
        fig, ax = plt.subplots(figsize=figsize)
        iterations = training_df['iteration']
        neuron_cols = [col for col in training_df.columns if col.startswith('neuron_')]
        
        if neuron_cols:
            colors = sns.color_palette("husl", len(neuron_cols))
            for i, col in enumerate(neuron_cols):
                if not training_df[col].isna().all():
                    ax.plot(iterations, training_df[col] * 100, 'o-', 
                           color=colors[i], label=col.replace('neuron_', 'N'), 
                           linewidth=2, markersize=4, alpha=0.8)
            
            ax.set_title('Individual Neuron Performance', fontweight='bold', pad=20)
            ax.set_xlabel('Training Iteration')
            ax.set_ylabel('Score (%)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Individual neuron plot saved: {output_path}")
            return None
        else:
            return fig

    def create_weights_evolution_plot(self, checkpoints_df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create model weights evolution plot"""
        if checkpoints_df.empty:
            return None
        
        # Get weight columns for first layer neurons
        weight_cols = [col for col in checkpoints_df.columns if 'weight_0' in col and 'layer_0' in col]
        
        if not weight_cols:
            return None
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        checkpoints = checkpoints_df['checkpoint']
        
        colors = sns.color_palette("viridis", len(weight_cols))
        for i, col in enumerate(weight_cols[:6]):  # Limit to 6 neurons for readability
            neuron_name = col.replace('layer_0_neuron_', 'N').replace('_weight_0', '')
            ax.plot(checkpoints, checkpoints_df[col], 'o-', 
                   color=colors[i], label=neuron_name, linewidth=2, 
                   markersize=4, alpha=0.8)
        
        ax.set_title('Model Weights Evolution (Layer 0)', fontweight='bold', pad=20)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Weight Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Weights evolution plot saved: {output_path}")
            return None
        else:
            return fig

    def create_final_weights_plot(self, checkpoints_df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create final weights bar plot"""
        if checkpoints_df.empty:
            return None
        
        # Get weight columns for first layer neurons
        weight_cols = [col for col in checkpoints_df.columns if 'weight_0' in col and 'layer_0' in col]
        
        if not weight_cols:
            return None
        
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        final_weights = checkpoints_df[weight_cols].iloc[-1]
        ax.barh(range(len(final_weights)), final_weights.values, 
               color=self.colors['moa'], alpha=self.style['alpha'])
        ax.set_yticks(range(len(final_weights)))
        ax.set_yticklabels([col.replace('layer_0_neuron_', 'N').replace('_weight_0', '') 
                          for col in final_weights.index])
        ax.set_title('Final Weights', fontweight='bold', pad=20)
        ax.set_xlabel('Weight Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Final weights plot saved: {output_path}")
            return None
        else:
            return fig

    def create_training_analysis(self, training_df: pd.DataFrame, checkpoints_df: pd.DataFrame, output_dir: str):
        """Create training progression analysis"""
        if training_df.empty:
            print("⚠️  No training data available")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Training Progression Analysis', fontsize=24, fontweight='bold', y=0.96)
        
        # 1. Overall Training Progress (Top Row, Spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        iterations = training_df['iteration']
        scores = training_df['avg_score'] * 100
        
        ax1.plot(iterations, scores, 'o-', color=self.colors['moa'], 
                linewidth=3, markersize=8, alpha=self.style['alpha'])
        
        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, scores, 1)
            p = np.poly1d(z)
            ax1.plot(iterations, p(iterations), '--', color=self.colors['error'], 
                    linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.2f}% per iteration')
        
        ax1.set_title('Training Score Progression', fontweight='bold', pad=20)
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Average Score (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # 2. Training Improvement Rate (Top Right)
        ax2 = fig.add_subplot(gs[0, 2])
        if len(scores) > 1:
            improvements = [scores.iloc[i] - scores.iloc[i-1] for i in range(1, len(scores))]
            ax2.plot(iterations[1:], improvements, 'o-', color=self.colors['success'], 
                    linewidth=2, markersize=6)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax2.set_title('Improvement Rate', fontweight='bold', pad=20)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Score Change (%)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Individual Neuron Performance (Middle Row)
        ax3 = fig.add_subplot(gs[1, :])
        neuron_cols = [col for col in training_df.columns if col.startswith('neuron_')]
        
        if neuron_cols:
            colors = sns.color_palette("husl", len(neuron_cols))
            for i, col in enumerate(neuron_cols):
                if not training_df[col].isna().all():
                    ax3.plot(iterations, training_df[col] * 100, 'o-', 
                            color=colors[i], label=col.replace('neuron_', 'N'), 
                            linewidth=2, markersize=4, alpha=0.8)
            
            ax3.set_title('Individual Neuron Performance', fontweight='bold', pad=20)
            ax3.set_xlabel('Training Iteration')
            ax3.set_ylabel('Score (%)')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        # 4. Model Weights Evolution (Bottom Row)
        if not checkpoints_df.empty:
            # Get weight columns for first layer neurons
            weight_cols = [col for col in checkpoints_df.columns if 'weight_0' in col and 'layer_0' in col]
            
            if weight_cols:
                ax4 = fig.add_subplot(gs[2, :2])
                checkpoints = checkpoints_df['checkpoint']
                
                colors = sns.color_palette("viridis", len(weight_cols))
                for i, col in enumerate(weight_cols[:6]):  # Limit to 6 neurons for readability
                    neuron_name = col.replace('layer_0_neuron_', 'N').replace('_weight_0', '')
                    ax4.plot(checkpoints, checkpoints_df[col], 'o-', 
                            color=colors[i], label=neuron_name, linewidth=2, 
                            markersize=4, alpha=0.8)
                
                ax4.set_title('Model Weights Evolution (Layer 0)', fontweight='bold', pad=20)
                ax4.set_xlabel('Checkpoint')
                ax4.set_ylabel('Weight Value')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # Weight Statistics
            ax5 = fig.add_subplot(gs[2, 2])
            if weight_cols:
                final_weights = checkpoints_df[weight_cols].iloc[-1]
                ax5.barh(range(len(final_weights)), final_weights.values, 
                        color=self.colors['moa'], alpha=self.style['alpha'])
                ax5.set_yticks(range(len(final_weights)))
                ax5.set_yticklabels([col.replace('layer_0_neuron_', 'N').replace('_weight_0', '') 
                                   for col in final_weights.index])
                ax5.set_title('Final Weights', fontweight='bold', pad=20)
                ax5.set_xlabel('Weight Value')
                ax5.grid(True, alpha=0.3)
        
        # Save the analysis
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_analysis.png'), 
                   dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Training analysis saved: {output_dir}/training_analysis.png")

    def create_separate_training_plots(self, training_df: pd.DataFrame, checkpoints_df: pd.DataFrame, output_dir: str):
        """Create separate files for each training plot"""
        if training_df.empty:
            print("⚠️  No training data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual training plots
        self.create_training_progress_plot(training_df, os.path.join(output_dir, 'training_progress.png'))
        self.create_improvement_rate_plot(training_df, os.path.join(output_dir, 'improvement_rate.png'))
        self.create_individual_neuron_plot(training_df, os.path.join(output_dir, 'individual_neuron_performance.png'))
        
        # Create checkpoint-based plots if available
        if not checkpoints_df.empty:
            self.create_weights_evolution_plot(checkpoints_df, os.path.join(output_dir, 'weights_evolution.png'))
            self.create_final_weights_plot(checkpoints_df, os.path.join(output_dir, 'final_weights.png'))
    
    def create_score_distribution_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create score distribution comparison plot"""
        if df.empty:
            return None
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        moa_scores = df['moa_score'] * 100
        sllm_scores = df['sllm_score'] * 100
        
        ax.hist([moa_scores, sllm_scores], bins=15, alpha=0.7, 
               label=['MOA', 'SingleLLM'], color=[self.colors['moa'], self.colors['single_llm']])
        ax.set_title('Score Distribution Comparison', fontweight='bold', pad=20)
        ax.set_xlabel('Success Rate (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Score distribution plot saved: {output_path}")
            return None
        else:
            return fig

    def create_box_plot_comparison(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create box plot comparison"""
        if df.empty:
            return None
        
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        moa_scores = df['moa_score'] * 100
        sllm_scores = df['sllm_score'] * 100
        
        box_data = [moa_scores, sllm_scores]
        bp = ax.boxplot(box_data, labels=['MOA', 'SingleLLM'], patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['moa'])
        bp['boxes'][1].set_facecolor(self.colors['single_llm'])
        
        for box in bp['boxes']:
            box.set_alpha(self.style['alpha'])
        
        ax.set_title('Performance Distribution', fontweight='bold', pad=20)
        ax.set_ylabel('Success Rate (%)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Box plot comparison saved: {output_path}")
            return None
        else:
            return fig

    def create_correlation_analysis_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create time vs performance correlation plot"""
        if df.empty:
            return None
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(df['moa_time'], df['moa_score'] * 100, color=self.colors['moa'], 
                  alpha=0.7, s=60, label='MOA')
        ax.scatter(df['sllm_time'], df['sllm_score'] * 100, color=self.colors['single_llm'], 
                  alpha=0.7, s=60, label='SingleLLM')
        
        ax.set_title('Time vs Performance Correlation', fontweight='bold', pad=20)
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_ylabel('Success Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Correlation analysis plot saved: {output_path}")
            return None
        else:
            return fig

    def create_performance_gap_analysis_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create performance gap analysis plot"""
        if df.empty:
            return None
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        gap_positive = df[df['performance_gap'] > 0]['performance_gap']
        gap_negative = df[df['performance_gap'] < 0]['performance_gap']
        gap_zero = df[df['performance_gap'] == 0]['performance_gap']
        
        ax.hist([gap_positive, gap_negative], bins=15, alpha=0.7,
               label=[f'MOA Wins ({len(gap_positive)})', f'SingleLLM Wins ({len(gap_negative)})'],
               color=[self.colors['success'], self.colors['error']])
        
        if len(gap_zero) > 0:
            ax.axvline(x=0, color=self.colors['neutral'], linewidth=3, 
                      label=f'Ties ({len(gap_zero)})')
        
        ax.set_title('Performance Gap Analysis', fontweight='bold', pad=20)
        ax.set_xlabel('Performance Gap (pp)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Performance gap analysis plot saved: {output_path}")
            return None
        else:
            return fig

    def create_efficiency_across_problems_plot(self, df: pd.DataFrame, output_path: str = None, figsize: tuple = None):
        """Create efficiency comparison across problems plot"""
        if df.empty:
            return None
        
        figsize = figsize or (16, 8)
        fig, ax = plt.subplots(figsize=figsize)
        problems = df['problem_id']
        ax.plot(problems, df['moa_efficiency'], 'o-', color=self.colors['moa'], 
               linewidth=2, markersize=6, label='MOA Efficiency', alpha=self.style['alpha'])
        ax.plot(problems, df['sllm_efficiency'], 's-', color=self.colors['single_llm'], 
               linewidth=2, markersize=6, label='SingleLLM Efficiency', alpha=self.style['alpha'])
        
        ax.set_title('Efficiency Comparison Across Problems', fontweight='bold', pad=20)
        ax.set_xlabel('Problem Number')
        ax.set_ylabel('Efficiency (Score/Time)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✅ Efficiency across problems plot saved: {output_path}")
            return None
        else:
            return fig

    def create_statistical_deep_dive(self, df: pd.DataFrame, output_dir: str):
        """Create detailed statistical analysis"""
        if df.empty:
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Statistical Deep Dive Analysis', fontsize=24, fontweight='bold', y=0.96)
        
        # 1. Score Distribution Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        
        moa_scores = df['moa_score'] * 100
        sllm_scores = df['sllm_score'] * 100
        
        ax1.hist([moa_scores, sllm_scores], bins=15, alpha=0.7, 
                label=['MOA', 'SingleLLM'], color=[self.colors['moa'], self.colors['single_llm']])
        ax1.set_title('Score Distribution Comparison', fontweight='bold', pad=20)
        ax1.set_xlabel('Success Rate (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box Plot Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        
        box_data = [moa_scores, sllm_scores]
        bp = ax2.boxplot(box_data, labels=['MOA', 'SingleLLM'], patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['moa'])
        bp['boxes'][1].set_facecolor(self.colors['single_llm'])
        
        for box in bp['boxes']:
            box.set_alpha(self.style['alpha'])
        
        ax2.set_title('Performance Distribution', fontweight='bold', pad=20)
        ax2.set_ylabel('Success Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation Analysis
        ax3 = fig.add_subplot(gs[1, :2])
        
        ax3.scatter(df['moa_time'], df['moa_score'] * 100, color=self.colors['moa'], 
                   alpha=0.7, s=60, label='MOA')
        ax3.scatter(df['sllm_time'], df['sllm_score'] * 100, color=self.colors['single_llm'], 
                   alpha=0.7, s=60, label='SingleLLM')
        
        ax3.set_title('Time vs Performance Correlation', fontweight='bold', pad=20)
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Success Rate (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Gap Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        
        gap_positive = df[df['performance_gap'] > 0]['performance_gap']
        gap_negative = df[df['performance_gap'] < 0]['performance_gap']
        gap_zero = df[df['performance_gap'] == 0]['performance_gap']
        
        ax4.hist([gap_positive, gap_negative], bins=15, alpha=0.7,
                label=[f'MOA Wins ({len(gap_positive)})', f'SingleLLM Wins ({len(gap_negative)})'],
                color=[self.colors['success'], self.colors['error']])
        
        if len(gap_zero) > 0:
            ax4.axvline(x=0, color=self.colors['neutral'], linewidth=3, 
                       label=f'Ties ({len(gap_zero)})')
        
        ax4.set_title('Performance Gap Analysis', fontweight='bold', pad=20)
        ax4.set_xlabel('Performance Gap (pp)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Efficiency Analysis (Bottom Row)
        ax5 = fig.add_subplot(gs[2, :])
        
        problems = df['problem_id']
        ax5.plot(problems, df['moa_efficiency'], 'o-', color=self.colors['moa'], 
                linewidth=2, markersize=6, label='MOA Efficiency', alpha=self.style['alpha'])
        ax5.plot(problems, df['sllm_efficiency'], 's-', color=self.colors['single_llm'], 
                linewidth=2, markersize=6, label='SingleLLM Efficiency', alpha=self.style['alpha'])
        
        ax5.set_title('Efficiency Comparison Across Problems', fontweight='bold', pad=20)
        ax5.set_xlabel('Problem Number')
        ax5.set_ylabel('Efficiency (Score/Time)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Save the analysis
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'statistical_analysis.png'), 
                   dpi=self.style['dpi'], bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Statistical analysis saved: {output_dir}/statistical_analysis.png")

    def create_separate_statistical_plots(self, df: pd.DataFrame, output_dir: str):
        """Create separate files for each statistical plot"""
        if df.empty:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual statistical plots
        self.create_score_distribution_plot(df, os.path.join(output_dir, 'score_distribution_comparison.png'))
        self.create_box_plot_comparison(df, os.path.join(output_dir, 'performance_distribution_boxplot.png'))
        self.create_correlation_analysis_plot(df, os.path.join(output_dir, 'time_vs_performance_correlation.png'))
        self.create_performance_gap_analysis_plot(df, os.path.join(output_dir, 'performance_gap_analysis.png'))
        self.create_efficiency_across_problems_plot(df, os.path.join(output_dir, 'efficiency_across_problems.png'))

def main():
    parser = argparse.ArgumentParser(
        description="Modern MOA Performance Analyzer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with separate files for each plot (default)
  python analyze_results.py --comparison results.json --training history.json --checkpoints checkpoints/ --output plots/

  # Full analysis with dashboard-style combined plots
  python analyze_results.py --comparison results.json --training history.json --checkpoints checkpoints/ --output plots/ --dashboard

  # Comparison only with separate files
  python analyze_results.py --comparison results.json --output plots/

  # Training analysis only with dashboard style
  python analyze_results.py --training history.json --checkpoints checkpoints/ --output plots/ --dashboard

  # Summary only (no visualizations)
  python analyze_results.py --comparison results.json --summary-only
        """
    )
    
    parser.add_argument("--comparison", "-c", type=str, help="Path to comparison results JSON file")
    parser.add_argument("--training", "-t", type=str, help="Path to training history JSON file")
    parser.add_argument("--checkpoints", "-k", type=str, help="Path to model checkpoints directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for plots and analysis")
    parser.add_argument("--summary-only", "-s", action="store_true", help="Print summary only, skip visualization")
    parser.add_argument("--dashboard", "-d", action="store_true", help="Create dashboard-style plots (default: separate files for each plot)")
    
    args = parser.parse_args()
    
    print("🚀 MOA Performance Analyzer v2.0")
    print("=" * 50)
    
    analyzer = MOAAnalyzer()
    
    # Load data
    comparison_df = pd.DataFrame()
    training_df = pd.DataFrame() 
    checkpoints_df = pd.DataFrame()
    
    if args.comparison:
        comparison_df = analyzer.load_comparison_results(args.comparison)
        if not comparison_df.empty:
            print(f"✅ Loaded {len(comparison_df)} comparison results")
    
    if args.training:
        training_df = analyzer.load_training_history(args.training)
        if not training_df.empty:
            print(f"✅ Loaded {len(training_df)} training iterations")
    
    if args.checkpoints:
        checkpoints_df = analyzer.load_model_checkpoints(args.checkpoints)
        if not checkpoints_df.empty:
            print(f"✅ Loaded {len(checkpoints_df)} model checkpoints")
    
    # Generate summary
    if not comparison_df.empty:
        stats = analyzer.calculate_summary_stats(comparison_df)
        analyzer.print_executive_summary(stats, training_df)
    
    # Generate visualizations
    if not args.summary_only:
        if not comparison_df.empty:
            if args.dashboard:
                # Dashboard style - all plots in single files
                analyzer.create_performance_dashboard(comparison_df, args.output)
                analyzer.create_statistical_deep_dive(comparison_df, args.output)
                print(f"\n🎨 Dashboard-style visualizations saved to: {args.output}/")
            else:
                # Separate files for each plot
                analyzer.create_separate_performance_plots(comparison_df, args.output)
                analyzer.create_separate_statistical_plots(comparison_df, args.output)
                print(f"\n🎨 Individual comparison plots saved to: {args.output}/")
        
        if not training_df.empty:
            if args.dashboard:
                # Dashboard style
                analyzer.create_training_analysis(training_df, checkpoints_df, args.output)
                print(f"\n🎨 Dashboard-style training analysis saved to: {args.output}/")
            else:
                # Separate files
                analyzer.create_separate_training_plots(training_df, checkpoints_df, args.output)
                print(f"\n🎨 Individual training plots saved to: {args.output}/")
        
        plot_style = "dashboard-style" if args.dashboard else "individual"
        print(f"\n🎨 All {plot_style} visualizations completed!")
    
    print("✨ Analysis complete!")

if __name__ == "__main__":
    main()
