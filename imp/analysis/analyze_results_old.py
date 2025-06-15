import argparse
import json
import os
import sys
import glob
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(filepath: str) -> List[Dict]:
    """Load comparison results from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "comparison_results" not in data:
            print(f"Invalid results file: {filepath} (missing 'comparison_results' key)")
            return []
            
        return data["comparison_results"]
    except Exception as e:
        print(f"Error loading results from {filepath}: {str(e)}")
        return []

def load_training_history(filepath: str) -> Dict:
    """Load training history from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading training history from {filepath}: {str(e)}")
        return {}

def load_model_checkpoints(checkpoint_dir: str) -> List[Dict]:
    """Load all model checkpoints from directory"""
    checkpoints = []
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_checkpoint_*.json"))
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for filepath in checkpoint_files:
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
                # Extract checkpoint number from filename
                checkpoint_num = int(os.path.basename(filepath).split('_')[-1].split('.')[0])
                checkpoint['checkpoint_num'] = checkpoint_num
                checkpoints.append(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint {filepath}: {str(e)}")
    
    return checkpoints

def load_final_model(filepath: str) -> Dict:
    """Load final model parameters"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading final model from {filepath}: {str(e)}")
        return {}

def extract_weights_progression(checkpoints: List[Dict]) -> Dict:
    """Extract weight progression from checkpoints"""
    progression = {
        'checkpoint_nums': [],
        'layer_weights': {}
    }
    
    for checkpoint in checkpoints:
        progression['checkpoint_nums'].append(checkpoint['checkpoint_num'])
        
        for layer_idx, layer in enumerate(checkpoint.get('layers', [])):
            layer_key = f'layer_{layer_idx}'
            if layer_key not in progression['layer_weights']:
                progression['layer_weights'][layer_key] = {}
            
            for neuron_idx, neuron in enumerate(layer.get('neurons', [])):
                neuron_key = f'neuron_{neuron_idx}'
                if neuron_key not in progression['layer_weights'][layer_key]:
                    progression['layer_weights'][layer_key][neuron_key] = []
                
                weights = neuron.get('w', [])
                # Store all weights for this neuron
                progression['layer_weights'][layer_key][neuron_key].append(weights)
    
    return progression

def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate detailed statistics from comparison results"""
    if not results:
        return {}
        
    stats = {
        "total_problems": len(results),
        "total_tests": 0,
        "moa": {
            "total_passed": 0,
            "scores": [],
            "times": [],
            "wins": 0,
            "ties": 0,
            "total_time": 0
        },
        "single_llm": {
            "total_passed": 0,
            "scores": [],
            "times": [],
            "wins": 0,
            "ties": 0,
            "total_time": 0
        },
        "performance_gaps": [],
        "time_differences": [],
        "score_distributions": {"moa": [], "single_llm": []},
        "time_distributions": {"moa": [], "single_llm": []}
    }
    
    for result in results:
        moa_data = result.get("moa", {})
        sllm_data = result.get("single_llm", {})
        
        # Skip invalid results
        if not moa_data or not sllm_data:
            continue
            
        # Extract test counts and passed tests
        total_tests = moa_data.get("total", 0)
        stats["total_tests"] += total_tests
        
        moa_passed = moa_data.get("passed", {})
        sllm_passed = sllm_data.get("passed", {})
        
        # Handle both dict and int formats for passed tests
        if isinstance(moa_passed, dict):
            moa_passed_count = sum(moa_passed.values())
        else:
            moa_passed_count = moa_passed
            
        if isinstance(sllm_passed, dict):
            sllm_passed_count = sum(sllm_passed.values())
        else:
            sllm_passed_count = sllm_passed
        
        stats["moa"]["total_passed"] += moa_passed_count
        stats["single_llm"]["total_passed"] += sllm_passed_count
        
        # Record scores
        moa_scores = moa_data.get("scores", {})
        sllm_scores = sllm_data.get("scores", {})
        
        if isinstance(moa_scores, dict):
            moa_score = np.mean(list(moa_scores.values())) if moa_scores else 0
        else:
            moa_score = moa_scores
            
        if isinstance(sllm_scores, dict):
            sllm_score = np.mean(list(sllm_scores.values())) if sllm_scores else 0
        else:
            sllm_score = sllm_scores
        
        stats["moa"]["scores"].append(moa_score)
        stats["single_llm"]["scores"].append(sllm_score)
        stats["score_distributions"]["moa"].append(moa_score)
        stats["score_distributions"]["single_llm"].append(sllm_score)
        
        # Record times
        moa_time = moa_data.get("time_seconds", 0)
        sllm_time = sllm_data.get("time_seconds", 0)
        
        stats["moa"]["times"].append(moa_time)
        stats["single_llm"]["times"].append(sllm_time)
        stats["moa"]["total_time"] += moa_time
        stats["single_llm"]["total_time"] += sllm_time
        stats["time_distributions"]["moa"].append(moa_time)
        stats["time_distributions"]["single_llm"].append(sllm_time)
        
        # Calculate performance gap (in percentage points)
        performance_gap = (moa_score - sllm_score) * 100
        stats["performance_gaps"].append(performance_gap)
        
        # Calculate time difference (positive means MOA was slower)
        time_diff = (moa_time - sllm_time)
        stats["time_differences"].append(time_diff)
        
        # Record wins
        if moa_score > sllm_score:
            stats["moa"]["wins"] += 1
        elif sllm_score > moa_score:
            stats["single_llm"]["wins"] += 1
        else:
            stats["moa"]["ties"] += 1
            stats["single_llm"]["ties"] += 1
    
    # Calculate overall statistics
    if stats["total_tests"] > 0:
        stats["moa"]["overall_score"] = stats["moa"]["total_passed"] / stats["total_tests"]
        stats["single_llm"]["overall_score"] = stats["single_llm"]["total_passed"] / stats["total_tests"]
    else:
        stats["moa"]["overall_score"] = 0
        stats["single_llm"]["overall_score"] = 0
    
    # Calculate averages and medians
    for model in ["moa", "single_llm"]:
        if stats[model]["scores"]:
            stats[model]["avg_score"] = np.mean(stats[model]["scores"])
            stats[model]["median_score"] = np.median(stats[model]["scores"])
            stats[model]["std_score"] = np.std(stats[model]["scores"])
        else:
            stats[model]["avg_score"] = 0
            stats[model]["median_score"] = 0
            stats[model]["std_score"] = 0
            
        if stats[model]["times"]:
            stats[model]["avg_time"] = np.mean(stats[model]["times"])
            stats[model]["median_time"] = np.median(stats[model]["times"])
            stats[model]["std_time"] = np.std(stats[model]["times"])
        else:
            stats[model]["avg_time"] = 0
            stats[model]["median_time"] = 0
            stats[model]["std_time"] = 0
    
    # Calculate performance gaps
    if stats["performance_gaps"]:
        stats["avg_performance_gap"] = np.mean(stats["performance_gaps"])
        stats["median_performance_gap"] = np.median(stats["performance_gaps"])
        stats["std_performance_gap"] = np.std(stats["performance_gaps"])
    else:
        stats["avg_performance_gap"] = 0
        stats["median_performance_gap"] = 0
        stats["std_performance_gap"] = 0
    
    # Calculate time differences
    if stats["time_differences"]:
        stats["avg_time_diff"] = np.mean(stats["time_differences"])
        stats["median_time_diff"] = np.median(stats["time_differences"])
        stats["std_time_diff"] = np.std(stats["time_differences"])
    else:
        stats["avg_time_diff"] = 0
        stats["median_time_diff"] = 0
        stats["std_time_diff"] = 0
    
    # Calculate efficiency ratios
    for model in ["moa", "single_llm"]:
        if stats[model]["avg_time"] > 0:
            stats[model]["efficiency"] = stats[model]["avg_score"] / stats[model]["avg_time"]
        else:
            stats[model]["efficiency"] = 0
    
    return stats

def analyze_training_history(training_history: Dict) -> Dict:
    """Analyze training history for score progression"""
    if not training_history or "history" not in training_history:
        return {}
    
    history = training_history["history"]
    analysis = {
        "iterations": [],
        "avg_scores": [],
        "score_progression": {},
        "problem_difficulties": [],
        "timestamps": []
    }
    
    for i, entry in enumerate(history):
        analysis["iterations"].append(i + 1)
        analysis["timestamps"].append(entry.get("timestamp", ""))
        
        scores = entry.get("scores", {})
        if scores:
            avg_score = np.mean(list(scores.values()))
            analysis["avg_scores"].append(avg_score)
            
            # Track individual neuron scores
            for neuron_key, score in scores.items():
                if neuron_key not in analysis["score_progression"]:
                    analysis["score_progression"][neuron_key] = []
                analysis["score_progression"][neuron_key].append(score)
        else:
            analysis["avg_scores"].append(0)
    
    return analysis

def print_summary(stats: Dict, training_analysis: Dict = None, detailed: bool = False):
    """Print a comprehensive summary of the analysis results"""
    if not stats:
        print("No valid statistics to report")
        return
        
    print("\n" + "="*60)
    print("           MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Main comparison table
    summary_table = [
        ["Metric", "MOA", "SingleLLM", "Difference"],
        ["Problems Analyzed", stats["total_problems"], stats["total_problems"], "-"],
        ["Total Test Cases", stats["total_tests"], stats["total_tests"], "-"],
        ["", "", "", ""],
        ["Tests Passed", f"{stats['moa']['total_passed']}", f"{stats['single_llm']['total_passed']}", f"{stats['moa']['total_passed'] - stats['single_llm']['total_passed']}"],
        ["Success Rate", f"{stats['moa']['avg_score']:.2%}", f"{stats['single_llm']['avg_score']:.2%}", f"{stats['avg_performance_gap']:.2f}pp"],
        ["Median Score", f"{stats['moa']['median_score']:.2%}", f"{stats['single_llm']['median_score']:.2%}", f"{stats['median_performance_gap']:.2f}pp"],
        ["", "", "", ""],
        ["Avg Time", f"{stats['moa']['avg_time']:.2f}s", f"{stats['single_llm']['avg_time']:.2f}s", f"{stats['avg_time_diff']:.2f}s"],
        ["Total Time", f"{stats['moa']['total_time']:.1f}s", f"{stats['single_llm']['total_time']:.1f}s", f"{stats['moa']['total_time'] - stats['single_llm']['total_time']:.1f}s"],
        ["Efficiency", f"{stats['moa']['efficiency']:.4f}", f"{stats['single_llm']['efficiency']:.4f}", f"{stats['moa']['efficiency'] - stats['single_llm']['efficiency']:.4f}"],
        ["", "", "", ""],
        ["Wins", stats["moa"]["wins"], stats["single_llm"]["wins"], f"{stats['moa']['wins'] - stats['single_llm']['wins']}"],
        ["Ties", stats["moa"]["ties"], "-", "-"]
    ]
    
    print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   • MOA wins: {stats['moa']['wins']}/{stats['total_problems']} ({stats['moa']['wins']/stats['total_problems']*100:.1f}%)")
    print(f"   • SingleLLM wins: {stats['single_llm']['wins']}/{stats['total_problems']} ({stats['single_llm']['wins']/stats['total_problems']*100:.1f}%)")
    print(f"   • Ties: {stats['moa']['ties']}/{stats['total_problems']} ({stats['moa']['ties']/stats['total_problems']*100:.1f}%)")
    
    performance_verdict = "MOA performs better" if stats['avg_performance_gap'] > 0 else "SingleLLM performs better"
    time_verdict = "MOA is faster" if stats['avg_time_diff'] < 0 else "SingleLLM is faster"
    
    print(f"   • Overall: {performance_verdict} by {abs(stats['avg_performance_gap']):.2f}pp")
    print(f"   • Speed: {time_verdict} by {abs(stats['avg_time_diff']):.2f}s on average")
    
    if training_analysis and training_analysis.get("avg_scores"):
        print(f"\n📈 TRAINING SUMMARY:")
        initial_score = training_analysis["avg_scores"][0] if training_analysis["avg_scores"] else 0
        final_score = training_analysis["avg_scores"][-1] if training_analysis["avg_scores"] else 0
        improvement = final_score - initial_score
        print(f"   • Training iterations: {len(training_analysis['avg_scores'])}")
        print(f"   • Initial avg score: {initial_score:.2%}")
        print(f"   • Final avg score: {final_score:.2%}")
        print(f"   • Improvement: {improvement:.2%} ({improvement*100:.2f}pp)")

def generate_comprehensive_plots(stats: Dict, results: List[Dict], training_analysis: Dict, 
                               weights_progression: Dict, output_dir: str):
    """Generate comprehensive visualization plots"""
    if not output_dir:
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    print(f"\n🎨 Generating comprehensive visualizations...")
    
    # 1. Training Score Progression
    if training_analysis and training_analysis.get("avg_scores"):
        plt.figure(figsize=(14, 8))
        
        iterations = training_analysis["iterations"]
        avg_scores = [s * 100 for s in training_analysis["avg_scores"]]
        
        plt.subplot(2, 2, 1)
        plt.plot(iterations, avg_scores, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Training Score Progression', fontsize=14, fontweight='bold')
        plt.xlabel('Training Iteration')
        plt.ylabel('Average Score (%)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, avg_scores, 1)
            p = np.poly1d(z)
            plt.plot(iterations, p(iterations), "r--", alpha=0.8, linewidth=1)
        
        # Individual neuron progression
        plt.subplot(2, 2, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(training_analysis.get("score_progression", {}))))
        for i, (neuron_key, scores) in enumerate(training_analysis.get("score_progression", {}).items()):
            if scores:
                plt.plot(iterations[:len(scores)], [s * 100 for s in scores], 
                        color=colors[i], label=neuron_key, linewidth=1.5, alpha=0.8)
        
        plt.title('Individual Neuron Score Progression', fontsize=14, fontweight='bold')
        plt.xlabel('Training Iteration')
        plt.ylabel('Score (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Score distribution over training
        plt.subplot(2, 2, 3)
        if len(avg_scores) > 10:
            # Create bins for different training phases
            early_scores = avg_scores[:len(avg_scores)//3]
            mid_scores = avg_scores[len(avg_scores)//3:2*len(avg_scores)//3]
            late_scores = avg_scores[2*len(avg_scores)//3:]
            
            plt.hist([early_scores, mid_scores, late_scores], 
                    bins=10, alpha=0.7, label=['Early', 'Mid', 'Late'], 
                    color=['red', 'orange', 'green'])
            plt.title('Score Distribution by Training Phase', fontsize=14, fontweight='bold')
            plt.xlabel('Score (%)')
            plt.ylabel('Frequency')
            plt.legend()
        else:
            plt.hist(avg_scores, bins=min(10, len(avg_scores)), alpha=0.7, color='blue')
            plt.title('Training Score Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Score (%)')
            plt.ylabel('Frequency')
        
        # Training improvement rate
        plt.subplot(2, 2, 4)
        if len(avg_scores) > 1:
            improvements = [avg_scores[i] - avg_scores[i-1] for i in range(1, len(avg_scores))]
            plt.plot(iterations[1:], improvements, 'g-', linewidth=2, marker='s', markersize=4)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.title('Training Improvement Rate', fontsize=14, fontweight='bold')
            plt.xlabel('Training Iteration')
            plt.ylabel('Score Improvement (pp)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_progression.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Model Weights Progression
    if weights_progression and weights_progression.get("checkpoint_nums"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Weights Progression Throughout Training', fontsize=16, fontweight='bold')
        
        checkpoint_nums = weights_progression["checkpoint_nums"]
        
        # Plot weights for each layer
        for layer_idx, (layer_key, layer_data) in enumerate(weights_progression["layer_weights"].items()):
            if layer_idx >= 4:  # Limit to 4 subplots
                break
                
            ax = axes[layer_idx // 2, layer_idx % 2]
            
            for neuron_key, weight_history in layer_data.items():
                # Extract first weight of each neuron over time
                first_weights = []
                for weights in weight_history:
                    if weights:
                        first_weights.append(weights[0])
                    else:
                        first_weights.append(0)
                
                if first_weights:
                    ax.plot(checkpoint_nums[:len(first_weights)], first_weights, 
                           marker='o', linewidth=2, label=neuron_key, alpha=0.8)
            
            ax.set_title(f'{layer_key.replace("_", " ").title()} Weights', fontweight='bold')
            ax.set_xlabel('Checkpoint Number')
            ax.set_ylabel('Weight Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(weights_progression["layer_weights"]), 4):
            fig.delaxes(axes[i // 2, i % 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weights_progression.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Comprehensive Model Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Score comparison
    ax = axes[0, 0]
    models = ["MOA", "SingleLLM"]
    scores = [stats["moa"]["avg_score"] * 100, stats["single_llm"]["avg_score"] * 100]
    bars = ax.bar(models, scores, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax.set_title('Average Success Rate', fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Time comparison
    ax = axes[0, 1]
    times = [stats["moa"]["avg_time"], stats["single_llm"]["avg_time"]]
    bars = ax.bar(models, times, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax.set_title('Average Execution Time', fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Win distribution
    ax = axes[0, 2]
    labels = ['MOA Wins', 'SingleLLM Wins', 'Ties']
    sizes = [stats["moa"]["wins"], stats["single_llm"]["wins"], stats["moa"]["ties"]]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                     startangle=90, textprops={'fontweight': 'bold'})
    ax.set_title('Win Distribution', fontweight='bold')
    
    # Performance gap distribution
    ax = axes[1, 0]
    if stats["performance_gaps"]:
        ax.hist(stats["performance_gaps"], bins=15, alpha=0.7, color='#C73E1D', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.axvline(x=stats["avg_performance_gap"], color='red', linestyle='-', linewidth=2, 
                  label=f'Mean: {stats["avg_performance_gap"]:.2f}pp')
        ax.set_title('Performance Gap Distribution', fontweight='bold')
        ax.set_xlabel('Performance Gap (MOA - SingleLLM) [pp]')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Time difference distribution
    ax = axes[1, 1]
    if stats["time_differences"]:
        ax.hist(stats["time_differences"], bins=15, alpha=0.7, color='#3F7CAC', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.axvline(x=stats["avg_time_diff"], color='blue', linestyle='-', linewidth=2,
                  label=f'Mean: {stats["avg_time_diff"]:.2f}s')
        ax.set_title('Time Difference Distribution', fontweight='bold')
        ax.set_xlabel('Time Difference (MOA - SingleLLM) [s]')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Efficiency comparison
    ax = axes[1, 2]
    efficiencies = [stats["moa"]["efficiency"], stats["single_llm"]["efficiency"]]
    bars = ax.bar(models, efficiencies, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax.set_title('Efficiency (Score/Time)', fontweight='bold')
    ax.set_ylabel('Efficiency')
    
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Problem-by-Problem Analysis
    if results:
        plt.figure(figsize=(16, 10))
        
        problem_numbers = list(range(1, len(results) + 1))
        moa_scores = []
        sllm_scores = []
        moa_times = []
        sllm_times = []
        
        for result in results:
            moa_data = result.get("moa", {})
            sllm_data = result.get("single_llm", {})
            
            # Extract scores
            moa_score_data = moa_data.get("scores", {})
            sllm_score_data = sllm_data.get("scores", {})
            
            if isinstance(moa_score_data, dict):
                moa_score = np.mean(list(moa_score_data.values())) * 100 if moa_score_data else 0
            else:
                moa_score = moa_score_data * 100
                
            if isinstance(sllm_score_data, dict):
                sllm_score = np.mean(list(sllm_score_data.values())) * 100 if sllm_score_data else 0
            else:
                sllm_score = sllm_score_data * 100
            
            moa_scores.append(moa_score)
            sllm_scores.append(sllm_score)
            moa_times.append(moa_data.get("time_seconds", 0))
            sllm_times.append(sllm_data.get("time_seconds", 0))
        
        # Score comparison plot
        plt.subplot(2, 1, 1)
        plt.plot(problem_numbers, moa_scores, 'b-', marker='o', linewidth=2, 
                markersize=4, label='MOA', alpha=0.8)
        plt.plot(problem_numbers, sllm_scores, 'r-', marker='s', linewidth=2, 
                markersize=4, label='SingleLLM', alpha=0.8)
        plt.title('Problem-by-Problem Score Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Problem Number')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Time comparison plot
        plt.subplot(2, 1, 2)
        plt.plot(problem_numbers, moa_times, 'b-', marker='o', linewidth=2, 
                markersize=4, label='MOA', alpha=0.8)
        plt.plot(problem_numbers, sllm_times, 'r-', marker='s', linewidth=2, 
                markersize=4, label='SingleLLM', alpha=0.8)
        plt.title('Problem-by-Problem Time Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Problem Number')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'problem_by_problem.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Statistical Distribution Comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Score distributions
    ax = axes[0, 0]
    if stats["score_distributions"]["moa"] and stats["score_distributions"]["single_llm"]:
        moa_scores_pct = [s * 100 for s in stats["score_distributions"]["moa"]]
        sllm_scores_pct = [s * 100 for s in stats["score_distributions"]["single_llm"]]
        
        ax.hist([moa_scores_pct, sllm_scores_pct], bins=15, alpha=0.7, 
               label=['MOA', 'SingleLLM'], color=['#2E86AB', '#A23B72'])
        ax.set_title('Score Distribution Comparison', fontweight='bold')
        ax.set_xlabel('Score (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Time distributions
    ax = axes[0, 1]
    if stats["time_distributions"]["moa"] and stats["time_distributions"]["single_llm"]:
        ax.hist([stats["time_distributions"]["moa"], stats["time_distributions"]["single_llm"]], 
               bins=15, alpha=0.7, label=['MOA', 'SingleLLM'], color=['#2E86AB', '#A23B72'])
        ax.set_title('Time Distribution Comparison', fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Box plots for scores
    ax = axes[1, 0]
    if stats["score_distributions"]["moa"] and stats["score_distributions"]["single_llm"]:
        moa_scores_pct = [s * 100 for s in stats["score_distributions"]["moa"]]
        sllm_scores_pct = [s * 100 for s in stats["score_distributions"]["single_llm"]]
        
        box_data = [moa_scores_pct, sllm_scores_pct]
        bp = ax.boxplot(box_data, labels=['MOA', 'SingleLLM'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2E86AB')
        bp['boxes'][1].set_facecolor('#A23B72')
        ax.set_title('Score Distribution Box Plot', fontweight='bold')
        ax.set_ylabel('Score (%)')
        ax.grid(True, alpha=0.3)
    
    # Box plots for times
    ax = axes[1, 1]
    if stats["time_distributions"]["moa"] and stats["time_distributions"]["single_llm"]:
        box_data = [stats["time_distributions"]["moa"], stats["time_distributions"]["single_llm"]]
        bp = ax.boxplot(box_data, labels=['MOA', 'SingleLLM'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2E86AB')
        bp['boxes'][1].set_facecolor('#A23B72')
        ax.set_title('Time Distribution Box Plot', fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All visualizations saved to: {output_dir}/")
    print(f"   📊 Generated plots:")
    print(f"      • training_progression.png - Training score and improvement analysis")
    print(f"      • weights_progression.png - Model parameter evolution")
    print(f"      • comprehensive_comparison.png - Overall performance comparison")
    print(f"      • problem_by_problem.png - Individual problem analysis")
    print(f"      • statistical_distributions.png - Statistical analysis")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis of MOA training and comparison results")
    parser.add_argument("--results", "-r", type=str, help="Path to comparison results JSON file")
    parser.add_argument("--training-history", "-t", type=str, help="Path to training history JSON file")
    parser.add_argument("--checkpoint-dir", "-c", type=str, help="Path to checkpoint directory")
    parser.add_argument("--final-model", "-f", type=str, help="Path to final model JSON file")
    parser.add_argument("--detailed", "-d", action="store_true", help="Display detailed statistics")
    parser.add_argument("--plots", "-p", type=str, help="Directory to save visualization plots")
    
    args = parser.parse_args()
    
    # Load all available data
    results = []
    training_analysis = {}
    weights_progression = {}
    
    if args.results:
        results = load_results(args.results)
        if not results:
            print("⚠️  No comparison results found")
    
    if args.training_history:
        training_history = load_training_history(args.training_history)
        if training_history:
            training_analysis = analyze_training_history(training_history)
            print(f"✅ Loaded training history with {len(training_analysis.get('iterations', []))} iterations")
        else:
            print("⚠️  No training history found")
    
    if args.checkpoint_dir:
        checkpoints = load_model_checkpoints(args.checkpoint_dir)
        if checkpoints:
            weights_progression = extract_weights_progression(checkpoints)
            print(f"✅ Loaded {len(checkpoints)} model checkpoints")
        else:
            print("⚠️  No model checkpoints found")
    
    if args.final_model:
        final_model = load_final_model(args.final_model)
        if final_model:
            print("✅ Loaded final model parameters")
        else:
            print("⚠️  No final model found")
    
    # Analyze comparison results if available
    if results:
        stats = calculate_statistics(results)
        print(f"✅ Analyzed {len(results)} comparison results")
        
        # Print comprehensive summary
        print_summary(stats, training_analysis, args.detailed)
        
        # Generate plots if requested
        if args.plots:
            try:
                generate_comprehensive_plots(stats, results, training_analysis, 
                                           weights_progression, args.plots)
            except ImportError as e:
                print(f"\n⚠️  Warning: Missing plotting dependencies: {str(e)}")
                print("   To generate plots, install required packages:")
                print("   pip install matplotlib seaborn")
            except Exception as e:
                print(f"\n❌ Error generating plots: {str(e)}")
                import traceback
                traceback.print_exc()
    else:
        print("❌ No comparison results to analyze. Please provide --results argument.")
    
    # If only training data is available, show training summary
    if not results and training_analysis:
        print("\n📈 TRAINING-ONLY ANALYSIS:")
        if training_analysis.get("avg_scores"):
            initial_score = training_analysis["avg_scores"][0]
            final_score = training_analysis["avg_scores"][-1]
            improvement = final_score - initial_score
            print(f"   • Training iterations: {len(training_analysis['avg_scores'])}")
            print(f"   • Initial avg score: {initial_score:.2%}")
            print(f"   • Final avg score: {final_score:.2%}")
            print(f"   • Total improvement: {improvement:.2%} ({improvement*100:.2f}pp)")
            
            if args.plots:
                # Generate training-only plots
                try:
                    os.makedirs(args.plots, exist_ok=True)
                    
                    plt.figure(figsize=(12, 8))
                    iterations = training_analysis["iterations"]
                    avg_scores = [s * 100 for s in training_analysis["avg_scores"]]
                    
                    plt.plot(iterations, avg_scores, 'b-', linewidth=2, marker='o', markersize=6)
                    plt.title('Training Score Progression', fontsize=16, fontweight='bold')
                    plt.xlabel('Training Iteration')
                    plt.ylabel('Average Score (%)')
                    plt.grid(True, alpha=0.3)
                    plt.ylim(0, 100)
                    
                    # Add trend line
                    if len(iterations) > 1:
                        z = np.polyfit(iterations, avg_scores, 1)
                        p = np.poly1d(z)
                        plt.plot(iterations, p(iterations), "r--", alpha=0.8, linewidth=2)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.plots, 'training_only_progression.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✅ Training progression plot saved to: {args.plots}/training_only_progression.png")
                except Exception as e:
                    print(f"❌ Error generating training plots: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
