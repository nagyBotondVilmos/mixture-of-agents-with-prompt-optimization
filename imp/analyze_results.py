import argparse
import json
import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from tabulate import tabulate
import matplotlib.pyplot as plt

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
            "ties": 0
        },
        "single_llm": {
            "total_passed": 0,
            "scores": [],
            "times": [],
            "wins": 0,
            "ties": 0
        },
        "performance_gaps": [],
        "time_differences": []
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
        
        moa_passed = moa_data.get("passed", 0)
        sllm_passed = sllm_data.get("passed", 0)
        
        stats["moa"]["total_passed"] += moa_passed
        stats["single_llm"]["total_passed"] += sllm_passed
        
        # Record scores
        moa_score = moa_data.get("score", 0)
        sllm_score = sllm_data.get("score", 0)
        
        stats["moa"]["scores"].append(moa_score)
        stats["single_llm"]["scores"].append(sllm_score)
        
        # Record times
        moa_time = moa_data.get("time_seconds", 0)
        sllm_time = sllm_data.get("time_seconds", 0)
        
        stats["moa"]["times"].append(moa_time)
        stats["single_llm"]["times"].append(sllm_time)
        
        # Calculate performance gap (in percentage points)
        performance_gap = (moa_score - sllm_score) * 100
        stats["performance_gaps"].append(performance_gap)
        
        # Calculate time difference (positive means SingleLLM was faster)
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
    stats["moa"]["overall_score"] = stats["moa"]["total_passed"] / stats["total_tests"] if stats["total_tests"] > 0 else 0
    stats["single_llm"]["overall_score"] = stats["single_llm"]["total_passed"] / stats["total_tests"] if stats["total_tests"] > 0 else 0
    
    stats["moa"]["avg_score"] = np.mean(stats["moa"]["scores"]) if stats["moa"]["scores"] else 0
    stats["single_llm"]["avg_score"] = np.mean(stats["single_llm"]["scores"]) if stats["single_llm"]["scores"] else 0
    
    stats["moa"]["avg_time"] = np.mean(stats["moa"]["times"]) if stats["moa"]["times"] else 0
    stats["single_llm"]["avg_time"] = np.mean(stats["single_llm"]["times"]) if stats["single_llm"]["times"] else 0
    
    stats["moa"]["median_score"] = np.median(stats["moa"]["scores"]) if stats["moa"]["scores"] else 0
    stats["single_llm"]["median_score"] = np.median(stats["single_llm"]["scores"]) if stats["single_llm"]["scores"] else 0
    
    stats["moa"]["median_time"] = np.median(stats["moa"]["times"]) if stats["moa"]["times"] else 0
    stats["single_llm"]["median_time"] = np.median(stats["single_llm"]["times"]) if stats["single_llm"]["times"] else 0
    
    # Calculate average performance gap (in percentage points)
    stats["avg_performance_gap"] = np.mean(stats["performance_gaps"]) if stats["performance_gaps"] else 0
    stats["median_performance_gap"] = np.median(stats["performance_gaps"]) if stats["performance_gaps"] else 0
    
    # Calculate average time difference
    stats["avg_time_diff"] = np.mean(stats["time_differences"]) if stats["time_differences"] else 0
    stats["median_time_diff"] = np.median(stats["time_differences"]) if stats["time_differences"] else 0
    
    # Calculate efficiency ratio (average score per second)
    stats["moa"]["efficiency"] = stats["moa"]["avg_score"] / stats["moa"]["avg_time"] if stats["moa"]["avg_time"] > 0 else 0
    stats["single_llm"]["efficiency"] = stats["single_llm"]["avg_score"] / stats["single_llm"]["avg_time"] if stats["single_llm"]["avg_time"] > 0 else 0
    
    return stats

def print_summary(stats: Dict, detailed: bool = False):
    """Print a summary of the analysis results"""
    if not stats:
        print("No valid statistics to report")
        return
        
    print("\n=== MODEL COMPARISON SUMMARY ===\n")
    
    # Summary table
    summary_table = [
        ["Total Problems", stats["total_problems"]],
        ["Total Test Cases", stats["total_tests"]],
        ["", ""],
        ["MOA Tests Passed", f"{stats['moa']['total_passed']} / {stats['total_tests']} ({stats['moa']['avg_score']:.2%})"],
        ["SingleLLM Tests Passed", f"{stats['single_llm']['total_passed']} / {stats['total_tests']} ({stats['single_llm']['avg_score']:.2%})"],
        ["Performance Gap", f"{stats['avg_performance_gap']:.2f}pp (MOA {'ahead' if stats['avg_performance_gap'] >= 0 else 'behind'})"],
        ["", ""],
        ["MOA Wins", stats["moa"]["wins"]],
        ["SingleLLM Wins", stats["single_llm"]["wins"]],
        ["Ties", stats["moa"]["ties"]],
        ["", ""],
        ["MOA Avg Time", f"{stats['moa']['avg_time']:.2f}s"],
        ["SingleLLM Avg Time", f"{stats['single_llm']['avg_time']:.2f}s"],
        ["Time Difference", f"{stats['avg_time_diff']:.2f}s ({'MOA faster' if stats['avg_time_diff'] < 0 else 'SingleLLM faster'})"]
    ]
    
    print(tabulate(summary_table, tablefmt="simple"))
    
    if detailed:
        print("\n=== DETAILED STATISTICS ===\n")
        
        # Detailed table with median values
        detailed_table = [
            ["Metric", "MOA", "SingleLLM", "Difference"],
            ["Avg Score", f"{stats['moa']['avg_score']:.2%}", f"{stats['single_llm']['avg_score']:.2%}", f"{stats['avg_performance_gap']:.2f}pp"],
            ["Median Score", f"{stats['moa']['median_score']:.2%}", f"{stats['single_llm']['median_score']:.2%}", f"{stats['median_performance_gap']:.2f}pp"],
            ["Avg Time", f"{stats['moa']['avg_time']:.2f}s", f"{stats['single_llm']['avg_time']:.2f}s", f"{stats['avg_time_diff']:.2f}s"],
            ["Median Time", f"{stats['moa']['median_time']:.2f}s", f"{stats['single_llm']['median_time']:.2f}s", f"{stats['median_time_diff']:.2f}s"],
            ["Efficiency (score/second)", f"{stats['moa']['efficiency']:.4f}", f"{stats['single_llm']['efficiency']:.4f}", f"{stats['moa']['efficiency'] - stats['single_llm']['efficiency']:.4f}"]
        ]
        
        print(tabulate(detailed_table, headers="firstrow", tablefmt="grid"))

def generate_plots(stats: Dict, results: List[Dict], output_dir: str):
    """Generate plots visualizing the comparison results"""
    if not stats or not output_dir:
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Score Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    
    models = ["MOA", "SingleLLM"]
    scores = [stats["moa"]["avg_score"] * 100, stats["single_llm"]["avg_score"] * 100]
    plt.bar(models, scores, color=['blue', 'orange'])
    plt.title('Average Score Comparison')
    plt.ylabel('Average Score (%)')
    plt.ylim(0, 100)
    
    for i, v in enumerate(scores):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.savefig(os.path.join(output_dir, 'score_comparison.png'))
    plt.close()
    
    # 2. Time Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    
    times = [stats["moa"]["avg_time"], stats["single_llm"]["avg_time"]]
    plt.bar(models, times, color=['blue', 'orange'])
    plt.title('Average Execution Time Comparison')
    plt.ylabel('Average Time (seconds)')
    
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    plt.close()
    
    # 3. Wins Comparison Pie Chart
    plt.figure(figsize=(8, 8))
    
    labels = ['MOA Wins', 'SingleLLM Wins', 'Ties']
    sizes = [stats["moa"]["wins"], stats["single_llm"]["wins"], stats["moa"]["ties"]]
    colors = ['blue', 'orange', 'gray']
    explode = (0.1, 0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Win Distribution')
    
    plt.savefig(os.path.join(output_dir, 'win_distribution.png'))
    plt.close()
    
    # 4. Performance Gap Distribution
    if stats["performance_gaps"]:
        plt.figure(figsize=(12, 6))
        
        plt.hist(stats["performance_gaps"], bins=10, alpha=0.7, color='green')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Distribution of Performance Gaps (MOA - SingleLLM)')
        plt.xlabel('Performance Gap (percentage points)')
        plt.ylabel('Number of Problems')
        
        plt.savefig(os.path.join(output_dir, 'performance_gap_distribution.png'))
        plt.close()
        
    # 5. Problem-by-Problem Comparison
    problem_numbers = list(range(1, len(results) + 1))
    moa_scores = [result.get("moa", {}).get("score", 0) * 100 for result in results]
    sllm_scores = [result.get("single_llm", {}).get("score", 0) * 100 for result in results]
    
    plt.figure(figsize=(14, 7))
    plt.plot(problem_numbers, moa_scores, 'b-', marker='o', label='MOA')
    plt.plot(problem_numbers, sllm_scores, 'orange', marker='s', label='SingleLLM')
    plt.title('Problem-by-Problem Score Comparison')
    plt.xlabel('Problem Number')
    plt.ylabel('Score (%)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(problem_numbers)
    
    plt.savefig(os.path.join(output_dir, 'problem_comparison.png'))
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results", "-r", type=str, required=True, help="Path to comparison results JSON file")
    parser.add_argument("--detailed", "-d", action="store_true", help="Display detailed statistics")
    parser.add_argument("--plots", "-p", type=str, help="Directory to save visualization plots")
    
    args = parser.parse_args()
    
    # Load the results
    results = load_results(args.results)
    if not results:
        print("No valid results found. Exiting.")
        sys.exit(1)
        
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print summary
    print_summary(stats, args.detailed)
    
    # Generate plots if requested
    if args.plots:
        try:
            import matplotlib
            generate_plots(stats, results, args.plots)
        except ImportError:
            print("\nWarning: matplotlib not installed. Skipping plot generation.")
            print("To generate plots, install matplotlib with: pip install matplotlib")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError in analysis: {str(e)}")
        import traceback
        traceback.print_exc() 