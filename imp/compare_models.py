import asyncio
import argparse
import json
import os
import time
import traceback
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import glob
import sys

from mymoa.moa import MOA
from singlellm.single_llm import SingleLLM
from common.secret import LLM_API_KEY_MOA, LLM_API_URL, LLM_MODEL, SIMILARITY_THRESHOLD
from common.shared_utils import run_test, generate_test_cases, save_to_json_file

class ModelComparer:
    """Compares output of MOA and SingleLLM models on coding problems"""
    
    def __init__(self, moa_param_file: str):
        self.moa_param_file = moa_param_file
        self.test_cases = []
        self.problems = []
        self.moa_model = None
        self.single_model = None
        self.comparison_results = []
        
    async def initialize_models(self):
        """Initialize both models - MOA from parameters file and SingleLLM directly"""
        print(f"Loading MOA model from {self.moa_param_file}")
        
        # Load MOA parameters
        # moa_params = load_model_params(self.moa_param_file)
        with open(self.moa_param_file, 'r') as f:
            moa_params = json.load(f)
        
        # Initialize MLP (MOA model)
        # Extract layer sizes from parameters
        layer_sizes = [len(layer["neurons"]) for layer in moa_params.get("layers", [])]
        if not layer_sizes:
            print("Error: Invalid parameter file - no layers found")
            return False
            
        try:
            # Create MOA model
            self.moa_model = await MOA.create(nouts=layer_sizes, model_goal=moa_params.get("model_goal", "coding"))
            self.moa_model.load_parameters(moa_params)
            print(f"MOA model initialized with {len(layer_sizes)} layers")
            
            # Create SingleLLM model (uses the last key in the MOA API keys for testing)
            self.single_model = SingleLLM(
                api_key=LLM_API_KEY_MOA[-1], 
                base_url=LLM_API_URL,
                model=LLM_MODEL
            )
            print("SingleLLM model initialized")
            
            return True
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_test_cases_from_file(self, filepath: str) -> bool:
        """Load test cases from a pre-generated JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if "test_cases" not in data:
                print(f"Invalid test cases file: {filepath} (missing 'test_cases' key)")
                return False
                
            self.test_cases = data["test_cases"]
            
            # If problem is included in the file, extract it
            problem_text = data.get("problem", "")
            if problem_text:
                self.problems.append({"description": problem_text})
                
            print(f"Loaded {len(self.test_cases)} test cases from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading test cases from {filepath}: {str(e)}")
            traceback.print_exc()
            return False
            
    def load_test_cases_from_directory(self, dir_path: str) -> List[Dict]:
        """Load all test case files from a directory and return problem descriptions"""
        if not os.path.isdir(dir_path):
            print(f"Test cases directory not found: {dir_path}")
            return []
            
        problems = []
        test_case_files = glob.glob(os.path.join(dir_path, "*test_cases*.json"))
        
        if not test_case_files:
            print(f"No test case files found in {dir_path}")
            return []
            
        print(f"Found {len(test_case_files)} test case files")
        
        for file_path in test_case_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if "problem" in data and "test_cases" in data:
                    problems.append({
                        "description": data["problem"],
                        "test_cases_file": file_path
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                
        print(f"Loaded {len(problems)} problems with test cases")
        return problems
    
    def save_test_cases(self, problem_text: str, test_cases_dir: str) -> str:
        """Save test cases for a problem to a file in the specified directory"""
        if not os.path.exists(test_cases_dir):
            os.makedirs(test_cases_dir, exist_ok=True)
            
        # Create a sanitized filename from the first few words of the problem
        words = problem_text.split()[:5]
        sanitized_name = "_".join(w.strip().lower() for w in words if w.strip().isalnum())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sanitized_name}_{timestamp}_test_cases.json"
        filepath = os.path.join(test_cases_dir, filename)
        
        # Save test cases and problem to file
        data = {
            "problem": problem_text,
            "test_cases": self.test_cases,
            "generated_at": datetime.now().isoformat()
        }
        
        if save_to_json_file(data, filepath):
            print(f"Saved {len(self.test_cases)} test cases to {filepath}")
            return filepath
        
        return ""
            
    async def generate_test_cases_for_problem(self, prompt: str, num_tests: int = 10, test_cases_dir: Optional[str] = None) -> bool:
        """Generate test cases for a problem prompt"""
        success, test_cases = await generate_test_cases(
            prompt, 
            self.single_model.client, 
            LLM_MODEL, 
            num_tests
        )
        
        if success and test_cases:
            self.test_cases = test_cases
            
            # Save test cases if directory is provided
            if test_cases_dir:
                self.save_test_cases(prompt, test_cases_dir)
                
            return True
        return False
    
    @staticmethod
    async def generate_test_cases_for_problem_static(prompt: str, client, model: str, num_tests: int = 10) -> Tuple[bool, List, str]:
        """Static version of test case generation for concurrent execution"""
        success, test_cases = await generate_test_cases(
            prompt, 
            client, 
            model, 
            num_tests
        )
        return success, test_cases, prompt
    
    async def evaluate_models(self, prompt: str) -> Dict:
        """Evaluate both models on the same prompt and test cases"""
        if not self.moa_model or not self.single_model:
            print("Models not initialized")
            return {}
            
        if not self.test_cases:
            print("No test cases available")
            return {}
        
        print("Evaluating MOA model...")
        moa_start_time = time.time()
        moa_output = await self.moa_model(prompt)
        moa_end_time = time.time()
        moa_time = moa_end_time - moa_start_time
        
        print("Evaluating SingleLLM model...")
        sllm_start_time = time.time()
        sllm_output = await self.single_model(prompt)
        sllm_end_time = time.time()
        sllm_time = sllm_end_time - sllm_start_time
        
        # Parse outputs to extract code
        try:
            moa_parsed = json.loads(moa_output)
            moa_code = moa_parsed.get("code", "")
        except:
            # If parsing fails, try to extract code using the parse_code_output function
            from mymoa.moa import parse_code_output
            parsed = parse_code_output(moa_output)
            moa_code = parsed.get("code", "") if parsed else ""
            
        try:
            sllm_parsed = json.loads(sllm_output)
            sllm_code = sllm_parsed.get("code", "")
        except:
            # If parsing fails, try to extract code using the parse_code_output function
            from mymoa.moa import parse_code_output
            parsed = parse_code_output(sllm_output)
            sllm_code = parsed.get("code", "") if parsed else ""
        
        # Run test cases on both outputs using dual metrics
        moa_results = {}
        sllm_results = {}
        
        moa_passed_similarity = 0
        moa_passed_exact = 0
        sllm_passed_similarity = 0
        sllm_passed_exact = 0
        total_tests = len(self.test_cases)
        
        for test_idx, test_case in enumerate(self.test_cases):
            moa_result = run_test(moa_code, test_case["input"], test_case["expected"], dual_metrics=True)
            moa_results[test_idx] = moa_result
            if moa_result['text_similarity'] >= SIMILARITY_THRESHOLD:
                moa_passed_similarity += 1
            if moa_result['exact_match'] >= 1.0:
                moa_passed_exact += 1
                
            sllm_result = run_test(sllm_code, test_case["input"], test_case["expected"], dual_metrics=True)
            sllm_results[test_idx] = sllm_result
            if sllm_result['text_similarity'] >= SIMILARITY_THRESHOLD:
                sllm_passed_similarity += 1
            if sllm_result['exact_match'] >= 1.0:
                sllm_passed_exact += 1
        
        # Calculate scores for both metrics
        moa_score_similarity = moa_passed_similarity / total_tests if total_tests > 0 else 0
        moa_score_exact = moa_passed_exact / total_tests if total_tests > 0 else 0
        sllm_score_similarity = sllm_passed_similarity / total_tests if total_tests > 0 else 0
        sllm_score_exact = sllm_passed_exact / total_tests if total_tests > 0 else 0
        
        # Create comparison result
        result = {
            "problem": prompt,
            "timestamp": datetime.now().isoformat(),
            "moa": {
                "scores": {
                    "text_similarity": moa_score_similarity,
                    "exact_match": moa_score_exact
                },
                "passed": {
                    "text_similarity": moa_passed_similarity,
                    "exact_match": moa_passed_exact
                },
                "total": total_tests,
                "time_seconds": moa_time,
                "code": moa_code,
                "test_results": moa_results
            },
            "single_llm": {
                "scores": {
                    "text_similarity": sllm_score_similarity,
                    "exact_match": sllm_score_exact
                },
                "passed": {
                    "text_similarity": sllm_passed_similarity,
                    "exact_match": sllm_passed_exact
                },
                "total": total_tests,
                "time_seconds": sllm_time,
                "code": sllm_code,
                "test_results": sllm_results
            }
        }
        
        self.comparison_results.append(result)
        
        # Print summary
        print("\nEvaluation Results:")
        print(f"MOA Model:")
        print(f"  Text Similarity: {moa_passed_similarity}/{total_tests} tests passed ({moa_score_similarity:.2f})")
        print(f"  Exact Match: {moa_passed_exact}/{total_tests} tests passed ({moa_score_exact:.2f})")
        print(f"  Time: {moa_time:.2f} seconds")
        print(f"SingleLLM:")
        print(f"  Text Similarity: {sllm_passed_similarity}/{total_tests} tests passed ({sllm_score_similarity:.2f})")
        print(f"  Exact Match: {sllm_passed_exact}/{total_tests} tests passed ({sllm_score_exact:.2f})")
        print(f"  Time: {sllm_time:.2f} seconds")
        
        return result
    
    def save_results(self, output_file: str) -> bool:
        """Save comparison results to a file"""
        return save_to_json_file(
            {"comparison_results": self.comparison_results}, 
            output_file
        )

async def generate_all_test_cases_concurrently(problems: List[Dict], single_model, num_tests: int = 10) -> List[Tuple[Dict, List]]:
    """Generate test cases for multiple problems concurrently"""
    print(f"Generating test cases for {len(problems)} problems concurrently...")
    
    # Create tasks for all problems
    tasks = []
    for problem in problems:
        # Handle different data structures
        problem_text = ""
        if isinstance(problem, dict):
            problem_text = problem.get("description", "")
        elif isinstance(problem, list) and len(problem) > 0:
            # If problem is a list, try to get the first element
            if isinstance(problem[0], dict):
                problem_text = problem[0].get("description", "")
            elif isinstance(problem[0], str):
                problem_text = problem[0]
        elif isinstance(problem, str):
            problem_text = problem
            
        if problem_text:
            task = ModelComparer.generate_test_cases_for_problem_static(
                problem_text,
                single_model.client,
                LLM_MODEL,
                num_tests
            )
            tasks.append(task)
        else:
            print(f"Warning: Could not extract problem text from: {type(problem)}")
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    problem_test_cases = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error generating test cases for problem {i+1}: {str(result)}")
            continue
            
        success, test_cases, prompt = result
        if success and test_cases:
            problem_test_cases.append((problems[i], test_cases))
        else:
            print(f"Failed to generate test cases for problem {i+1}")
    
    return problem_test_cases

async def save_generated_test_cases(problem_test_cases: List[Tuple[Dict, List]], test_cases_dir: str) -> None:
    """Save all generated test cases to individual files"""
    if not os.path.exists(test_cases_dir):
        os.makedirs(test_cases_dir, exist_ok=True)
    
    for problem, test_cases in problem_test_cases:
        problem_text = problem.get("description", "")
        
        # Create a sanitized filename
        words = problem_text.split()[:5]
        sanitized_name = "_".join(w.strip().lower() for w in words if w.strip().isalnum())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sanitized_name}_{timestamp}_test_cases.json"
        filepath = os.path.join(test_cases_dir, filename)
        
        # Save test cases and problem to file
        data = {
            "problem": problem_text,
            "test_cases": test_cases,
            "generated_at": datetime.now().isoformat()
        }
        
        if save_to_json_file(data, filepath):
            print(f"Saved {len(test_cases)} test cases to {filepath}")
            
            # Add the file path to the problem for future reference
            problem["test_cases_file"] = filepath

async def run_comparison(moa_param_file: str, output_file: str, num_problems: int = 5, 
                         problems_file: Optional[str] = None, test_cases_dir: Optional[str] = None,
                         save_test_cases: bool = False, concurrent_generation: bool = True):
    """Run comparison on multiple problems"""
    comparer = ModelComparer(moa_param_file)
    
    # Initialize models
    if not await comparer.initialize_models():
        print("Failed to initialize models")
        return False
    
    # Check input parameters
    if not test_cases_dir and not problems_file:
        print("Error: Either problems file or test cases directory must be specified")
        return False
    
    # Create test cases directory if saving is enabled but directory doesn't exist
    if save_test_cases and test_cases_dir:
        os.makedirs(test_cases_dir, exist_ok=True)
        print(f"Will save generated test cases to {test_cases_dir}")
    
    # Check if we should use pre-generated test cases
    if test_cases_dir and os.path.isdir(test_cases_dir):
        print(f"Looking for pre-generated test cases in {test_cases_dir}")
        problems = comparer.load_test_cases_from_directory(test_cases_dir)
        
        if not problems:
            print("No valid test cases found in directory")
            # Only fall back to problems file if one was specified
            if problems_file:
                print("Falling back to problems file")
                use_pregenerated = False
            else:
                print("No problems file provided as fallback")
                return False
        else:
            use_pregenerated = True
    else:
        # If test_cases_dir was specified but doesn't exist or isn't a directory
        if test_cases_dir and not os.path.isdir(test_cases_dir) and save_test_cases:
            print(f"Test cases directory not found, will create: {test_cases_dir}")
        elif test_cases_dir:
            print(f"Test cases directory not found: {test_cases_dir}")
            if not problems_file:
                print("No problems file provided as fallback")
                return False
        use_pregenerated = False
    
    # If not using pre-generated test cases, load problems from file
    if not use_pregenerated:
        if not problems_file or not os.path.isfile(problems_file):
            print(f"Problems file not found: {problems_file}")
            return False
            
        try:
            with open(problems_file, 'r') as f:
                problems_data = json.load(f)
                
            if "problems" in problems_data:
                problems = problems_data["problems"]
                print(f"Debug: Found 'problems' key, loaded {len(problems)} problems")
                if problems and len(problems) > 0:
                    print(f"Debug: First problem type: {type(problems[0])}")
                    if isinstance(problems[0], dict):
                        print(f"Debug: First problem keys: {list(problems[0].keys())}")
                    elif isinstance(problems[0], list):
                        print(f"Debug: First problem is a list with {len(problems[0])} elements")
                        if len(problems[0]) > 0:
                            print(f"Debug: First element type: {type(problems[0][0])}")
            else:
                problems = [problems_data]  # Assume the file itself is a problem
                print(f"Debug: No 'problems' key found, treating entire file as single problem")
        except Exception as e:
            print(f"Error loading problems: {str(e)}")
            traceback.print_exc()
            return False
    
    # If there are more problems than requested, select a random subset
    if len(problems) > num_problems:
        print(f"Selecting {num_problems} random problems from {len(problems)} available")
        problems = random.sample(problems, num_problems)
    else:
        print(f"Using all {len(problems)} available problems")
    
    if not problems:
        print("No problems to evaluate")
        return False
    
    # If not using pre-generated test cases and concurrent generation is enabled,
    # generate all test cases concurrently first
    if not use_pregenerated and concurrent_generation:
        print("Generating test cases concurrently...")
        problem_test_cases = await generate_all_test_cases_concurrently(problems, comparer.single_model)
        
        # Save generated test cases if requested
        if save_test_cases and test_cases_dir:
            await save_generated_test_cases(problem_test_cases, test_cases_dir)
        
        # Process each problem and its test cases
        for i, (problem, test_cases) in enumerate(problem_test_cases, 1):
            # Handle different data structures for problem_text
            problem_text = ""
            if isinstance(problem, dict):
                problem_text = problem.get("description", "")
            elif isinstance(problem, list) and len(problem) > 0:
                if isinstance(problem[0], dict):
                    problem_text = problem[0].get("description", "")
                elif isinstance(problem[0], str):
                    problem_text = problem[0]
            elif isinstance(problem, str):
                problem_text = problem
                
            print(f"\n=== Problem {i}/{len(problem_test_cases)} ===")
            
            if not problem_text:
                print(f"Skipping problem {i} with no description")
                continue
            
            # Set the test cases for this problem
            comparer.test_cases = test_cases
            
            # Evaluate models
            print(f"Evaluating models...")
            await comparer.evaluate_models(problem_text)
    else:
        # Process problems sequentially (either with pre-generated test cases
        # or generating them one by one)
        for i, problem in enumerate(problems, 1):
            # Handle different data structures
            problem_text = ""
            if isinstance(problem, dict):
                problem_text = problem.get("description", "")
            elif isinstance(problem, list) and len(problem) > 0:
                # If problem is a list, try to get the first element
                if isinstance(problem[0], dict):
                    problem_text = problem[0].get("description", "")
                elif isinstance(problem[0], str):
                    problem_text = problem[0]
            elif isinstance(problem, str):
                problem_text = problem
                
            if not problem_text:
                print(f"Skipping problem {i} with no description")
                continue
                
            print(f"\n=== Problem {i}/{len(problems)} ===")
            
            # Check if this problem has pre-generated test cases
            if use_pregenerated and "test_cases_file" in problem:
                print(f"Loading pre-generated test cases...")
                if not comparer.load_test_cases_from_file(problem["test_cases_file"]):
                    print(f"Failed to load test cases for problem {i}, skipping")
                    continue
            else:
                # Generate test cases
                print(f"Generating test cases...")
                if not await comparer.generate_test_cases_for_problem(
                    problem_text, 
                    test_cases_dir=test_cases_dir if save_test_cases else None
                ):
                    print(f"Failed to generate test cases for problem {i}, skipping")
                    continue
            
            # Evaluate models
            print(f"Evaluating models...")
            await comparer.evaluate_models(problem_text)
    
    # Save results
    comparer.save_results(output_file)
    return True

def main():
    parser = argparse.ArgumentParser(description="Compare MOA and SingleLLM models")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to MOA model parameters file")
    parser.add_argument("--problems", "-p", type=str, help="Path to problems file (required if --test-cases not provided)")
    parser.add_argument("--output", "-o", type=str, default="./comparison_results.json", help="Output file for results")
    parser.add_argument("--count", "-c", type=int, default=5, help="Number of problems to compare (selects random subset if more are available)")
    parser.add_argument("--test-cases", "-t", type=str, help="Directory containing pre-generated test cases (required if --problems not provided)")
    parser.add_argument("--save-test-cases", "-s", action="store_true", help="Save generated test cases to the test-cases directory")
    parser.add_argument("--concurrent", action="store_true", help="Generate test cases concurrently for faster execution")
    parser.add_argument("--seed", type=int, help="Random seed for problem selection")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Validate that at least one of problems or test-cases is provided
    if not args.problems and not args.test_cases:
        print("Error: You must specify either --problems or --test-cases")
        parser.print_help()
        sys.exit(1)
    
    # Run comparison in asyncio loop
    asyncio.run(run_comparison(
        moa_param_file=args.model,
        output_file=args.output,
        num_problems=args.count,
        problems_file=args.problems,
        test_cases_dir=args.test_cases,
        save_test_cases=args.save_test_cases,
        concurrent_generation=args.concurrent
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Comparison interrupted by user")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
