#!/usr/bin/env python3
import json
import re
import os
import sys
import tempfile
import subprocess
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from openai import AsyncOpenAI
import difflib

def extract_json(text: str) -> Optional[Dict]:
    """Extract a JSON object from text, handling various formats"""
    try:
        # First try to parse the entire text as JSON
        return json.loads(text)
    except:
        # Look for JSON in code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except:
                pass
        
        # Look for JSON with brace pattern
        brace_pattern = r"\{[\s\S]*\}"
        matches = re.findall(brace_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except:
                pass
        
        return None

def one_on_one_comparison(expected_output: str, actual_output: str) -> float:
    """Compare the expected and actual outputs of a single test case"""
    if expected_output == actual_output:
        return 1.0
    else:
        return 0.0

def text_similarity_comparison(expected_output: str, actual_output: str) -> float:
    """Compare the expected and actual outputs of a single test case using text similarity"""
    return difflib.SequenceMatcher(None, expected_output, actual_output).ratio()

def run_test(code: str, test_input: str, expected_output: str, comparison_function: Callable[[str, str], float] = None, dual_metrics: bool = False) -> float | Dict[str, float]:
    """Run a single test case on the provided code
    
    Args:
        code: The code to test
        test_input: Input for the test
        expected_output: Expected output
        comparison_function: Single comparison function (for backward compatibility)
        dual_metrics: If True, returns both text similarity and one-on-one comparison
    
    Returns:
        float: Single metric result (when dual_metrics=False)
        Dict[str, float]: Both metrics (when dual_metrics=True) with keys 'text_similarity' and 'exact_match'
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp:
            temp.write(code)
            temp_path = temp.name
        
        process = subprocess.Popen(
            [sys.executable, temp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(input=test_input, timeout=5)
            actual_output = stdout.rstrip()
            expected_output = str(expected_output).rstrip()
            
            if dual_metrics:
                # Return both metrics
                return {
                    'text_similarity': text_similarity_comparison(expected_output, actual_output),
                    'exact_match': one_on_one_comparison(expected_output, actual_output)
                }
            else:
                # Return single metric for backward compatibility
                return comparison_function(expected_output, actual_output)
                
        except subprocess.TimeoutExpired:
            process.kill()
            if dual_metrics:
                return {'text_similarity': 0.0, 'exact_match': 0.0}
            else:
                return 0.0
    except Exception as e:
        print(f"Error running test: {str(e)}")
        traceback.print_exc()
        if dual_metrics:
            return {'text_similarity': 0.0, 'exact_match': 0.0}
        else:
            return 0.0
    finally:
        # Clean up temp file
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass

def extract_system_prompt(text: str) -> Optional[str]:
    """Extract a system prompt enclosed in <system_prompt> tags"""
    pattern = r"<system_prompt>([\s\S]*?)</system_prompt>"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def format_problem_prompt(prompt: str) -> str:
    """Format the problem prompt to emphasize exact console I/O and output format"""
    formatted_prompt = f"""
# CODING PROBLEM

{prompt}

## IMPORTANT REQUIREMENTS

1. Your code MUST read input DIRECTLY from standard input (stdin)
2. Your code MUST write output DIRECTLY to standard output (stdout) 
3. The output format MUST MATCH EXACTLY what is specified in the problem
4. Pay careful attention to capitalization, spacing, and formatting in the examples

(Examples in the problem show the EXACT expected output format - match it precisely!)
"""
    return formatted_prompt

def save_to_json_file(data: Any, filepath: str, indent: int = 2) -> bool:
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving to JSON file: {str(e)}")
        traceback.print_exc()
        return False

async def generate_test_cases(prompt: str, client: AsyncOpenAI, model: str, num_tests: int = 10) -> Tuple[bool, Optional[List[Dict]]]:
    """Generate test cases for a problem prompt using an LLM"""
    print(f"Generating {num_tests} test cases for the problem...")
    
    try:
        # Define expected output format in the prompt
        test_gen_prompt = f"""
I need test cases for the following coding problem:

{prompt}

Generate {num_tests} test cases with input and expected output. Cover different edge cases and scenarios.
Include simple test cases and more complex scenarios.

Each test case must include EXACTLY two fields:
1. "input": a single string that will be passed to the solution
2. "expected": the expected output as a string

Format your response as a valid JSON with this exact structure:
{{
    "test_cases": [
        {{
            "input": "test input here",
            "expected": "expected output here"
        }},
        ...
    ]
}}

Input and output should be a single string, not a list of strings.
If you need the newline character, use \\n.
"""
        system_prompt = """You are a test case generator for coding problems.

IMPORTANT RULES:
1. ONLY generate a JSON response with the test_cases array
2. Each test case must have EXACTLY 'input' and 'expected' fields as strings
3. Do NOT include any explanations outside the JSON
4. Make sure your JSON is valid and properly formatted
5. Include a variety of test cases: simple, edge cases, large inputs, etc.
6. Be PRECISE about the expected output format - it should exactly match what a solution would return
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_gen_prompt}
        ]
        
        # Use the client to generate test cases
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            ),
            timeout=120  # 2 minute timeout
        )
        
        result = response.choices[0].message.content
        
        # Try to extract JSON from the response
        test_cases_data = extract_json(result)
        if not test_cases_data or "test_cases" not in test_cases_data:
            print("Failed to extract valid test cases JSON")
            print("Raw output:", result)
            return False, None
            
        test_cases = test_cases_data["test_cases"]
        print(f"Generated {len(test_cases)} test cases")
        return True, test_cases
        
    except Exception as e:
        print(f"Error generating test cases: {str(e)}")
        traceback.print_exc()
        return False, None

def save_test_cases(prompt: str, test_cases: List[Dict], filepath: str) -> bool:
    """Save test cases to a JSON file"""
    data = {
        "problem": prompt,
        "test_cases": test_cases,
        "generated_at": __import__('datetime').datetime.now().isoformat()
    }
    return save_to_json_file(data, filepath, indent=4)
