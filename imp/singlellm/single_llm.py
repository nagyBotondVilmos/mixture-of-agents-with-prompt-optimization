import asyncio
from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import json
import traceback
from mymoa.moa import NeuronOutput, parse_code_output

def create_enhanced_problem_prompt(prompt: str) -> str:
    """Format a problem prompt to emphasize exact console I/O and output format requirements"""
    enhanced_prompt = f"""
{prompt}

Your code MUST:
1. Read input directly from standard input (stdin)
2. Write output directly to standard output (stdout)
3. Match the EXACT output format specified in the problem statement
4. Handle all test cases, including edge cases
5. Be case-sensitive and space-sensitive in its output

(Examples in the problem show the EXACT expected output format - match it precisely!)
"""
    return enhanced_prompt 

class SingleLLM:
    """Single LLM model that processes inputs directly without a neural network structure"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        """Initialize the single LLM model with API credentials and optional model parameters"""
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.model_goal = "coding"
        self.initialized = True
        self.execution_history = []
        
    async def __call__(self, prompt: str) -> str:
        """Process a prompt and return the result - interface compatible with MLP.__call__"""
        if not self.initialized:
            raise ValueError("Parameters not initialized!")
            
        # Clear previous execution history
        self.execution_history = []
        
        print(f"SingleLLM processing input")
        
        system_prompt = """You are an expert programmer tasked with solving a coding problem.
Your solution must be efficient, correct, and follow the exact requirements.
Analyze the problem carefully, implement an optimal solution, and ensure it handles all edge cases."""
        
        # Enhance prompt with formatting requirements
        enhanced_prompt = create_enhanced_problem_prompt(prompt)
        
        # Additional prompt guidance for JSON output format
        enhanced_prompt += """
IMPORTANT: Your response MUST be formatted as a valid JSON object with these fields:
language: the programming language of the code (use "python")
code: the code solution to the prompt as plain text

EXAMPLE JSON RESPONSE FORMAT:
{
    "language": "python",
    "code": "# Read input from stdin\\nimport sys\\n\\n# Your algorithm here\\n\\n# Write output to stdout\\nprint('EXACT_EXPECTED_OUTPUT')"
}

DO NOT include explanations, comments, or any text outside the JSON object.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        try:
            # Make the API call with timeout
            result = None
            
            try:
                # Use a timeout for the LLM call similar to the MLP implementation
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3
                    ),
                    timeout=90  # 90 second timeout to match nn.py
                )
                result = response.choices[0].message.content
            except asyncio.TimeoutError:
                print("SingleLLM processing timed out after 90 seconds")
                result = json.dumps({
                    "language": "python",
                    "code": "# LLM timeout\nimport sys\nprint('ERROR: LLM processing timeout')",
                    "error": "timeout"
                })
            except Exception as e:
                print(f"Error in LLM call: {str(e)}")
                traceback.print_exc()
                result = json.dumps({
                    "language": "python",
                    "code": f"# Error in LLM call\nimport sys\nprint('ERROR: {str(e).replace('\"', '\\\"')}')",
                    "error": str(e)
                })
            
            # Parse the output and add to execution history
            parsed_output = parse_code_output(result)
            
            # Create NeuronOutput to maintain compatibility with MLP
            neuron_output = NeuronOutput(
                layer_idx=0,
                neuron_idx=0,
                input_text=enhanced_prompt,
                output_text=result,
                parsed_output=parsed_output
            )
            
            # Store in execution history similar to MLP
            self.execution_history.append({
                "layer_idx": 0,
                "outputs": [neuron_output.to_dict()]
            })
            
            print(f"SingleLLM processed input")
            return result
            
        except Exception as e:
            print(f"Error in single LLM processing: {str(e)}")
            traceback.print_exc()
            
            # Create error response and add to history
            error_result = json.dumps({
                "language": "python",
                "code": f"# Error in LLM processing\nimport sys\nprint('ERROR: {str(e).replace('\"', '\\\"')}')",
                "error": str(e)
            })
            
            # Create error output for history
            error_output = NeuronOutput(
                layer_idx=0,
                neuron_idx=0,
                input_text=enhanced_prompt,
                output_text=error_result,
                parsed_output=parse_code_output(error_result)
            )
            
            self.execution_history.append({
                "layer_idx": 0,
                "error": str(e),
                "outputs": [error_output.to_dict()]
            })
            
            return error_result
    
    def parameters(self) -> Dict[str, Any]:
        """Return model parameters - interface compatible with MLP.parameters()"""
        if self.model_params:
            return self.model_params
        else:
            return {
                "model_goal": self.model_goal,
                "model": self.model
            }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history with neuron outputs - interface compatible with MLP"""
        return self.execution_history
    
    def get_neuron_output(self, layer_idx: int, neuron_idx: int) -> Optional[Dict[str, Any]]:
        """Get output from the LLM - interface compatible with MLP.get_neuron_output()"""
        if layer_idx != 0 or neuron_idx != 0 or not self.execution_history:
            return None
            
        # Get the first (and only) neuron output
        try:
            return self.execution_history[0]["outputs"][0]
        except (IndexError, KeyError):
            return None 