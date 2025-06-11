import random
import asyncio
from openai import AsyncOpenAI
from mymoa.parameters import BiasGenerator, weight_to_text
from common.secret import LLM_API_KEY_MOA, LLM_API_URL, LLM_MODEL
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union
import json

class NeuronOutput:
    """Class to store and track neuron outputs for testing"""
    def __init__(self, layer_idx: int, neuron_idx: int, input_text: str, output_text: str, parsed_output: Optional[Dict] = None):
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.input_text = input_text
        self.output_text = output_text
        self.parsed_output = parsed_output
        self.timestamp = asyncio.get_event_loop().time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "layer_idx": self.layer_idx,
            "neuron_idx": self.neuron_idx,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "parsed_output": self.parsed_output,
            "timestamp": self.timestamp
        }

def parse_code_output(output_text: str) -> Optional[Dict[str, Any]]:
    """Parse the output text to extract code in JSON format or from code blocks"""
    try:
        # Try to parse the entire output as JSON
        parsed = json.loads(output_text)
        if isinstance(parsed, dict) and "language" in parsed and "code" in parsed:
            return parsed
    except:
        # If not valid JSON, try to extract JSON block using common markers
        try:
            if "```json" in output_text and "```" in output_text:
                json_content = output_text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_content)
                if isinstance(parsed, dict) and "language" in parsed and "code" in parsed:
                    return parsed
        except:
            pass
            
        # Try to extract Python code blocks
        try:
            if "```python" in output_text and "```" in output_text:
                code_content = output_text.split("```python")[1].split("```")[0].strip()
                return {
                    "language": "python",
                    "code": code_content
                }
            elif "```" in output_text:
                # Try with generic code block
                parts = output_text.split("```")
                if len(parts) >= 3:  # At least one complete code block
                    code_content = parts[1].strip()
                    return {
                        "language": "python",
                        "code": code_content
                    }
        except:
            pass
    
    # If all else fails, check if the text itself looks like Python code
    if "def " in output_text or "import " in output_text or "print(" in output_text:
        # It might be raw code without markers
        return {
            "language": "python",
            "code": output_text.strip()
        }
    
    # If all attempts fail, return None
    return None

class Neuron:
    def __init__(self,
                 model_goal: str,
                 neuron_idx: int,
                 layer_idx: int,
                 is_final: bool = False,
                 initialized: bool = False):
        self.bias_generator = BiasGenerator()
        self.system_prompt = None
        self.w = None
        self.model_goal = model_goal
        self.neuron_idx = neuron_idx
        self.layer_idx = layer_idx
        self.is_final = is_final
        self.initialized = initialized
        self.last_output = None  # Store the last output for testing

    @classmethod
    async def create(cls,
                    nin: int,
                    model_goal: str,
                    neuron_idx: int,
                    layer_idx: int,
                    is_final: bool = False,
                    initialized: bool = False):
        neuron = cls(model_goal, neuron_idx, layer_idx, is_final, initialized)
        
        if initialized:
            try:
                neuron.w = [random.uniform(0, 1) for _ in range(nin)]
                neuron.system_prompt = await neuron.bias_generator.generate_random_bias(model_goal)
                neuron.initialized = True
            except Exception as e:
                print(f"Error initializing neuron ({layer_idx}, {neuron_idx}): {str(e)}")
                traceback.print_exc()
                raise
        
        return neuron

    def load_parameters(self,
                        parameters: dict,
                        model_goal: str):
        self.w = parameters["w"]
        self.system_prompt = parameters["system_prompt"]
        self.model_goal = model_goal
        self.initialized = True
    
    def construct_prompt(self,
                         prompt: str,
                         previous_solutions: list[str]):
        # Base prompt instructions
        output_format_instructions = """
IMPORTANT: Your response MUST be formatted as a valid JSON object with these fields:
language: the programming language of the code (use "python")
code: the code solution to the prompt as plain text

Your code MUST:
1. Read input directly from standard input (stdin)
2. Write output directly to standard output (stdout)
3. Match the EXACT output format specified in the problem statement
4. Handle all test cases, including edge cases
5. Be case-sensitive and space-sensitive in its output

EXAMPLE JSON RESPONSE FORMAT:
{
    "language": "python",
    "code": "# Read input from stdin\\nimport sys\\n\\n# Your algorithm here\\n\\n# Write output to stdout\\nprint('EXACT_EXPECTED_OUTPUT')"
}

DO NOT include explanations, comments, or any text outside the JSON object.
"""
        # Construct the prompt based on previous solutions
        if len(previous_solutions) == 0:
            formatted_prompt = f"{prompt}\n\n{output_format_instructions}"
        else:
            context = "\n\n".join([
                f"Previous solution number {i+1} (significance: {weight_to_text(self.w[i])}): {sol}"
                for i, sol in enumerate(previous_solutions)
            ])
            formatted_prompt = f"Previous solutions:\n{context}\n\nOriginal prompt: {prompt}\n\n{output_format_instructions}"
            
        # Add final instruction if this is the final neuron
        if self.is_final:
            formatted_prompt += "\nThis is the final output that will be presented to the user. Make sure your code is correct, efficient, and follows best practices."
            
        return formatted_prompt

    async def __call__(self, x):
        if not self.initialized:
            raise ValueError(f"Neuron ({self.layer_idx}, {self.neuron_idx}) not initialized!")
            
        if isinstance(x, tuple):
            prompt, previous_solutions = x
        else:
            prompt, previous_solutions = x, []  # Initialize as empty list if no previous solutions
        
        print(f"\t\tNeuron ({self.layer_idx}, {self.neuron_idx}) processing input")
        
        if self.w is None or self.system_prompt is None:
            raise ValueError(f"Neuron ({self.layer_idx}, {self.neuron_idx}) has uninitialized weights or system prompt")
        
        formatted_prompt = self.construct_prompt(prompt, previous_solutions)
        
        # Enhance system prompt to enforce console IO and exact output
        enhanced_system_prompt = f"""
{self.system_prompt}

CRITICAL FORMATTING INSTRUCTIONS:

1. Your code MUST read input from standard input (stdin) and write output to standard output (stdout).

2. The output format MUST EXACTLY match what is required in the problem description:
   - Match EXACT case ("yes" vs "Yes" vs "YES")
   - Match EXACT spacing and line breaks
   - Match EXACT punctuation and symbols

3. Your response must be a valid JSON with this format:
{{
    "language": "python",
    "code": "# Your python code here"
}}

4. DO NOT include explanations outside the JSON.

5. IMPORTANT: Pay close attention to the required output format in examples. If the example shows the output as "3 5", your code must print exactly "3 5" (not "[3, 5]" or anything else).
"""
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        result = None
        
        try:
            # Create a fresh client for each request to avoid connection issues
            client_kwargs = {
                "api_key": LLM_API_KEY_MOA[self.layer_idx % len(LLM_API_KEY_MOA)],
                "base_url": LLM_API_URL
            }
            fresh_client = AsyncOpenAI(**client_kwargs)
            
            # Set timeout for the LLM call
            response = await fresh_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.3,
                timeout=90  # 90 second client timeout
            )
            result = response.choices[0].message.content
            
            # Explicitly close the client after use
            await fresh_client.close()
            
        except Exception as e:
            print(f"\t\tError calling LLM for neuron ({self.layer_idx}, {self.neuron_idx}): {str(e)}")
            traceback.print_exc()
            # Return a structured error message that can be parsed
            result = json.dumps({
                "language": "python",
                "code": f"# Error in LLM call\nimport sys\nprint('ERROR: {str(e).replace('\"', '\\\"')}')",
                "error": str(e)
            })
        
        print(f"\t\tNeuron ({self.layer_idx}, {self.neuron_idx}) processed input")

        # Parse the output to extract code (handle errors)
        try:
            parsed_output = parse_code_output(result)
        except Exception as e:
            print(f"\t\tError parsing output for neuron ({self.layer_idx}, {self.neuron_idx}): {str(e)}")
            parsed_output = {
                "language": "python",
                "code": f"# Error parsing output\nimport sys\nprint('ERROR: {str(e).replace('\"', '\\\"')}')",
                "error": "parsing_error"
            }
        
        # Store output for testing
        self.last_output = NeuronOutput(
            layer_idx=self.layer_idx,
            neuron_idx=self.neuron_idx,
            input_text=formatted_prompt,
            output_text=result,
            parsed_output=parsed_output
        )
        
        return result

    def parameters(self):
        params = {
            "w": self.w,
            "system_prompt": self.system_prompt
        }
        return params
        
    def get_last_output(self) -> Optional[NeuronOutput]:
        """Get the last output from this neuron"""
        return self.last_output

class Layer:
    def __init__(self,
                 nin: int,
                 nout: int,
                 model_goal: str,
                 layer_idx: int,
                 is_final_layer: bool = False,
                 initialized: bool = False):
        self.nin = nin
        self.nout = nout
        self.model_goal = model_goal
        self.layer_idx = layer_idx
        self.is_final_layer = is_final_layer
        self.initialized = initialized
        self.neurons = []
        self.last_outputs = []  # Store the last outputs for testing
    
    @classmethod
    async def create(cls,
                    nin: int,
                    nout: int,
                    model_goal: str,
                    layer_idx: int,
                    is_final_layer: bool = False,
                    initialized: bool = False):
        layer = cls(nin, nout, model_goal, layer_idx, is_final_layer, initialized)
        layer.neurons = [
            await Neuron.create(
                nin=nin,
                model_goal=model_goal,
                neuron_idx=neuron_idx,
                layer_idx=layer_idx,
                is_final=(is_final_layer and neuron_idx == 0),
                initialized=initialized
            ) for neuron_idx in range(nout)
        ]
        return layer

    def load_parameters(self,
                        parameters: list,
                        model_goal: str):
        for neuron, neuron_params in zip(self.neurons, parameters):
            neuron.load_parameters(neuron_params, model_goal)
        self.model_goal = model_goal
        self.initialized = True

    async def __call__(self, x):
        print(f"\tLayer {self.layer_idx} processing input")
        
        # Process neurons with timeouts and error handling
        self.last_outputs = []
        
        # Create tasks for all neurons to run in parallel
        async def process_neuron(neuron, neuron_idx):
            try:
                # Set a timeout for neuron processing
                result = await asyncio.wait_for(neuron(x), timeout=90)  # 1.5 minute timeout per neuron
                
                # Store successful output
                output = neuron.get_last_output()
                return {
                    "neuron_idx": neuron_idx,
                    "result": result,
                    "output": output,
                    "success": True
                }
            except asyncio.TimeoutError:
                print(f"\t\tNeuron {neuron_idx} processing timed out after 90 seconds")
                # Create a fallback result
                fallback_result = json.dumps({
                    "language": "python",
                    "code": "# Neuron timeout\nimport sys\nprint('ERROR: Neuron processing timeout')",
                    "error": "timeout"
                })
                
                # Add fallback output for history
                fallback_output = NeuronOutput(
                    layer_idx=self.layer_idx,
                    neuron_idx=neuron_idx,
                    input_text=str(x),
                    output_text=fallback_result,
                    parsed_output=parse_code_output(fallback_result)
                )
                
                return {
                    "neuron_idx": neuron_idx,
                    "result": fallback_result,
                    "output": fallback_output,
                    "success": False
                }
            except Exception as e:
                print(f"\t\tError processing neuron {neuron_idx}: {str(e)}")
                traceback.print_exc()
                # Create a fallback result
                fallback_result = json.dumps({
                    "language": "python",
                    "code": f"# Neuron error\nimport sys\nprint('ERROR: {str(e).replace('\"', '\\\"')}')",
                    "error": str(e)
                })
                
                # Add fallback output for history
                fallback_output = NeuronOutput(
                    layer_idx=self.layer_idx,
                    neuron_idx=neuron_idx,
                    input_text=str(x),
                    output_text=fallback_result,
                    parsed_output=parse_code_output(fallback_result)
                )
                
                return {
                    "neuron_idx": neuron_idx,
                    "result": fallback_result,
                    "output": fallback_output,
                    "success": False
                }
        
        # Create and gather all neuron tasks to run them concurrently
        tasks = [process_neuron(neuron, i) for i, neuron in enumerate(self.neurons)]
        results = await asyncio.gather(*tasks)
        
        # Sort results by neuron index to maintain order
        results.sort(key=lambda x: x["neuron_idx"])
        
        # Extract results and outputs in original order
        results_list = [r["result"] for r in results]
        self.last_outputs = [r["output"] for r in results]
        
        print(f"\tLayer {self.layer_idx} processed input")
        
        # If no results, return error
        if not results_list:
            return json.dumps({
                "language": "python",
                "code": "# All neurons failed\nimport sys\nprint('ERROR: All neurons failed')",
                "error": "all_failed"
            })
        
        return results_list[0] if len(results_list) == 1 else results_list

    def parameters(self):
        return {
            "neurons": [n.parameters() for n in self.neurons]
        }
        
    def get_outputs(self) -> List[NeuronOutput]:
        """Get outputs from all neurons in this layer"""
        return self.last_outputs

class MOA:
    def __init__(self,
                 nouts: list[int],
                 model_goal: str = "coding",
                 initialized: bool = False):
        self.initialized = initialized
        self.sz = [1] + nouts
        self.model_goal = model_goal
        self.layers = []
        self.execution_history = []  # Track execution history for testing
    
    @classmethod
    async def create(cls,
                    nouts: list[int] = [1],
                    model_goal: str = "coding",
                    initialized: bool = False):
        mlp = cls(nouts, model_goal, initialized)
        print(f"MOA initializing with {len(nouts)} layers")
        
        mlp.layers = [
            await Layer.create(
                nin=mlp.sz[layer_idx],
                nout=mlp.sz[layer_idx+1],
                model_goal=model_goal,
                layer_idx=layer_idx,
                is_final_layer=(layer_idx == len(nouts)-1),
                initialized=initialized
            ) for layer_idx in range(len(nouts))
        ]

        print(f"MOA initialized with {len(mlp.layers)} layers")
        return mlp

    def load_parameters(self,
                        parameters: dict):
        for layer, layer_params in zip(self.layers, parameters["layers"]):
            layer.load_parameters(layer_params["neurons"], parameters["model_goal"])
        self.model_goal = parameters["model_goal"]
        self.initialized = True

    async def __call__(self, prompt):
        if not self.initialized:
            raise ValueError("Parameters not initialized!")
        
        # Clear previous execution history
        self.execution_history = []
        
        print(f"MOA processing input")
        x = prompt
        
        for i, layer in enumerate(self.layers):
            try:
                # Define layer processing function based on layer index
                if i == 0:
                    async def process_layer():
                        return await layer(x)
                else:
                    async def process_layer():
                        return await layer((prompt, x if isinstance(x, list) else [x]))
                
                # Use a shorter timeout per layer to prevent hanging
                x = await asyncio.wait_for(process_layer(), timeout=120)  # 2 minute timeout per layer
                
                # Store layer outputs in execution history if available
                try:
                    self.execution_history.append({
                        "layer_idx": i,
                        "outputs": [output.to_dict() for output in layer.get_outputs()]
                    })
                except Exception as e:
                    print(f"Error storing execution history for layer {i}: {str(e)}")
                    # Create a placeholder to track the error
                    self.execution_history.append({
                        "layer_idx": i,
                        "error": str(e)
                    })
                
            except asyncio.TimeoutError:
                print(f"Layer {i} processing timed out after 120 seconds")
                # Continue with a minimal result on timeout
                x = json.dumps({
                    "language": "python",
                    "code": "# Layer timeout occurred\nimport sys\nprint('ERROR: Layer processing timeout')",
                    "error": "timeout"
                })
                
                # Add error to execution history
                self.execution_history.append({
                    "layer_idx": i,
                    "error": "timeout",
                    "outputs": []
                })
                
            except Exception as e:
                print(f"Error processing layer {i}: {str(e)}")
                traceback.print_exc()
                # Continue with a minimal result on error
                x = json.dumps({
                    "language": "python",
                    "code": f"# Error in layer processing\nimport sys\nprint('ERROR: {str(e).replace('\"', '\\\"')}')",
                    "error": str(e)
                })
                
                # Add error to execution history
                self.execution_history.append({
                    "layer_idx": i,
                    "error": str(e),
                    "outputs": []
                })
            
            # Force garbage collection to help prevent memory leaks
            import gc
            gc.collect()
            
        print(f"MOA processed input")
        return x

    def parameters(self):
        return {
            "model_goal": self.model_goal,
            "layers": [l.parameters() for l in self.layers]
        }
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the full execution history with all neuron outputs for testing"""
        return self.execution_history
        
    def get_neuron_output(self, layer_idx: int, neuron_idx: int) -> Optional[Dict[str, Any]]:
        """Get output from a specific neuron by layer and neuron index"""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return None
            
        if neuron_idx < 0 or neuron_idx >= len(self.layers[layer_idx].neurons):
            return None
            
        output = self.layers[layer_idx].neurons[neuron_idx].get_last_output()
        return output.to_dict() if output else None
