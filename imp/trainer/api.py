from flask import Flask, request, jsonify
import asyncio
from mymoa.moa import MOA
from mymoa.parameters import (
    validate_request_structure,
    validate_parameters,
    validate_layer_sizes
)
import traceback

app = Flask(__name__)
model = None

async def initialize_model(user_init: bool = False,
                         model_goal: str = "coding",
                         parameters: dict = None,
                         layer_sizes: list[int] = None):
    global model
    try:
        if user_init:
            if not parameters or "layers" not in parameters:
                raise ValueError("Parameters with layers required when user_init is true")
                
            complete_params = {
                "model_goal": model_goal,
                "layers": parameters["layers"]
            }
            
            is_valid, error_message = validate_parameters(complete_params)
            if not is_valid:
                raise ValueError(f"Invalid parameters: {error_message}")
            nouts = [len(layer["neurons"]) for layer in parameters["layers"]]
        else:
            is_valid, error_message = validate_layer_sizes(layer_sizes)
            if not is_valid:
                raise ValueError(f"Invalid layer sizes: {error_message}")
            nouts = layer_sizes
        
        model = await MOA.create(
            nouts=nouts,
            model_goal=model_goal,
            initialized=not user_init
        )
        
        if user_init:
            model.load_parameters(complete_params)
        
        return model
        
    except Exception as e:
        print(f"Error in initialize_model: {str(e)}")
        raise

@app.route("/init", methods=["POST"])
def initialize():
    global model
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No initialization data provided"}), 400
            
        # Validate request structure
        is_valid, error_message = validate_request_structure(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400
        
        # Create a new event loop for initialization to avoid conflicts
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        
        try:
            # Run initialization with a timeout
            async def init_with_timeout():
                return await asyncio.wait_for(
                    initialize_model(
                        user_init=data["user_init"],
                        model_goal=data["model_goal"],
                        parameters=data.get("parameters"),
                        layer_sizes=data.get("layer_sizes")
                    ),
                    timeout=180  # 3 minute timeout
                )
                
            model = new_loop.run_until_complete(init_with_timeout())
        except asyncio.TimeoutError:
            print("Model initialization timed out after 180 seconds")
            new_loop.close()
            return jsonify({"error": "Model initialization timed out"}), 504
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            traceback.print_exc()
            new_loop.close()
            return jsonify({"error": str(e)}), 500
        finally:
            # Ensure the loop is closed
            if not new_loop.is_closed():
                new_loop.close()
        
        if model is None:
            return jsonify({"error": "Failed to initialize model"}), 500
            
        return jsonify({
            "message": "Model initialized successfully",
            "initialization_type": "user_parameters" if data["user_init"] else "random",
            "model_goal": data["model_goal"],
            "layer_sizes": data.get("layer_sizes") if not data["user_init"] else [len(layer["neurons"]) for layer in data["parameters"]["layers"]]
        })
        
    except Exception as e:
        print(f"Error in initialize endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
async def predict():
    if model is None:
        return jsonify({"error": "Model not initialized. Call /init first"}), 400
        
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        result = await model(data["prompt"])
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/execution-history", methods=["GET"])
def get_execution_history():
    """Get the complete execution history with all neuron outputs"""
    if model is None:
        return jsonify({"error": "Model not initialized. Call /init first"}), 400
        
    try:
        history = model.get_execution_history()
        return jsonify({
            "execution_history": history,
            "layer_count": len(model.layers),
            "neuron_counts": [len(layer.neurons) for layer in model.layers]
        })
    except Exception as e:
        print(f"Error in execution-history endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/neuron-output/<int:layer_idx>/<int:neuron_idx>", methods=["GET"])
def get_neuron_output(layer_idx, neuron_idx):
    """Get output from a specific neuron"""
    if model is None:
        return jsonify({"error": "Model not initialized. Call /init first"}), 400
        
    try:
        output = model.get_neuron_output(layer_idx, neuron_idx)
        if output is None:
            return jsonify({"error": f"No output found for neuron ({layer_idx}, {neuron_idx})"}), 404
            
        return jsonify(output)
    except Exception as e:
        print(f"Error in neuron-output endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/extract-code/<int:layer_idx>/<int:neuron_idx>", methods=["GET"])
def extract_code(layer_idx, neuron_idx):
    """Extract formatted code from a specific neuron for testing"""
    if model is None:
        return jsonify({"error": "Model not initialized. Call /init first"}), 400
        
    try:
        output = model.get_neuron_output(layer_idx, neuron_idx)
        if output is None:
            return jsonify({"error": f"No output found for neuron ({layer_idx}, {neuron_idx})"}), 404
           
        # Import parse_code_output and handle multiple extraction methods
        from mymoa.moa import parse_code_output
        
        # First try using the parsed output if available
        if output.get("parsed_output"):
            return jsonify({
                "language": output["parsed_output"]["language"],
                "code": output["parsed_output"]["code"],
                "layer_idx": layer_idx,
                "neuron_idx": neuron_idx,
                "parsed": True
            })
        
        # Otherwise try to parse it ourselves
        parsed = parse_code_output(output["output_text"])
        if parsed and "code" in parsed:
            return jsonify({
                "language": parsed["language"],
                "code": parsed["code"],
                "layer_idx": layer_idx,
                "neuron_idx": neuron_idx,
                "parsed": True
            })
            
        # If we couldn't parse, provide a sample and the raw text
        sample_code = '''
# Read input from stdin
import sys
input_str = sys.stdin.read().strip()
# Example parsing input
lines = input_str.strip().split('\\n')
# Process the problem logic
# Output the exact required format
print(result)  # Match the expected format precisely
'''
            
        return jsonify({
            "error": "Could not parse code from output",
            "raw_output": output["output_text"],
            "sample_code": sample_code,
            "parsed": False
        }), 422
        
    except Exception as e:
        print(f"Error in extract-code endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/test-neurons", methods=["POST"])
def test_neurons():
    """Test code from multiple neurons with a single request"""
    if model is None:
        return jsonify({"error": "Model not initialized. Call /init first"}), 400
        
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        
        # Create a new event loop for this request to avoid conflicts with other requests
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        
        # Run the model with a timeout
        try:
            # Wrap the model call in a timeout
            async def run_with_timeout():
                return await asyncio.wait_for(model(data["prompt"]), timeout=len(model.layers) * 120)
                
            result = new_loop.run_until_complete(run_with_timeout())
        except asyncio.TimeoutError:
            print(f"Model execution timed out after {len(model.layers) * 120} seconds")
            # Clean up the loop
            new_loop.close()
            return jsonify({
                "error": "Model execution timed out", 
                "neuron_outputs": []
            }), 504
        except Exception as e:
            print(f"Error running model: {str(e)}")
            traceback.print_exc()
            # Clean up the loop
            new_loop.close()
            return jsonify({
                "error": f"Error: {str(e)}", 
                "neuron_outputs": []
            }), 500
        finally:
            # Ensure the loop is closed even if we return early
            if not new_loop.is_closed():
                new_loop.close()
            
        # Import parse_code_output for code extraction
        from mymoa.moa import parse_code_output
        
        # Collect all neuron outputs for testing
        test_results = []
        for layer_idx, layer in enumerate(model.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                output = neuron.get_last_output()
                if output:
                    # Initialize with basic data
                    code_data = {
                        "layer_idx": layer_idx,
                        "neuron_idx": neuron_idx,
                        "parsed": False,
                        "output_text": output.output_text
                    }
                    
                    # Try multiple methods to extract code
                    if output.parsed_output:
                        # Use the neuron's parsed output
                        code_data["parsed"] = True
                        code_data["language"] = output.parsed_output["language"]
                        code_data["code"] = output.parsed_output["code"]
                    else:
                        # Try our own parsing
                        try:
                            parsed = parse_code_output(output.output_text)
                            if parsed and "code" in parsed:
                                code_data["parsed"] = True
                                code_data["language"] = parsed["language"]
                                code_data["code"] = parsed["code"]
                                print(f"Successfully parsed code for neuron ({layer_idx}, {neuron_idx})")
                        except Exception as e:
                            print(f"Error parsing code for neuron ({layer_idx}, {neuron_idx}): {str(e)}")
                    
                    # Check for input/output operations
                    if code_data.get("parsed", False) and "code" in code_data:
                        code = code_data["code"]
                        if "input(" in code or "sys.stdin" in code or "stdin" in code:
                            code_data["has_input"] = True
                        else:
                            code_data["has_input"] = False
                            
                        if "print(" in code or "sys.stdout" in code or "stdout" in code:
                            code_data["has_output"] = True
                        else:
                            code_data["has_output"] = False
                        
                    test_results.append(code_data)
        
        return jsonify({
            "final_result": result,
            "neuron_outputs": test_results,
            "output_count": len(test_results)
        })
        
    except Exception as e:
        print(f"Error in test-neurons endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/params", methods=["GET"])
def get_parameters():
    if model is None:
        return jsonify({"error": "Model not initialized. Call /init first"}), 400
        
    try:
        params = model.parameters()
        return jsonify(params)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/params", methods=["POST"])
def update_parameters():
    if model is None:
        return jsonify({"error": "Model not initialized. Call /initialize first"}), 400
        
    try:
        new_params = request.get_json()
        if not new_params:
            return jsonify({"error": "No parameters provided"}), 400
            
        is_valid, error_message = validate_parameters(new_params)
        if not is_valid:
            return jsonify({"error": f"Invalid parameters: {error_message}"}), 400
            
        model.load_parameters(new_params)
        return jsonify({"message": "Parameters updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)
