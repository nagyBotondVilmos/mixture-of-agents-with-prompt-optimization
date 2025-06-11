import time
from openai import AsyncOpenAI
from common.secret import LLM_API_KEY_AUX, LLM_API_URL, LLM_MODEL

### BIAS

class BiasGenerator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=LLM_API_KEY_AUX, base_url=LLM_API_URL)

    def _extract_content(self, text: str) -> str:
        """Extract content from various delimiter formats, with multiple fallbacks"""
        # Try different delimiters in order of preference
        delimiters = [
            ('```', '```'),  # Triple backticks
            ('<system_prompt>', '</system_prompt>'),  # XML tags
            ('<rephrased_text>', '</rephrased_text>'),  # XML tags
            ('"""', '"""'),  # Triple quotes
            ('"', '"'),  # Double quotes
            ("BEGIN:", "END:"),  # BEGIN/END markers
        ]
        
        for start_delim, end_delim in delimiters:
            if start_delim in text and end_delim in text:
                try:
                    parts = text.split(start_delim, 1)
                    if len(parts) > 1:
                        content = parts[1].split(end_delim, 1)[0].strip()
                        if content:
                            return content
                except:
                    continue
                        
        # If no delimiters found, use a regex pattern to find something that looks like a system prompt
        try:
            import re
            prompt_patterns = [
                r"You are (.*?)(?:\.|$)",
                r"Act as (.*?)(?:\.|$)",
                r"Your role is (.*?)(?:\.|$)",
            ]
            
            for pattern in prompt_patterns:
                matches = re.search(pattern, text, re.DOTALL)
                if matches:
                    return text
        except:
            pass
                
        # If all else fails, just return the text itself
        return text.strip()

    async def generate_random_bias(self, llm_goal: str = "coding"):
        generator_prompt = f"""
Your job is to generate a system prompt for an LLM specialized in {llm_goal}.
The system prompt should influence the LLM's style, tone, and approach.

IMPORTANT: Your response MUST follow this format exactly:
```
[Your system prompt text here]
```

Make sure the system prompt:
1. Is focused on producing exact, correct outputs
2. Emphasizes attention to detail and precise formatting
3. Encourages careful reading of problem specifications
4. Discourages fluff or unnecessary text in outputs

For a coding LLM, ensure the system prompt:
- Emphasizes writing clean, efficient code
- Focuses on exact input/output formats
- Encourages proper error handling
- Prioritizes simplicity and readability
"""
        user_prompt = f"Generate a system prompt for an LLM focused on {llm_goal}."
        bias = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.2,
            seed=int(time.time())
        )
        bias = bias.choices[0].message.content
        return self._extract_content(bias)


### WEIGHTS

def weight_to_text(weight):
    if weight >= 0 and weight < 0.2:
        return "very low"
    elif weight >= 0.2 and weight < 0.4:
        return "low"
    elif weight >= 0.4 and weight < 0.6:
        return "medium"
    elif weight >= 0.6 and weight < 0.8:
        return "high"
    elif weight >= 0.8 and weight < 1:
        return "very high"
    else:
        return "extreme"

def text_to_weight(text):
    if text == "very low":
        return 0
    elif text == "low":
        return 0.2
    elif text == "medium":
        return 0.4
    elif text == "high":
        return 0.6
    elif text == "very high":
        return 0.8
    elif text == "extreme":
        return 1
    else:
        return 0

### VALIDATION

def validate_request_structure(data: dict) -> tuple[bool, str]:
    """Validate the basic structure of the initialization request"""
        
    # Check user_init presence and type
    if "user_init" not in data:
        return False, "Missing required 'user_init' flag"
    if not isinstance(data["user_init"], bool):
        return False, "user_init must be a boolean"
        
    # Check model_goal presence and type
    if "model_goal" not in data:
        return False, "Missing required 'model_goal' field"
    if not isinstance(data["model_goal"], str):
        return False, "model_goal must be a string"
        
    # Check for mutually exclusive fields
    if data["user_init"]:
        if "layer_sizes" in data:
            return False, "layer_sizes should not be provided when user_init is true"
        if "parameters" not in data:
            return False, "parameters field required when user_init is true"
    else:
        if "parameters" in data:
            return False, "parameters should not be provided when user_init is false"
        if "layer_sizes" not in data:
            return False, "layer_sizes field required when user_init is false"
            
    return True, ""

def validate_layer_sizes(layer_sizes) -> tuple[bool, str]:
    """Validate layer sizes for random initialization"""
    if not isinstance(layer_sizes, list):
        return False, "layer_sizes must be a list"
    
    if len(layer_sizes) == 0:
        return False, "layer_sizes cannot be empty"
        
    if not all(isinstance(size, int) for size in layer_sizes):
        return False, "all layer sizes must be integers"
        
    if not all(size > 0 for size in layer_sizes):
        return False, "all layer sizes must be positive"
        
    return True, ""

def validate_neuron(neuron: dict, layer_idx: int, neuron_idx: int) -> tuple[bool, str]:
    """Validate a single neuron's parameters"""
        
    required_neuron_keys = {"w", "system_prompt"}
    if not all(key in neuron for key in required_neuron_keys):
        return False, f"Neuron {neuron_idx} in layer {layer_idx} missing required keys. Required: {required_neuron_keys}"
        
    if not isinstance(neuron["w"], list):
        return False, f"Weights (w) in neuron {neuron_idx}, layer {layer_idx} must be a list"
        
    if not all(isinstance(w, (int, float)) for w in neuron["w"]):
        return False, f"All weights in neuron {neuron_idx}, layer {layer_idx} must be numbers"
        
    if not isinstance(neuron["system_prompt"], str):
        return False, f"system_prompt in neuron {neuron_idx}, layer {layer_idx} must be a string"
        
    return True, ""

def validate_layer(layer: dict, layer_idx: int) -> tuple[bool, str]:
    """Validate a single layer's parameters"""
        
    if "neurons" not in layer:
        return False, f"Layer {layer_idx} missing 'neurons' key"
        
    if not isinstance(layer["neurons"], list):
        return False, f"Layer {layer_idx} neurons must be a list"
    
    for neuron_idx, neuron in enumerate(layer["neurons"]):
        is_valid, error_message = validate_neuron(neuron, layer_idx, neuron_idx)
        if not is_valid:
            return False, error_message
            
    return True, ""

def validate_parameters(params: dict) -> tuple[bool, str]:
    """Validate the complete parameters structure for user initialization"""
    try:
        
        required_top_keys = {"model_goal", "layers"}
        if not all(key in params for key in required_top_keys):
            return False, f"Missing required top-level keys. Required: {required_top_keys}"
        
        # check for additional keys
        allowed_additional_keys = {"model_goal", "layers"}
        additional_keys = set(params.keys()) - allowed_additional_keys
        if additional_keys:
            return False, f"Additional keys are not allowed: {additional_keys}"
        
        if not isinstance(params["model_goal"], str):
            return False, "model_goal must be a string"
        
        if not isinstance(params["layers"], list):
            return False, "layers must be a list"
            
        for layer_idx, layer in enumerate(params["layers"]):
            is_valid, error_message = validate_layer(layer, layer_idx)
            if not is_valid:
                return False, error_message
                    
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
