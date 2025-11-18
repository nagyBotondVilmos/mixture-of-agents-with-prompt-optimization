LLM_API_KEY_MOA = [                            # multiple API keys to prevent rate limiting (must be a list, minimum 1 key)
    "api-key-1",
    "api-key-2",
    "api-key-3"
]

LLM_API_KEY_TRAINER = "api-key-trainer"        # API key for the trainer

LLM_API_KEY_AUX = "api-key-aux"                # API key for the auxillary LLM

LLM_API_URL = "https://api.model-provider.com"  # API URL for the LLM
LLM_MODEL = "model-name"                        # model name
LLM_MAX_TOKENS = 1024                           # max tokens for the LLM

MY_API_URL = "http://localhost:5000"            # API URL for the API server

SIMILARITY_THRESHOLD = 0.95                     # similarity threshold for text comparison (evaluation metric), in range [0, 1]
