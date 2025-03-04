import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class LLMModel:
    """
    A flexible class for loading and generating text using different LLMs.
    The user has full control over the model initialization and generation parameters.

    Attributes:
        model_name (str): Name of the pretrained model.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForCausalLM): The loaded LLM model.
        generation_config (GenerationConfig): Default generation configuration.
    """

    def __init__(self,
                 model_name: str,
                 torch_dtype: Optional[torch.dtype] = torch.bfloat16,
                 device_map: str = "auto",
                 attn_implementation: str = "flash_attention_2",
                 quantization_config: Optional[Dict[str, Any]] = None,
                 additional_params: Optional[Dict[str, Any]] = None):
        """
        A flexible class for loading and generating text using different LLMs.
        The user has full control over the model initialization and generation parameters.
        Initializes the model with user-defined parameters.

        Args:
            model_name (str): Name or path of the model.
            torch_dtype (Optional[torch.dtype]): The precision type (default: bfloat16).
            device_map (str): Device placement strategy (default: "auto").
            attn_implementation (str): Attention mechanism implementation.
            quantization_config (Optional[Dict[str, Any]]): Quantization settings.
            additional_params (Optional[Dict[str, Any]]): Extra parameters for loading the model.
        """
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with user-defined parameters
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "attn_implementation": attn_implementation
        }

        if quantization_config:
            model_kwargs.update(quantization_config)

        if additional_params:
            model_kwargs.update(additional_params)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()

        # Load default generation config
        self.generation_config = GenerationConfig.from_pretrained(model_name)

    def generate(self,
                 prompt: str,
                 system_prompt: str = "",
                 custom_config: Optional[Dict[str, Any]] = None,
                 tokenizer_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates text using the model with a given prompt and configuration.

        Args:
            prompt (str): The user input prompt.
            system_prompt (str): The system prompt defining the model's role.
            custom_config (Optional[Dict[str, Any]]): Custom generation parameters.
            tokenizer_params (Optional[Dict[str, Any]]): Custom tokenization parameters.

        Returns:
            str: The generated text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Default tokenizer parameters
        tokenizer_params_default = {
            'truncation': True,
            'add_generation_prompt': True,
            'return_tensors': "pt"
        }

        # Update default tokenizer parameters with user-defined ones
        if tokenizer_params:
            tokenizer_params_default.update(tokenizer_params)

        # Apply tokenizer with the final parameters
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                       **tokenizer_params_default
                                                       ).to(next(self.model.parameters()).device)

        # Use either default config or user-defined config for generation
        generation_params = self.generation_config.to_dict()
        if custom_config:
            generation_params.update(custom_config)

        # Generate output
        output = self.model.generate(input_ids, **generation_params)

        # Decode and return the generated text
        return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
