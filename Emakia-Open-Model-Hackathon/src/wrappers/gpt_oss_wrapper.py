"""
GPT-OSS model wrapper for the Emakia Validator Agent.

This module provides a wrapper for open-source GPT models, implementing the base wrapper interface.
"""

import asyncio
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from loguru import logger

from .base_wrapper import BaseModelWrapper


class GPTOSSWrapper(BaseModelWrapper):
    """
    Wrapper for open-source GPT models.
    
    This class provides a unified interface for interacting with open-source
    GPT models using the Hugging Face transformers library.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GPT-OSS wrapper.
        
        Args:
            config: Configuration dictionary for GPT-OSS
        """
        super().__init__(config)
        
        # Initialize model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"GPT-OSS wrapper initialized with device: {self.device}")
    
    def _load_model(self):
        """
        Load the model and tokenizer.
        """
        try:
            # Map model names to Hugging Face model IDs
            model_mapping = {
                "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
                "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf",
                "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
                "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
                "qwen-7b": "Qwen/Qwen-7B-Chat",
                "qwen-14b": "Qwen/Qwen-14B-Chat",
                "qwen-72b": "Qwen/Qwen-72B-Chat"
            }
            
            model_id = model_mapping.get(self.model_name, self.model_name)
            
            logger.info(f"Loading model: {model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the local GPT model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response with metadata
        """
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get generation parameters
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Calculate token usage
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = response_tokens.shape[0]
            total_tokens = input_tokens + output_tokens
            
            return {
                'content': response_text.strip(),
                'model': self.model_name,
                'usage': {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': total_tokens
                },
                'finish_reason': 'stop'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def classify(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Classify text into predefined categories using the local GPT model.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            
        Returns:
            Classification results
        """
        # Create classification prompt
        categories_str = ", ".join(categories)
        prompt = f"""
        Classify the following text into one of these categories: {categories_str}
        
        Text: {text}
        
        Please respond with a JSON object containing:
        1. "category": the most appropriate category
        2. "confidence": confidence score between 0 and 1
        3. "reasoning": brief explanation for the classification
        
        Response:
        """
        
        try:
            response = await self.generate(prompt, temperature=0.1)
            
            # Parse response (assuming JSON format)
            import json
            try:
                result = json.loads(response['content'])
                return {
                    'category': result.get('category'),
                    'confidence': result.get('confidence', 0.0),
                    'reasoning': result.get('reasoning', ''),
                    'all_categories': categories,
                    'model': self.model_name
                }
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return {
                    'category': 'unknown',
                    'confidence': 0.0,
                    'reasoning': 'Failed to parse model response',
                    'all_categories': categories,
                    'model': self.model_name,
                    'raw_response': response['content']
                }
                
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            raise
    
    async def validate(self, content: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content against specified rules using the local GPT model.
        
        Args:
            content: Content to validate
            validation_rules: Rules to validate against
            
        Returns:
            Validation results
        """
        # Create validation prompt
        rules_str = "\n".join([f"- {rule}: {desc}" for rule, desc in validation_rules.items()])
        prompt = f"""
        Validate the following content against these rules:
        
        {rules_str}
        
        Content: {content}
        
        Please respond with a JSON object containing:
        1. "is_valid": boolean indicating if content passes all rules
        2. "violations": list of rule violations found
        3. "confidence": confidence score between 0 and 1
        4. "suggestions": suggestions for improvement
        
        Response:
        """
        
        try:
            response = await self.generate(prompt, temperature=0.1)
            
            # Parse response
            import json
            try:
                result = json.loads(response['content'])
                return {
                    'is_valid': result.get('is_valid', False),
                    'violations': result.get('violations', []),
                    'confidence': result.get('confidence', 0.0),
                    'suggestions': result.get('suggestions', []),
                    'model': self.model_name,
                    'rules_checked': list(validation_rules.keys())
                }
            except json.JSONDecodeError:
                return {
                    'is_valid': False,
                    'violations': ['Failed to parse validation response'],
                    'confidence': 0.0,
                    'suggestions': ['Check model response format'],
                    'model': self.model_name,
                    'rules_checked': list(validation_rules.keys()),
                    'raw_response': response['content']
                }
                
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on GPT-OSS wrapper.
        
        Returns:
            Health status information
        """
        base_health = super().health_check()
        
        # Add GPT-OSS-specific health info
        base_health.update({
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'device': self.device,
            'cuda_available': torch.cuda.is_available()
        })
        
        return base_health
