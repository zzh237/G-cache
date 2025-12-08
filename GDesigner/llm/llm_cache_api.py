"""
API-only LLM (no cache - baseline)
Simple wrapper for Qwen API without any cache simulation
"""
import os
from typing import List, Dict, Optional, Tuple, Any
from openai import AsyncOpenAI
from GDesigner.llm.llm_registry import LLMRegistry


@LLMRegistry.register('qwen-plus')
class QwenAPI:
    """
    Pure API baseline - no cache
    Just calls Qwen API for text generation
    """
    def __init__(self, model_name: str = "qwen-plus", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    async def agen(self, messages: List[Dict], **kwargs) -> str:
        """Standard API call"""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return response.choices[0].message.content
    
    async def agen_with_cache(
        self,
        messages: List[Dict],
        past_key_values: Optional[Any] = None,
        latent_steps: int = 10,
        **kwargs
    ) -> Tuple[str, None]:
        """
        API doesn't support cache - just call agen()
        Returns (text, None) for compatibility
        """
        text = await self.agen(messages, **kwargs)
        return text, None  # No cache


@LLMRegistry.register('qwen-turbo')
class QwenTurboAPI(QwenAPI):
    """Alias for qwen-turbo model"""
    def __init__(self, **kwargs):
        super().__init__(model_name="qwen-turbo", **kwargs)
