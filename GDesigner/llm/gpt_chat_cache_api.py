"""
Cache-enabled LLM using API (simulated cache for testing)
Uses your free Qwen API but simulates cache communication
"""
import aiohttp
import torch
from typing import List, Union, Optional, Tuple, Dict
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


@LLMRegistry.register('GPTChatCacheAPI')
class GPTChatCacheAPI(LLM):
    """
    Cache-enabled LLM using API
    Uses Qwen API for text generation + simulates cache for testing
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.hidden_dim = 4096  # Qwen hidden dimension
        self.num_layers = 32    # Qwen number of layers
    
    def _simulate_cache(self, text: str, past_cache: Optional[Tuple] = None) -> Tuple:
        """
        Simulate KV-cache based on text length
        For testing structure without GPU
        """
        seq_len = len(text.split())
        
        # If we have past cache, extend it
        if past_cache is not None:
            # Extend existing cache
            extended_cache = []
            for layer_cache in past_cache:
                k, v = layer_cache
                # Add new tokens to cache
                new_k = torch.randn(1, seq_len, self.hidden_dim)
                new_v = torch.randn(1, seq_len, self.hidden_dim)
                extended_k = torch.cat([k, new_k], dim=1)
                extended_v = torch.cat([v, new_v], dim=1)
                extended_cache.append((extended_k, extended_v))
            return tuple(extended_cache)
        else:
            # Create new cache
            cache = []
            for _ in range(self.num_layers):
                k = torch.randn(1, seq_len, self.hidden_dim)
                v = torch.randn(1, seq_len, self.hidden_dim)
                cache.append((k, v))
            return tuple(cache)
    
    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def _api_call(self, messages: List[Dict]) -> str:
        """Call Qwen API"""
        request_url = f"{MINE_BASE_URL}/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {MINE_API_KEYS}'
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        
        print(f"\n🌐 [API CALL] Calling Qwen API...")
        print(f"   Model: {self.model_name}")
        print(f"   URL: {request_url}")
        print(f"   Messages: {len(messages)} messages")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=data) as response:
                print(f"   Status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"   ❌ ERROR: {error_text}")
                    response.raise_for_status()
                
                response_data = await response.json()
                text = response_data['choices'][0]['message']['content']
                print(f"   ✅ SUCCESS: Received {len(text)} characters")
                return text
    
    async def agen_with_cache(self, messages: List[Message], 
                              past_key_values: Optional[Tuple] = None,
                              latent_steps: int = 10) -> Tuple[str, Tuple]:
        """
        Generate text with simulated cache
        Uses API for text, simulates cache for structure testing
        """
        print(f"\n📦 [CACHE] agen_with_cache called")
        
        # Convert messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Add cache info to prompt if available (text-based cache simulation)
        if past_key_values is not None:
            print(f"   🔄 Using cached context from {len(past_key_values)} layers")
            # Add a hint that we're using cached context
            cache_hint = "\n[Using cached context from previous agents]"
            messages[-1]["content"] += cache_hint
        else:
            print(f"   🆕 No previous cache, starting fresh")
        
        # Call API for text generation
        text = await self._api_call(messages)
        
        # Simulate cache generation
        print(f"   🧪 Generating simulated cache...")
        kv_cache = self._simulate_cache(text, past_key_values)
        print(f"   ✅ Cache generated: {len(kv_cache)} layers")
        
        return text, kv_cache
    
    async def agen(self, messages: List[Message], **kwargs) -> str:
        """Standard generation without cache (for compatibility)"""
        text, _ = await self.agen_with_cache(messages, **kwargs)
        return text
