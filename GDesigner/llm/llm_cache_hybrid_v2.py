"""
Hybrid Cache LLM V2: Agent-type-aware cache generation
Follows LatentMAS pattern: intermediate agents only generate cache, final agent generates text
"""
import torch
from typing import List, Dict, Optional, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hybrid_cache_model_v2 import HybridCacheModel
from GDesigner.llm.llm_registry import LLMRegistry


@LLMRegistry.register('hybrid_cache_v2')
class HybridCacheLLMV2:
    """
    V2: Agent-type-aware cache generation (LatentMAS-style)
    - Intermediate agents: Only generate_latent_batch() → cache only
    - Final/judger agents: generate_latent_batch() + generate_text_batch() → cache + text
    """
    def __init__(self, 
                 cache_model_name: str = "Qwen/Qwen3-4B",
                 api_model_name: str = "qwen-flash",
                 device: str = "cuda:0",
                 device_map: str = "auto"):
        self.model_name = api_model_name
        
        # Initialize hybrid model
        self.hybrid_model = HybridCacheModel(
            cache_model_name=cache_model_name,
            api_model_name=api_model_name,
            device=device,
            device_map=device_map
        )
        self.tokenizer = self.hybrid_model.tokenizer
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        """Convert messages to text prompt"""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"
        text += "<|assistant|>\n"
        return text
    
    async def agen(self, messages: List[Dict], **kwargs) -> str:
        """Standard generation (API only, no cache)"""
        text, _ = await self.hybrid_model.generate_text_batch_api(
            messages,
            past_key_values=None,
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        return text[0]
    
    async def agen_with_cache(
        self,
        messages: List[Dict],
        past_key_values: Optional[Tuple] = None,
        latent_steps: int = 10,
        agent_type: str = "intermediate",  # NEW: "intermediate" or "judger"
        generation_mode: str = "api_hint",
        **kwargs
    ) -> Tuple[str, Tuple]:
        """
        Generate with cache (agent-type-aware, LatentMAS-style)
        
        Args:
            agent_type: "intermediate" (latent cache only) or "judger" (text only, uses cache)
            generation_mode: "api_hint", "hybrid", "local" (only used for judger)
        
        Returns:
            (text_response, kv_cache)
        """
        print(f"\n📦 [STEP 7] HybridCacheLLMV2.agen_with_cache() - Agent type: {agent_type}")
        
        # Tokenize prompt
        prompt = self._messages_to_text(messages)
        max_length = getattr(self.tokenizer, 'model_max_length', 2048)
        if max_length > 100000:
            max_length = 2048
        
        encoded = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = encoded["input_ids"].to(self.hybrid_model.device)
        attention_mask = encoded["attention_mask"].to(self.hybrid_model.device)
        
        print(f"   ✅ Tokenized to {input_ids.shape[1]} tokens")
        
        # Branch based on agent type (LatentMAS pattern)
        if agent_type == "intermediate":
            # Intermediate: Only generate latent cache (NO text generation)
            print(f"\n🔗 [STEP 8] Intermediate agent - Calling generate_latent_batch() only")
            cache_kv, last_hidden = self.hybrid_model.generate_latent_batch(
                input_ids,
                attention_mask,
                latent_steps=latent_steps,
                past_key_values=past_key_values,
            )
            print(f"   ✅ Generated cache: {len(cache_kv)} layers, seq_len={cache_kv[0][0].shape[2]}")
            print(f"   ⏭️  Skipping text generation (intermediate agent)")
            return "", cache_kv
        
        else:  # judger
            # Judger: Only generate text (NO latent cache generation, uses existing cache)
            print(f"\n📝 [STEP 8] Judger agent - Calling generate_text_batch() only (no latent generation)")
            print(f"   🔗 Using accumulated cache from predecessors: {len(past_key_values)} layers" if past_key_values else "   🆕 No cache from predecessors")
            
            if generation_mode == "hybrid":
                text, cache_kv = await self.hybrid_model.generate_text_batch_hybrid(
                    input_ids,
                    messages,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,  # Use input cache directly
                    init_hidden=None,  # No init_hidden since we didn't call generate_latent_batch
                    max_tokens=kwargs.get("max_tokens", 4096)
                )
            elif generation_mode == "local":
                text, cache_kv = self.hybrid_model.generate_text_batch(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,  # Use input cache directly
                    max_new_tokens=kwargs.get("max_tokens", 1024),
                    init_hidden=None  # No init_hidden
                )
            else:  # api_hint
                text, _ = await self.hybrid_model.generate_text_batch_api(
                    messages,
                    past_key_values=past_key_values,  # Use input cache directly
                    max_tokens=kwargs.get("max_tokens", 1024)
                )
                cache_kv = past_key_values  # Return input cache unchanged
            
            print(f"   ✅ Generated text: {len(text[0])} characters")
            return text[0], cache_kv
