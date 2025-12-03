"""
Cache-enabled LLM wrapper - Integrates LatentMAS cache generation with GDesigner
"""
import torch
from typing import List, Union, Optional, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from GDesigner.llm.format import Message
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

try:
    from vllm import LLM as vLLM_Engine, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _past_length(past_key_values):
    if not past_key_values:
        return 0
    return past_key_values[0][0].shape[-2]


@LLMRegistry.register('GPTChatCache')
class GPTChatCache(LLM):
    """LLM with cache extraction and injection capabilities"""
    
    def __init__(self, model_name: str, device: str = "cuda", use_vllm: bool = True):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        
        if self.use_vllm:
            # vLLM for fast generation
            self.vllm_engine = vLLM_Engine(
                model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                enable_prefix_caching=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # HuggingFace model for cache generation
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device
            ).eval()
        else:
            # Fallback to HuggingFace only
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device
            ).eval()
    
    def _apply_latent_realignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """Realign hidden states to input embedding space (from LatentMAS)"""
        # Simplified version - just normalize
        norm = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_norm = self.hf_model.get_input_embeddings().weight.norm(dim=1).mean()
        return hidden * (target_norm / norm)
    
    @torch.no_grad()
    def generate_latent_cache(self, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor,
                             latent_steps: int = 10,
                             past_key_values: Optional[Tuple] = None) -> Tuple:
        """Generate KV-cache with latent steps (from LatentMAS)"""
        
        # Initial forward pass
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        # Generate latent steps
        for _ in range(latent_steps):
            # Realign to embedding space
            latent_vec = self._apply_latent_realignment(last_hidden)
            latent_embed = latent_vec.unsqueeze(1)
            
            # Forward pass with latent embedding
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device
            )
            
            outputs = self.hf_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        return past
    
    async def agen_with_cache(self, messages: List[Message], 
                              past_key_values: Optional[Tuple] = None,
                              latent_steps: int = 10) -> Tuple[str, Tuple]:
        """Generate text with cache extraction"""
        
        # Prepare input
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Generate cache with latent steps
        kv_cache = self.generate_latent_cache(
            input_ids, 
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values
        )
        
        # Generate text
        if self.use_vllm:
            sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
            outputs = self.vllm_engine.generate([prompt], sampling_params)
            text = outputs[0].outputs[0].text.strip()
        else:
            # Use HF model for generation
            gen_outputs = self.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                past_key_values=kv_cache,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            text = self.tokenizer.decode(gen_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return text, kv_cache
    
    async def agen(self, messages: List[Message], **kwargs) -> str:
        """Standard generation without cache (for compatibility)"""
        text, _ = await self.agen_with_cache(messages, **kwargs)
        return text
