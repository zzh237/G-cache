"""
Latent Cache Wrapper: Integrates LatentMAS cache generation into G-cache
Copies core functions from LatentMAS/models.py for KV-cache manipulation
"""
import torch
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer


def _past_length(past_key_values: Optional[Tuple]) -> int:
    """Get length of KV-cache"""
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class LatentCacheModel:
    """
    Minimal wrapper for LatentMAS-style cache generation
    Core functions: generate_latent_batch + generate_text_batch
    """
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(device).eval()
        
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
    
    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        latent_steps: int = 10,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """
        Generate KV-cache with latent reasoning (from LatentMAS)
        Returns ONLY cache, no text generation
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq_len]")
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        
        # Extend attention mask if using past cache
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        
        # Step 1: Forward pass with prompt
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        
        # Step 2: Generate latent_steps tokens (NO text output!)
        for step in range(latent_steps):
            # Use hidden state as embedding directly (simplified - no realignment)
            latent_embed = last_hidden.unsqueeze(1)  # [B, 1, D]
            
            # Extend attention mask
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            
            # Forward pass with latent embedding
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        return past  # Return KV-cache ONLY
    
    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        """
        Generate text from KV-cache (from LatentMAS)
        Converts cache → text tokens
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq_len]")
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        
        # Extend attention mask if using past cache
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        
        # Generate text with cache
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        
        return generations, outputs.past_key_values
