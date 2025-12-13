"""
Hybrid Cache Model: Small local model for cache + API for text
Solves: API can't return cache, but we don't need expensive model for cache
"""
import torch
import os
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class HybridCacheModel:
    """
    Hybrid approach:
    1. Use SMALL local model (e.g., Qwen2.5-1.5B) for cache generation
    2. Use API (qwen-plus) for final text generation
    
    Why this works:
    - Cache generation doesn't need strong reasoning (just forward pass)
    - Final text generation needs strong reasoning (use API)
    - Cost: Small GPU + free API credits
    """
    def __init__(
        self,
        cache_model_name: str = "Qwen/Qwen3-4B",  # Qwen3: 1.7B/3B/4B/8B/14B/32B available
        api_model_name: str = "qwen-plus",  # API for text
        device: str = "cuda:0",
        use_alignment: bool = True,  # Enable LatentMAS alignment matrix
        device_map: str = None  # NEW: "auto" for multi-GPU, None for single GPU
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_alignment = use_alignment
        self._alignment_matrix = None
        self._target_norm = None
        
        # Small local model for cache generation ONLY
        print(f"Loading model for cache: {cache_model_name}")
        if device_map:
            print(f"Using device_map: {device_map}")
        self.tokenizer = AutoTokenizer.from_pretrained(cache_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load with device_map for multi-GPU support
        if device_map:
            self.cache_model = AutoModelForCausalLM.from_pretrained(
                cache_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
            ).eval()
        else:
            self.cache_model = AutoModelForCausalLM.from_pretrained(
                cache_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device).eval()
        
        # API client for text generation
        # Use API_KEY and BASE_URL from .env
        api_key = os.getenv("API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            raise ValueError("API key not found! Set API_KEY in .env file")
        
        print(f"[API] Using base_url: {base_url}")
        self.api_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.api_model_name = api_model_name
        
        # Build alignment matrix (training-free, computed once)
        if self.use_alignment:
            self._build_alignment_matrix()
    
    def _build_alignment_matrix(self):
        """Build W_a ≈ W_out^(-1) * W_in (LatentMAS models.py:195-220)"""
        with torch.no_grad():
            W_in = self.cache_model.get_input_embeddings().weight.detach().float()
            W_out = self.cache_model.get_output_embeddings().weight.detach().float()
            
            # Solve: W_out * W_a = W_in
            gram = torch.matmul(W_out.T, W_out)
            gram += 1e-5 * torch.eye(gram.shape[0], device=gram.device)
            rhs = torch.matmul(W_out.T, W_in)
            self._alignment_matrix = torch.linalg.solve(gram, rhs).to(self.device)
            self._target_norm = W_in.norm(dim=1).mean().to(self.device)
    
    def _apply_alignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply: e = normalize(h * W_a) (LatentMAS models.py:239-248)"""
        if not self.use_alignment:
            return hidden
        
        aligned = torch.matmul(hidden.float(), self._alignment_matrix)
        norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (self._target_norm / norm)
        return aligned.to(hidden.dtype)
    
    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        latent_steps: int = 10,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """
        Generate KV-cache using SMALL local model
        EXACT implementation from LatentMAS models.py:313-382
        """
        print(f"\n   🧠 [STEP 8a] HybridCacheModel.generate_latent_batch() - Generating cache using small local model")
        print(f"   📊 [LOCAL-MODEL] Input: {input_ids.shape[1]} tokens, Latent steps: {latent_steps}")
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        
        # Convert tuple cache to DynamicCache if needed
        if past_key_values is not None and isinstance(past_key_values, tuple):
            from transformers import DynamicCache
            print(f"   🔄 Converting tuple cache to DynamicCache for model compatibility...")
            dynamic_cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(past_key_values):
                dynamic_cache.update(key, value, layer_idx)
            past_key_values = dynamic_cache
        
        # Handle past_key_values attention mask (LatentMAS lines 320-329)
        print(f"\n   📐 [STEP 8a][DIMENSIONS] BEFORE Step 1 - Initial forward pass:")
        print(f"      • input_ids: {input_ids.shape}")
        print(f"      • attention_mask (before concat): {attention_mask.shape}")
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[0][0].shape[-2] if past_key_values else 0
            print(f"      • INPUT past_key_values: {len(past_key_values)} layers, seq_len={past_len}")
            print(f"        - Layer 0 Key: {past_key_values[0][0].shape}")
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
                print(f"      • attention_mask (after concat): {attention_mask.shape} = {past_len} (past) + {input_ids.shape[1]} (new)")
        else:
            print(f"      • INPUT past_key_values: None")
        
        # Step 1: Initial forward pass (LatentMAS lines 331-338)
        outputs = self.cache_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        
        print(f"\n   📐 [STEP 8a][DIMENSIONS] AFTER Step 1 - Initial forward pass:")
        print(f"      • OUTPUT past_key_values: {len(past)} layers, seq_len={past[0][0].shape[2]}")
        print(f"        - Layer 0 Key: {past[0][0].shape}")
        print(f"        - Layer 0 Value: {past[0][1].shape}")
        print(f"      • last_hidden: {last_hidden.shape} [batch, hidden_dim]")
        if past_key_values is not None:
            input_past_len = past_key_values[0][0].shape[2] if hasattr(past_key_values[0][0], 'shape') else past_key_values.get_seq_length()
            output_past_len = past[0][0].shape[2]
            print(f"      • Cache grew by: {output_past_len - input_past_len} tokens (added {input_ids.shape[1]} input tokens)")
        
        # Step 2: Latent reasoning loop (LatentMAS lines 348-378)
        print(f"\n   🔄 [STEP 8a][STEP 2] Starting latent reasoning loop ({latent_steps} steps)...")
        for step in range(latent_steps):
            past_len_before = past[0][0].shape[2]
            
            # Apply alignment matrix (LatentMAS line 348)
            latent_vec = self._apply_alignment(last_hidden)
            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]
            
            # Calculate past length (LatentMAS lines 361-362)
            past_len = past[0][0].shape[-2] if past else 0
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            
            # Forward pass with embedding (LatentMAS lines 363-370)
            outputs = self.cache_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            past_len_after = past[0][0].shape[2]
            if step == 0 or step == latent_steps - 1:
                print(f"      • Latent step {step+1}/{latent_steps}: cache {past_len_before} → {past_len_after} (+{past_len_after - past_len_before} token)")
        
        final_seq_len = past[0][0].shape[2]
        print(f"\n   📐 [STEP 8a][DIMENSIONS] FINAL - After all latent steps:")
        print(f"      • Final past_key_values: {len(past)} layers, seq_len={final_seq_len}")
        print(f"        - Layer 0 Key: {past[0][0].shape}")
        print(f"        - Layer 0 Value: {past[0][1].shape}")
        print(f"   ✅ [LOCAL-MODEL] Cache generation complete: {len(past)} layers, {final_seq_len} tokens")
        return past  # LatentMAS line 380
    
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
        Generate text using LOCAL model with cache tensors DIRECTLY
        EXACT implementation from LatentMAS models.py:216-265
        
        This is the REAL cache usage - tensors are passed directly to model.generate()
        """
        print(f"\n   🎯 [STEP 9a] HybridCacheModel.generate_text_batch() - Generating text using cache TENSORS directly")
        
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        
        # Handle past_key_values and cache_position
        cache_position = None
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[-2]
            print(f"   🔗 [LOCAL-MODEL of 9a] Using key values cache tensors: {len(past_key_values)} layers, {past_len} tokens")
            # Create cache_position for new tokens
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            # Create attention mask for past + current tokens
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        else:
            print(f"   🆕 [LOCAL-MODEL of 9a] No cache")
        
        # Generate with cache tensors (LatentMAS lines 244-253)
        print(f"   ⚙️ [LOCAL-MODEL of 9a] Calling model.generate() with cache tensors...")
        print(f"\n   📐 [DIMENSIONS] Input dimensions for model.generate():")
        print(f"      • input_ids: {input_ids.shape} (batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]})")
        print(f"      • attention_mask: {attention_mask.shape}")
        if past_key_values is not None:
            print(f"      • past_key_values: {len(past_key_values)} layers")
            print(f"        - Layer 0 Key: {past_key_values[0][0].shape} [batch, heads, seq_len, head_dim]")
            print(f"        - Layer 0 Value: {past_key_values[0][1].shape} [batch, heads, seq_len, head_dim]")
            print(f"        - Cached sequence length: {past_key_values[0][0].shape[2]} tokens")
        else:
            print(f"      • past_key_values: None")
        if cache_position is not None:
            print(f"      • cache_position: {cache_position.shape} = {cache_position.tolist()}")
        else:
            print(f"      • cache_position: None")
        print(f"      • max_new_tokens: {max_new_tokens}")
        print(f"      • temperature: {temperature}, top_p: {top_p}")
        
        outputs = self.cache_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,  # ← FIX: Tell model where new tokens go!
        )
        print(f"\n   📐 [DIMENSIONS] Output dimensions from model.generate():")
        print(f"      • sequences: {outputs.sequences.shape} (batch_size={outputs.sequences.shape[0]}, total_seq_len={outputs.sequences.shape[1]})")
        if outputs.past_key_values is not None:
            print(f"      • past_key_values: {len(outputs.past_key_values)} layers")
            print(f"        - Layer 0 Key: {outputs.past_key_values[0][0].shape}")
            print(f"        - Layer 0 Value: {outputs.past_key_values[0][1].shape}")
            print(f"        - Final sequence length: {outputs.past_key_values[0][0].shape[2]} tokens")
        print(f"   ✅ [LOCAL-MODEL of 9a] model.generate() complete")
        
        # Decode generated text (LatentMAS lines 254-260)
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        
        print(f"   ✅ [LOCAL-MODEL] Generated {len(generations[0])} characters using cache tensors")
        print(f"   📝 [LOCAL-MODEL] Generated text preview: {generations[0][:150]}...")
        return generations, outputs.past_key_values
    
    async def generate_text_batch_api(
        self,
        messages: List[Dict],
        past_key_values: Optional[Tuple] = None,
        max_tokens: int = 256,
    ) -> Tuple[List[str], None]:
        """
        Generate text using API with cache context
        
        Args:
            messages: Chat messages
            past_key_values: Real KV-cache from small local model
            max_tokens: Max tokens to generate
        
        Returns:
            (text_list, None) - API doesn't return cache
        """
        print(f"\n   🔄 [STEP 9b] HybridCacheModel.generate_text_batch_api() - Converting cache converted text to text context using API")
        
        # Inject cache context if available
        if past_key_values:
            # Cache is a TUPLE of (key, value) pairs, one per layer
            # Each key/value is a torch.Tensor with shape [batch, heads, seq_len, hidden_dim]
            print(f"   🔍 [DEBUG] Cache structure: past_key_values type: {type(past_key_values)}")
            print(f"      - Type: tuple of {len(past_key_values)} layers")
            print(f"      - Layer 0 type: {type(past_key_values[0])}")
            print(f"      - Layer 0 key shape: {past_key_values[0][0].shape}")
            print(f"      - Layer 0 value shape: {past_key_values[0][1].shape}")
            print(f"      - Data type: {past_key_values[0][0].dtype}")
            print(f"      - Device: {past_key_values[0][0].device}")
            
            seq_len = past_key_values[0][0].shape[2]
            cache_info = f"[Using {len(past_key_values)} layers of cached reasoning from previous agents, {seq_len} tokens]"
            print(f"\n   ⚠️ [IMPORTANT] Cache is a TENSOR object, NOT text!")
            print(f"   ⚠️ [IMPORTANT] API cannot use tensors directly - converting to text hint")
            print(f"   ✅ [API-CACHE] Cache available: {len(past_key_values)} layers, {seq_len} tokens")
            print(f"   📝 [API-CACHE] Injecting cache context as TEXT PREFIX: '{cache_info}'")
            messages = messages.copy()
            if messages and messages[-1].get("role") == "user":
                original_content = messages[-1]["content"]
                messages[-1]["content"] = cache_info + "\n\n" + original_content
                print(f"   📋 [API-CACHE] Modified prompt preview:")
                print(f"      Original: {original_content[:100]}...")
                print(f"      Modified: {messages[-1]['content'][:150]}...")
        else:
            print(f"   ⚠️ [API-CACHE] No cache used - API will be the solo source past_key_values type: {type(past_key_values)}")
        
        # Generate with API
        print(f"   🌐 [API] Calling {self.api_model_name} API (NOT using cache directly, only text context)...")
        response = await self.api_client.chat.completions.create(
            model=self.api_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        text = response.choices[0].message.content
        print(f"   ✅ [API] Received response: {len(text)} characters")
        print(f"   📝 [API] Response preview: {text[:150]}...")
        return [text], None
    
    async def generate_text_batch_hybrid(
        self,
        input_ids: torch.Tensor,
        messages: List[Dict],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        max_tokens: int = 256,
    ) -> Tuple[List[str], Optional[Tuple]]:
        """
        TRUE Hybrid: Combine local model (real cache) + API (high quality)
        
        Flow:
        1. Local model uses cache tensors directly (fast, saves computation)
        2. API refines the output (high quality)
        3. Returns updated cache for next agent
        
        Best of both worlds!
        """
        print(f"\n   ⭐ [HYBRID] Using TRUE hybrid approach...")
        
        # Step 1: Generate with local model using REAL cache
        print(f"   📍 [HYBRID] Local model with real cache tensors, past_key_values, call generate_text_batch()")
        local_text, new_cache = self.generate_text_batch(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,  # ← Real cache usage!
            max_new_tokens=max_tokens,
        )
        
        # Step 2: Use local output as context for API
        print(f"   📍 [HYBRID] Step 2: API refinement with cache converted text as context")
        print(f"   🔍 [HYBRID] Cache converted text preview: {local_text[0][:150]}...")
        messages = messages.copy()
        if messages and messages[-1].get("role") == "user":
            original_user_msg = messages[-1]["content"]
            print(f"   🔍 [HYBRID] Original user message: {original_user_msg[:100]}...")
            context = f"Previous reasoning from local model:\n{local_text[0]}\n\n"
            print(f"   🔍 [HYBRID] cache converted text length: {len(context)} chars, preview: {context[:100]}...")
            messages[-1]["content"] = context + messages[-1]["content"]
            print(f"   🔍 [HYBRID] Modified user message (with cache converted text): {messages[-1]['content'][:150]}...")
        
        # Step 3: Get high-quality output from API
        api_text, _ = await self.generate_text_batch_api(
            messages,
            past_key_values=None,  # Don't send cache to API (already used by local)
            max_tokens=max_tokens,
        )
        
        print(f"   ✅ [HYBRID] Complete! Local cache used + API refined")
        return api_text, new_cache  # Best of both worlds!
