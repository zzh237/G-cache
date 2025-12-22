"""
Cache-enabled Math Solver Agent V2 - Agent-type-aware cache generation
Follows LatentMAS: intermediate agents generate cache only, judger generates text
"""
from typing import List, Any, Dict, Optional, Tuple
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.coding.python_executor import execute_code_get_return


@AgentRegistry.register('MathSolverCacheV2')
class MathSolverCacheV2(Node):
    """
    V2: Agent-type-aware cache generation
    - Intermediate agents: Only generate cache (no text output)
    - Judger agents: Generate cache + text output
    """
    
    def __init__(self, id: str = None, role: str = None, domain: str = "", llm_name: str = "",
                 agent_type: str = "intermediate",  # NEW: "intermediate" or "judger"
                 cache_mode: str = "hybrid", 
                 generation_mode: str = "api_hint", 
                 max_new_tokens: int = 512,
                 latent_only: bool = False,  # NEW: Keep only latent tokens
                 latent_steps: int = 10):  # NEW: Number of latent steps
        """
        Args:
            agent_type: "intermediate" (cache only) or "judger" (cache + text)
            cache_mode: "hybrid" (text+cache), "latent_only" (cache only), "text_only" (no cache)
            generation_mode: "api_hint", "hybrid", "local" (only used for judger)
            max_new_tokens: Maximum tokens to generate (only for judger)
            latent_only: If True, only keep latent tokens (discard input tokens)
            latent_steps: Number of latent reasoning steps
        """
        super().__init__(id, "MathSolverCacheV2", domain, llm_name)
        
        # Use cache-enabled LLM V2
        self.llm = LLMRegistry.get(llm_name, use_cache=True)
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        
        # Agent type (NEW!)
        self.agent_type = agent_type  # "intermediate" or "judger"
        
        # Cache communication mode
        self.cache_mode = cache_mode
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.latent_only = latent_only
        self.latent_steps = latent_steps
    
    @staticmethod
    def _slice_tensor(tensor, tokens_to_keep: int):
        """Slice tensor to keep only last N tokens (LatentMAS-style)"""
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()
    
    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        """Truncate past_key_values to keep only last N tokens (LatentMAS-style)"""
        if past_kv is None or tokens_to_keep <= 0:
            return None
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)
    
    def _process_inputs(self, raw_inputs: Dict[str, str], 
                       spatial_info: Dict[str, Dict], 
                       temporal_info: Dict[str, Dict],
                       has_cache: bool = False) -> tuple:
        """Process inputs with cache-aware prompting"""
        system_prompt = self.constraint
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        
        # For intermediate agents in cache mode, minimal text (rely on cache)
        if self.agent_type == "intermediate" and self.cache_mode == "hybrid":
            print(f"   📊 [INTERMEDIATE AGENT] Minimal text prompt (relying on cache)")
            # No text context needed - cache carries the information
        
        # For judger agents, may want more context
        elif self.agent_type == "judger" and self.cache_mode != "text_only":
            print(f"   📊 [JUDGER AGENT] Full prompt with cache support")
            # Judger can use both cache and text context if needed
        
        return system_prompt, user_prompt
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict, temporal_info: Dict):
        """Sync execution (not used in async mode)"""
        raise NotImplementedError("Use async_execute instead")
    
    async def _async_execute(self, input: Dict[str, str], 
                            spatial_info: Dict[str, Dict], 
                            temporal_info: Dict[str, Dict]) -> str:
        """
        Execute with agent-type-aware cache generation
        
        Flow:
        - Intermediate agents: generate_latent_batch() only → cache
        - Judger agents: generate_latent_batch() + generate_text_batch() → cache + text
        """
        graph = getattr(self, 'graph', None)
        print(f"\n🎯 [STEP 3] MathSolverCacheV2._async_execute()")
        print(f"   Agent ID: {self.id}")
        print(f"   Agent Type: {self.agent_type}")
        print(f"   🎭 Role: {self.role}")
        print(f"   📋 Constraint (first 80 chars): {self.constraint[:80]}...")
        
        # Step 1: Get fused cache from predecessors
        past_kv = None
        has_cache = False
        if graph and hasattr(graph, 'get_fused_cache') and self.cache_mode != "text_only":
            print(f"\n🎯 [STEP 4] Calling graph.get_fused_cache() for node {self.id}")
            past_kv = graph.get_fused_cache(self)
            has_cache = past_kv is not None
            print(f"\n📥 [STEP 5] Received fused cache: {has_cache}")
            if has_cache:
                print(f"   📊 Cache: {len(past_kv)} layers, seq_len={past_kv[0][0].shape[2]}")
        
        # Step 2: Build prompt
        print(f"\n📝 [STEP 5a] Building prompt")
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, has_cache)
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        
        # Step 3: Generate with agent-type-aware cache
        if hasattr(self.llm, 'agen_with_cache') and self.cache_mode != "text_only":
            print(f"\n🧠 [STEP 6] Calling llm.agen_with_cache() - Agent type: {self.agent_type}")
            
            response, kv_cache = await self.llm.agen_with_cache(
                messages, 
                past_key_values=past_kv,
                latent_steps=self.latent_steps,
                agent_type=self.agent_type,  # NEW: Pass agent type!
                generation_mode=self.generation_mode,
                max_tokens=self.max_new_tokens,
                agent_id=self.id,  # NEW: Pass agent ID for logging
            )
            
            # Step 4: Truncate cache if latent_only mode (LatentMAS-style)
            if self.latent_only and kv_cache is not None:
                print(f"\n✂️  [INTERMEDIATE] Truncating cache to keep only {self.latent_steps} latent tokens")
                original_len = kv_cache[0][0].shape[2]
                kv_cache = self._truncate_past(kv_cache, self.latent_steps)
                new_len = kv_cache[0][0].shape[2] if kv_cache else 0
                print(f"   📊 [INTERMEDIATE] Cache truncated: {original_len} → {new_len} tokens")
            
            # Step 5: Store cache for successors
            if graph and hasattr(graph, 'store_node_cache'):
                graph.store_node_cache(self.id, kv_cache)
                print(f"\n💾 [STEP 10] Stored cache for node {self.id}")
            
            # For intermediate agents, response is empty - return empty string
            if self.agent_type == "intermediate":
                print(f"\n⏭️  [INTERMEDIATE] No text output - cache is the real output")
                return ""  # Empty output - cache communication is via graph.node_caches, not text
            else:
                # Execute code if Programming Expert role
                if self.role == "Programming Expert" and response.strip():
                    try:
                        print(f"\n🐍 [CODE EXECUTION] Executing Python code for Programming Expert...")
                        code = response.lstrip("```python\n").rstrip("\n```")
                        answer = execute_code_get_return(code)
                        response += f"\nthe answer is {answer}"
                        print(f"   ✅ Code executed successfully, answer: {answer}")
                    except Exception as e:
                        print(f"   ⚠️ Code execution failed: {e}")
                        # Keep original response if execution fails
                
                print(f"\n✅ [JUDGER] Returning text output: {len(response)} chars")
                return response
        else:
            # Fallback: text-only
            response = await self.llm.agen(messages)
            
            # Execute code if Programming Expert role
            if self.role == "Programming Expert" and response.strip():
                try:
                    print(f"\n🐍 [CODE EXECUTION] Executing Python code for Programming Expert...")
                    code = response.lstrip("```python\n").rstrip("\n```")
                    answer = execute_code_get_return(code)
                    response += f"\nthe answer is {answer}"
                    print(f"   ✅ Code executed successfully, answer: {answer}")
                except Exception as e:
                    print(f"   ⚠️ Code execution failed: {e}")
            
            return response
    
    @property
    def verbose(self):
        """Check if verbose mode is enabled"""
        return getattr(self, '_verbose', False)
