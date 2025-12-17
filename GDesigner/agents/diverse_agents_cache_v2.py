"""
Diverse Cache-enabled Agents V2 - Following LatentMAS hierarchical pattern
Creates 4 different specialized agents: Math, Science, Code, Inspector
"""
from typing import List, Any, Dict, Optional, Tuple
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


class DiverseAgentCacheV2(Node):
    """
    Base class for diverse cache-enabled agents
    Each subclass represents a different specialized agent
    """
    
    def __init__(self, id: str = None, agent_class: str = "DiverseAgentCacheV2", 
                 domain: str = "", llm_name: str = "",
                 role: str = "Math Solver",  # Specific role for this agent
                 agent_type: str = "intermediate",
                 generation_mode: str = "api_hint", 
                 max_new_tokens: int = 512,
                 latent_only: bool = False,
                 latent_steps: int = 10):
        super().__init__(id, agent_class, domain, llm_name)
        
        self.llm = LLMRegistry.get(llm_name, use_cache=True)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = role  # Specific role (Math Solver, Mathematical Analyst, etc.)
        self.constraint = self.prompt_set.get_constraint(self.role)
        
        self.agent_type = agent_type
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.latent_only = latent_only
        self.latent_steps = latent_steps
    
    @staticmethod
    def _slice_tensor(tensor, tokens_to_keep: int):
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()
    
    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
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
        system_prompt = self.constraint
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        return system_prompt, user_prompt
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict, temporal_info: Dict):
        raise NotImplementedError("Use async_execute instead")
    
    async def _async_execute(self, input: Dict[str, str], 
                            spatial_info: Dict[str, Dict], 
                            temporal_info: Dict[str, Dict]) -> str:
        graph = getattr(self, 'graph', None)
        print(f"\n🎯 [{self.role}] Agent {self.id} executing (Type: {self.agent_type})")
        
        # Get fused cache from predecessors
        past_kv = None
        has_cache = False
        if graph and hasattr(graph, 'get_fused_cache'):
            past_kv = graph.get_fused_cache(self)
            has_cache = past_kv is not None
            if has_cache:
                print(f"   📥 Received cache: {len(past_kv)} layers, seq_len={past_kv[0][0].shape[2]}")
        
        # Build prompt
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, has_cache)
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        
        # Generate with cache
        if hasattr(self.llm, 'agen_with_cache'):
            print(f"   🧠 Generating with {self.role} perspective...")
            
            response, kv_cache = await self.llm.agen_with_cache(
                messages, 
                past_key_values=past_kv,
                latent_steps=self.latent_steps,
                agent_type=self.agent_type,
                generation_mode=self.generation_mode,
                max_tokens=self.max_new_tokens,
                agent_id=self.id,
            )
            
            # Truncate cache if latent_only
            if self.latent_only and kv_cache is not None:
                original_len = kv_cache[0][0].shape[2]
                kv_cache = self._truncate_past(kv_cache, self.latent_steps)
                new_len = kv_cache[0][0].shape[2] if kv_cache else 0
                print(f"   ✂️  Cache truncated: {original_len} → {new_len} tokens")
            
            # Store cache
            if graph and hasattr(graph, 'store_node_cache'):
                graph.store_node_cache(self.id, kv_cache)
            
            if self.agent_type == "intermediate":
                print(f"   ⏭️  [{self.role}] Cache generated, no text output")
                return ""
            else:
                print(f"   ✅ [{self.role}] Generated {len(response)} chars")
                return response
        else:
            response = await self.llm.agen(messages)
            return response


@AgentRegistry.register('MathAgentCacheV2')
class MathAgentCacheV2(DiverseAgentCacheV2):
    """Math Solver Agent - Solves problems using mathematical reasoning"""
    def __init__(self, id: str = None, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(id, "MathAgentCacheV2", domain, llm_name, 
                        role="Math Solver", **kwargs)


@AgentRegistry.register('AnalystAgentCacheV2')
class AnalystAgentCacheV2(DiverseAgentCacheV2):
    """Mathematical Analyst Agent - Analyzes problems step-by-step"""
    def __init__(self, id: str = None, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(id, "AnalystAgentCacheV2", domain, llm_name, 
                        role="Mathematical Analyst", **kwargs)


@AgentRegistry.register('CodeAgentCacheV2')
class CodeAgentCacheV2(DiverseAgentCacheV2):
    """Programming Expert Agent - Solves problems using code"""
    def __init__(self, id: str = None, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(id, "CodeAgentCacheV2", domain, llm_name, 
                        role="Programming Expert", **kwargs)


@AgentRegistry.register('InspectorAgentCacheV2')
class InspectorAgentCacheV2(DiverseAgentCacheV2):
    """Inspector Agent - Verifies and validates solutions"""
    def __init__(self, id: str = None, domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(id, "InspectorAgentCacheV2", domain, llm_name, 
                        role="Inspector", **kwargs)
