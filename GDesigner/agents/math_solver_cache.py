"""
Cache-enabled Math Solver Agent - Actually uses cache communication
"""
from typing import List, Any, Dict
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register('MathSolverCache')
class MathSolverCache(Node):
    """Math solver that uses cache communication"""
    
    def __init__(self, id: str = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "MathSolverCache", domain, llm_name)
        
        # Use cache-enabled LLM
        self.llm = LLMRegistry.get('GPTChatCache', llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
    
    async def _process_inputs(self, raw_inputs: Dict[str, str], 
                             spatial_info: Dict[str, Dict], 
                             temporal_info: Dict[str, Dict]) -> tuple:
        """Process inputs into system and user prompts"""
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        
        # Add spatial info (from other agents in same round)
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} ({info['role']}): {info['output']}\n\n"
        
        if spatial_str:
            user_prompt += f"\nOther agents' outputs:\n{spatial_str}"
        
        # Add temporal info (from previous rounds)
        temporal_str = ""
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id} ({info['role']}): {info['output']}\n\n"
        
        if temporal_str:
            user_prompt += f"\nPrevious round outputs:\n{temporal_str}"
        
        return system_prompt, user_prompt
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict, temporal_info: Dict):
        """Sync execution (not used in async mode)"""
        raise NotImplementedError("Use async_execute instead")
    
    async def _async_execute(self, input: Dict[str, str], 
                            spatial_info: Dict[str, Dict], 
                            temporal_info: Dict[str, Dict]) -> str:
        """Execute with cache extraction and injection"""
        
        # Get graph reference (set by CacheGraph)
        graph = getattr(self, 'graph', None)
        
        # Prepare messages
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        # Get fused cache from predecessors (if cache enabled)
        past_kv = None
        if graph and hasattr(graph, 'get_fused_cache'):
            past_kv = graph.get_fused_cache(self)
        
        # Generate with cache
        if hasattr(self.llm, 'agen_with_cache'):
            response, kv_cache = await self.llm.agen_with_cache(
                messages, 
                past_key_values=past_kv,
                latent_steps=10
            )
            
            # Store cache for other agents to use
            if graph and hasattr(graph, 'store_node_cache'):
                graph.store_node_cache(self.id, kv_cache)
        else:
            # Fallback to regular generation
            response = await self.llm.agen(messages)
        
        return response
