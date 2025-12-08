"""
Cache-enabled Math Solver Agent - Combines GDesigner graph with LatentMAS cache
Novel contribution: Graph-guided multi-agent KV-cache communication
"""
from typing import List, Any, Dict, Optional, Tuple
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register('MathSolverCache')
class MathSolverCache(Node):
    """
    Math solver with dual-channel communication:
    1. Text channel: GDesigner's spatial/temporal graph structure
    2. Latent channel: LatentMAS's KV-cache sharing
    
    Key innovation: Graph topology guides which caches to fuse
    """
    
    def __init__(self, id: str = None, role: str = None, domain: str = "", llm_name: str = "",
                 cache_mode: str = "hybrid", generation_mode: str = "api_hint"):
        """
        Args:
            cache_mode: "hybrid" (text+cache), "latent_only" (cache only), "text_only" (no cache)
            generation_mode: "api_hint" (API with text hint), "hybrid" (local+API), "local" (local only)
        """
        super().__init__(id, "MathSolverCache", domain, llm_name)
        
        # Use cache-enabled LLM
        self.llm = LLMRegistry.get(llm_name, use_cache=True)
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        
        # Cache communication mode
        self.cache_mode = cache_mode  # "hybrid", "latent_only", "text_only"
        self.generation_mode = generation_mode  # "api_hint", "hybrid", "local"
    
    def _process_inputs(self, raw_inputs: Dict[str, str], 
                       spatial_info: Dict[str, Dict], 
                       temporal_info: Dict[str, Dict],
                       has_cache: bool = False) -> tuple:
        """Process inputs with cache-aware prompting"""
        system_prompt = self.constraint
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        
        # Hybrid mode: Use both text and cache
        if self.cache_mode == "hybrid":
            # Add text info from graph structure
            spatial_str = ""
            temporal_str = ""
            for id, info in spatial_info.items():
                spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
            for id, info in temporal_info.items():
                temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"
            
            if spatial_str:
                user_prompt += f"\n\nSpatial context (same round):\n{spatial_str}"
            if temporal_str:
                user_prompt += f"\n\nTemporal context (previous rounds):\n{temporal_str}"
            
            # Indicate cache availability
            if has_cache:
                user_prompt += "\n\n[Note: Latent representations from predecessor agents are also available in KV-cache]"
        
        # Latent-only mode: Minimal text, rely on cache
        elif self.cache_mode == "latent_only":
            if has_cache:
                user_prompt += "\n\n[Using latent context from predecessor agents via KV-cache]"
            # Don't add text from other agents
        
        # Text-only mode: Traditional GDesigner (fallback)
        else:  # text_only
            spatial_str = ""
            temporal_str = ""
            for id, info in spatial_info.items():
                spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
            for id, info in temporal_info.items():
                temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"
            
            user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
            user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        
        return system_prompt, user_prompt
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict, temporal_info: Dict):
        """Sync execution (not used in async mode)"""
        raise NotImplementedError("Use async_execute instead")
    


    async def _async_execute(self, input: Dict[str, str], 
                            spatial_info: Dict[str, Dict], 
                            temporal_info: Dict[str, Dict]) -> str:
        """
        Execute with graph-guided cache fusion (LatentMAS integration)
        
        Flow:
        1. Get fused cache from graph predecessors (graph-guided)
        2. Generate latent cache with LatentMAS (generate_latent_batch)
        3. Generate text from cache (generate_text_batch)
        4. Store cache for successors (graph-guided sharing)
        """
        graph = getattr(self, 'graph', None)
        
        # Step 1: Graph-guided cache retrieval
        past_kv = None
        has_cache = False
        if graph and hasattr(graph, 'get_fused_cache') and self.cache_mode != "text_only":
            past_kv = graph.get_fused_cache(self)  # Fused from spatial predecessors
            has_cache = past_kv is not None
            print(f"\n📥 [{self.id}] Received fused cache: {has_cache}")
        
        # Step 2: Process inputs
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, has_cache)
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        
        # Step 3: Generate with LatentMAS cache
        if hasattr(self.llm, 'agen_with_cache') and self.cache_mode != "text_only":
            print(f"🧠 [{self.id}] Generating with LatentMAS cache (latent_steps=10)")
            # Uses: generate_latent_batch + generate_text_batch
            response, kv_cache = await self.llm.agen_with_cache(
                messages, 
                past_key_values=past_kv,  # Input: fused cache from graph
                latent_steps=10,
                generation_mode=self.generation_mode,  # Pass generation mode
            )
            
            # Step 4: Store cache for graph successors
            if graph and hasattr(graph, 'store_node_cache'):
                graph.store_node_cache(self.id, kv_cache)
                print(f"💾 [{self.id}] Stored cache for {len(self.spatial_successors)} successors")
        else:
            # Fallback: text-only
            response = await self.llm.agen(messages)
        
        return response
    
    @property
    def verbose(self):
        """Check if verbose mode is enabled"""
        return getattr(self, '_verbose', False)
