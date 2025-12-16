"""
Cache-enabled Final Decision Agents V2 - Agent-type-aware cache usage
Follows LatentMAS: judger agents use accumulated cache from predecessors
"""
from typing import List, Any, Dict, Optional
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register('FinalReferCacheV2')
class FinalReferCacheV2(Node):
    """
    V2: Cache-aware final decision agent (judger)
    Uses accumulated cache from intermediate agents, generates final text
    """
    def __init__(self, id: str = None, domain: str = "", llm_name: str = "",
                 generation_mode: str = "api_hint", 
                 max_new_tokens: int = 2048):
        super().__init__(id, "FinalReferCacheV2", domain, llm_name)
        
        # Use cache-enabled LLM V2
        self.llm = LLMRegistry.get(llm_name, use_cache=True)
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
    
    def _process_inputs(self, raw_inputs: Dict[str, str], 
                       spatial_info: Dict[str, Any], 
                       temporal_info: Dict[str, Any]) -> tuple:
        """Process inputs for final decision"""
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        
        spatial_str = ""
        for id, info in spatial_info.items():
            # Skip empty outputs from intermediate agents (they only generate cache)
            if info['output'].strip():
                spatial_str += id + ": " + info['output'] + "\n\n"
        
        # If no text outputs, add a note
        if not spatial_str.strip():
            spatial_str = "[Note: Intermediate agents generated latent cache is used to generate the response]\n\n"
        
        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = f"{decision_few_shot} The task is:\n\n {raw_inputs['task']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str}"
        
        return system_prompt, user_prompt
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], 
                temporal_info: Dict[str, Any]):
        raise NotImplementedError("Use async_execute instead")
    
    async def _async_execute(self, input: Dict[str, str], 
                            spatial_info: Dict[str, Any], 
                            temporal_info: Dict[str, Any]) -> str:
        """
        Execute with cache-aware judger pattern
        
        Flow:
        - Get fused cache from intermediate agents
        - Generate final text using accumulated cache (NO new latent generation)
        """
        graph = getattr(self, 'graph', None)
        print(f"\n🏆 [STEP 11] FinalReferCacheV2 (Judger) - Aggregating {len(spatial_info)} agent outputs")
        
        # Debug: Check graph attribute
        print(f"\n🔍 [DEBUG] Checking graph attribute:")
        print(f"   graph = {graph}")
        print(f"   graph type = {type(graph)}")
        print(f"   graph is None? {graph is None}")
        if graph:
            print(f"   hasattr(graph, 'get_fused_cache')? {hasattr(graph, 'get_fused_cache')}")
            print(f"   graph class: {graph.__class__.__name__}")
            print(f"   graph attributes: {[attr for attr in dir(graph) if not attr.startswith('_')][:10]}")
        
        # Step 1: Get fused cache from predecessors (intermediate agents)
        past_kv = None
        has_cache = False
        if graph and hasattr(graph, 'get_fused_cache'):
            print(f"\n🎯 [STEP 11.1] Calling graph.get_fused_cache() for judger node {self.id}")
            past_kv = graph.get_fused_cache(self)
            has_cache = past_kv is not None
            print(f"\n📥 [STEP 11.2] Received fused cache: {has_cache}")
            if has_cache:
                print(f"   📊 Cache: {len(past_kv)} layers, seq_len={past_kv[0][0].shape[2]}")
        
        # Step 2: Build prompt
        print(f"\n📝 [STEP 11.3] Building final decision prompt")
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        
        # Step 3: Generate final text using cache (judger mode)
        if hasattr(self.llm, 'agen_with_cache'):
            print(f"\n🧠 [STEP 11.4] Calling llm.agen_with_cache() - Agent type: judger")
            
            response, kv_cache = await self.llm.agen_with_cache(
                messages, 
                past_key_values=past_kv,
                latent_steps=0,  # Judger doesn't generate new latent cache
                agent_type="judger",  # NEW: Judger mode!
                generation_mode=self.generation_mode,
                max_tokens=self.max_new_tokens,
                agent_id=self.id,  # NEW: Pass agent ID for logging
            )
            
            # Store cache for potential future use (though judger is usually last)
            if graph and hasattr(graph, 'store_node_cache'):
                graph.store_node_cache(self.id, kv_cache)
                print(f"\n💾 [STEP 11.5] Stored cache for judger node {self.id}")
            
            print(f"\n✅ [STEP 11.6] Judger decision complete: {len(response)} chars")
            print(f"\n📝 [STEP 11.7] Full conversation context for Agent {self.id}:")
            print(f"   ========== SYSTEM PROMPT ==========")
            print(f"   {system_prompt}")
            print(f"   ========== USER PROMPT ==========")
            print(f"   {user_prompt[:10000]}...") if len(user_prompt) > 10000 else print(f"   {user_prompt}")
            print(f"   ========== AGENT RESPONSE ==========")
            print(f"   {response}")
            print(f"   ====================================")
            return response
        else:
            # Fallback: text-only
            print(f"\n🧠 [STEP 11.4] Fallback to text-only generation")
            response = await self.llm.agen(messages)
            return response
