"""
CacheDesigner Graph: Extends GDesigner Graph with KV-cache communication
Minimal changes to add cache-to-cache communication on top of GDesigner
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from GDesigner.graph.graph import Graph
from GDesigner.graph.node import Node

class CacheFuser(nn.Module):
    """Minimal cache fuser for KV-cache communication between agents"""
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_gates = nn.Parameter(torch.zeros(num_layers))
        self.fusion_weights = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, receiver_cache, sharer_caches, edge_weights, tau=0.5):
        """Fuse caches with gating"""
        if not sharer_caches:
            return receiver_cache
        
        fused = []
        for l in range(self.num_layers):
            gate = torch.sigmoid(self.layer_gates[l] / tau)
            r_cache = receiver_cache[l] if l < len(receiver_cache) else None
            if r_cache is None:
                fused.append(None)
                continue
            
            # Aggregate sharer caches
            agg = sum(w * sc[l] for w, sc in zip(edge_weights, sharer_caches) if l < len(sc))
            agg = agg / len(sharer_caches) if sharer_caches else 0
            
            # Residual fusion
            fused.append(r_cache + gate * self.fusion_weights[l] * agg)
        return fused

class CacheGraph(Graph):
    """
    CacheDesigner: Graph with cache-to-cache communication
    Extends GDesigner Graph with minimal KV-cache fusion capability
    """
    def __init__(self, *args, use_cache_communication: bool = True, 
                 hidden_dim: int = 4096, num_cache_layers: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache_communication = use_cache_communication
        
        if use_cache_communication:
            self.cache_fuser = CacheFuser(hidden_dim, num_cache_layers)
            # Store node caches
            self.node_caches: Dict[str, Any] = {}
    
    def store_node_cache(self, node_id: str, cache: Any):
        """Store KV-cache for a node after execution"""
        if self.use_cache_communication:
            print(f"\n💾 [GRAPH] Storing cache for node {node_id}")
            print(f"   Cache layers: {len(cache) if cache else 0}")
            if cache:
                print(f"   Cache shape: {cache[0][0].shape if len(cache) > 0 else 'N/A'}")
            self.node_caches[node_id] = cache
    
    def get_fused_cache(self, node: Node) -> Optional[Any]:
        """Get fused cache for a node from its spatial predecessors"""
        print(f"\n🔄 [GRAPH] Getting fused cache for node {node.id}")
        
        if not self.use_cache_communication:
            print(f"   ⚠️ Cache communication disabled")
            return None
            
        if not node.spatial_predecessors:
            print(f"   ℹ️ No spatial predecessors")
            return None
        
        print(f"   Predecessors: {[p.id for p in node.spatial_predecessors]}")
        
        # Collect caches from predecessors
        sharer_caches = []
        edge_weights = []
        for pred in node.spatial_predecessors:
            if pred.id in self.node_caches:
                print(f"   ✅ Found cache from {pred.id}")
                sharer_caches.append(self.node_caches[pred.id])
                edge_weights.append(1.0 / len(node.spatial_predecessors))
            else:
                print(f"   ❌ No cache from {pred.id}")
        
        if not sharer_caches:
            print(f"   ⚠️ No predecessor caches available")
            return None
        
        # Get receiver's own cache (if exists)
        receiver_cache = self.node_caches.get(node.id, None)
        
        # Fuse caches
        if receiver_cache is not None:
            print(f"   🧪 Fusing {len(sharer_caches)} caches with receiver cache")
            return self.cache_fuser(receiver_cache, sharer_caches, edge_weights)
        
        print(f"   🔄 Using first sharer cache (no receiver cache yet)")
        return sharer_caches[0]  # Use first sharer if no receiver cache
    
    async def arun(self, input: Dict[str, str], num_rounds: int = 3, 
                   max_tries: int = 3, max_time: int = 600) -> List[Any]:
        """Override arun to include cache communication"""
        # Clear caches at start
        if self.use_cache_communication:
            self.node_caches.clear()
        
        # Pass graph reference to all nodes so they can access cache methods
        for node in self.nodes.values():
            node.graph = self
        
        # Call parent's arun
        return await super().arun(input, num_rounds, max_tries, max_time)
