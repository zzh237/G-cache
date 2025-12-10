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
    """Graph-guided cache fusion (supports both tensor and text caches)"""
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Learnable fusion weights per layer (for tensor caches)
        self.layer_gates = nn.Parameter(torch.zeros(num_layers))
        self.fusion_weights = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, sharer_caches: List, edge_weights: List[float], tau=0.5):
        """Fuse multiple caches from graph neighbors"""
        if not sharer_caches:
            return None
        
        # Check cache type: tensor (local model) or dict (API)
        first_cache = sharer_caches[0]
        
        # Text-based cache (API mode)
        if isinstance(first_cache, dict):
            return self._fuse_text_caches(sharer_caches, edge_weights)
        
        # Tensor-based cache (local model mode)
        elif isinstance(first_cache, tuple):
            return self._fuse_tensor_caches(sharer_caches, edge_weights, tau)
        
        return None
    
    def _fuse_text_caches(self, sharer_caches: List[Dict], edge_weights: List[float]) -> Dict:
        """Fuse text-based caches (API mode)"""
        # Weighted concatenation of summaries
        fused_summary = ""
        for w, cache in zip(edge_weights, sharer_caches):
            summary = cache.get('summary', '')
            if summary:
                # Add weight as importance indicator
                importance = "High" if w > 0.5 else "Medium" if w > 0.3 else "Low"
                fused_summary += f"[{importance} priority]: {summary}\n"
        
        return {'summary': fused_summary.strip()}
    
    def _fuse_tensor_caches(self, sharer_caches: List[Tuple], edge_weights: List[float], tau: float) -> Tuple:
        """Fuse tensor-based KV-caches (local model mode)"""
        # LatentMAS-style: If only one cache, return it directly (no fusion needed)
        if len(sharer_caches) == 1:
            return sharer_caches[0]
        
        # For multiple caches: Check if all have same sequence length
        seq_lengths = [cache[0][0].shape[2] for cache in sharer_caches if len(cache) > 0]
        if not seq_lengths:
            return None
        
        # If sequence lengths differ, use LatentMAS approach: take first cache only
        # (LatentMAS doesn't fuse - it passes sequentially)
        if len(set(seq_lengths)) > 1:
            print(f"   ⚠️ Sequence length mismatch: {seq_lengths}. Using first cache only (LatentMAS-style).")
            return sharer_caches[0]
        
        # All caches have same length - safe to fuse
        fused_layers = []
        for l in range(self.num_layers):
            gate = torch.sigmoid(self.layer_gates[l] / tau)
            
            # Weighted average of caches at layer l
            fused_k, fused_v = None, None
            for w, cache in zip(edge_weights, sharer_caches):
                if l >= len(cache):
                    continue
                k, v = cache[l]  # (batch, heads, seq, dim)
                
                if fused_k is None:
                    fused_k = w * k
                    fused_v = w * v
                else:
                    fused_k = fused_k + w * k
                    fused_v = fused_v + w * v
            
            # Apply learnable gate
            if fused_k is not None:
                fused_k = gate * self.fusion_weights[l] * fused_k
                fused_v = gate * self.fusion_weights[l] * fused_v
                fused_layers.append((fused_k, fused_v))
        
        return tuple(fused_layers) if fused_layers else None

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
            print(f"\n💾 [STEP 11] CacheGraph.store_node_cache() - Storing cache for node {node_id}")
            print(f"   Cache layers: {len(cache) if cache else 0}")
            if cache:
                print(f"   Cache shape: {cache[0][0].shape if len(cache) > 0 else 'N/A'}")
            self.node_caches[node_id] = cache
    
    def get_fused_cache(self, node: Node) -> Optional[Tuple]:
        """Get fused cache for a node from its spatial predecessors (LatentMAS-style)"""
        print(f"\n🔄 [STEP 4] CacheGraph.get_fused_cache() - Getting fused cache for node {node.id}")
        
        if not self.use_cache_communication:
            print(f"   ⚠️ Cache communication disabled")
            return None
            
        if not node.spatial_predecessors:
            print(f"   ℹ️ No spatial predecessors")
            return None
        
        print(f"   Predecessors: {[p.id for p in node.spatial_predecessors]}")
        
        # Collect caches from predecessors (graph-guided)
        sharer_caches = []
        edge_weights = []
        for pred in node.spatial_predecessors:
            if pred.id in self.node_caches:
                cache = self.node_caches[pred.id]
                if cache is not None:
                    print(f"   ✅ Found cache from {pred.id} ({len(cache)} layers)")
                    sharer_caches.append(cache)
                    # Use GCN edge weights if available
                    edge_weight = self.gcn.get_edge_weight(pred.id, node.id) if hasattr(self, 'gcn') else 1.0
                    edge_weights.append(edge_weight)
            else:
                print(f"   ❌ No cache from {pred.id}")
        
        if not sharer_caches:
            print(f"   ⚠️ No predecessor caches available")
            return None
        
        # Normalize edge weights
        total_weight = sum(edge_weights)
        edge_weights = [w / total_weight for w in edge_weights]
        
        # Fuse caches using learnable fusion
        print(f"   🧪 Fusing {len(sharer_caches)} caches with weights {edge_weights}")
        fused = self.cache_fuser(sharer_caches, edge_weights)
        
        if fused is None:
            print(f"   ❌ [ERROR] Cache fusion returned None! Checking why...")
            print(f"      - sharer_caches type: {type(sharer_caches[0]) if sharer_caches else 'empty'}")
            print(f"      - num_layers in fuser: {self.cache_fuser.num_layers}")
            if sharer_caches and isinstance(sharer_caches[0], tuple):
                print(f"      - actual cache layers: {len(sharer_caches[0])}")
        else:
            print(f"   ✅ Fusion successful: {len(fused)} layers")
        
        return fused
    
    async def arun(self, input: Dict[str, str], num_rounds: int = 3, 
                   max_tries: int = 3, max_time: int = 600) -> List[Any]:
        """Override arun to include cache communication"""
        print(f"\n🚀 [STEP 1] CacheGraph.arun() - Starting graph execution")
        # Clear caches at start
        if self.use_cache_communication:
            self.node_caches.clear()
            print(f"   ✅ Cleared node caches")
        
        # Pass graph reference to all nodes so they can access cache methods
        for node in self.nodes.values():
            node.graph = self
        print(f"   ✅ Set graph reference on {len(self.nodes)} nodes")
        
        # Call parent's arun
        print(f"   ➡️  Calling parent Graph.arun()...")
        return await super().arun(input, num_rounds, max_tries, max_time)
