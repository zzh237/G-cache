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
        print(f"   🔍 Cache type: {type(first_cache).__name__}")
        
        # Text-based cache (API mode)
        if isinstance(first_cache, dict):
            return self._fuse_text_caches(sharer_caches, edge_weights)
        
        # DynamicCache (HuggingFace cache object) - check by class name or __len__
        elif type(first_cache).__name__ == 'DynamicCache' or (hasattr(first_cache, '__len__') and not isinstance(first_cache, (tuple, list))):
            print(f"   🔄 Converting DynamicCache to tuple format...")
            print(f"   🔍 DynamicCache attributes: {dir(first_cache)[:10]}...")  # Show first 10 attributes
            converted_caches = []
            for cache in sharer_caches:
                # DynamicCache can be indexed like cache[layer_idx] -> (key, value)
                # Or accessed via to_legacy_cache() method
                if hasattr(cache, 'to_legacy_cache'):
                    tuple_cache = cache.to_legacy_cache()
                else:
                    # Try direct indexing
                    tuple_cache = tuple((cache[i][0], cache[i][1]) for i in range(len(cache)))
                converted_caches.append(tuple_cache)
            return self._fuse_tensor_caches(converted_caches, edge_weights, tau)
        
        # Tensor-based cache (local model mode)
        elif isinstance(first_cache, tuple):
            return self._fuse_tensor_caches(sharer_caches, edge_weights, tau)
        
        print(f"   ❌ Unknown cache type, returning None")
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
        
        # If sequence lengths differ, truncate all to shortest length (LatentMAS-inspired)
        if len(set(seq_lengths)) > 1:
            min_len = min(seq_lengths)
            print(f"   ⚠️ Sequence length mismatch: {seq_lengths}. Truncating all to {min_len} tokens.")
            truncated_caches = []
            for cache in sharer_caches:
                truncated_layers = []
                for k, v in cache:
                    # Keep only the last min_len tokens
                    k_truncated = k[..., -min_len:, :].contiguous()
                    v_truncated = v[..., -min_len:, :].contiguous()
                    truncated_layers.append((k_truncated, v_truncated))
                truncated_caches.append(tuple(truncated_layers))
            sharer_caches = truncated_caches
            seq_lengths = [min_len] * len(sharer_caches)
        
        # Now all caches have same length - safe to fuse
        # Use actual number of layers from cache, not self.num_layers
        actual_num_layers = len(sharer_caches[0])
        print(f"   📊 Detected {actual_num_layers} layers in cache (fuser has {self.num_layers} params)")
        fused_layers = []
        for l in range(actual_num_layers):
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
            
            # Apply learnable gate (only if layer exists in fuser params)
            if fused_k is not None:
                if l < self.num_layers:
                    gate = torch.sigmoid(self.layer_gates[l] / tau)
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
            print(f"\n💾 [STEP 10] CacheGraph.store_node_cache() - Storing cache for node {node_id}")
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
        collected = []
        print(f"\n   🔍 Iterating through {len(node.spatial_predecessors)} spatial predecessors...")
        for pred in node.spatial_predecessors:
            print(f"      • pred type: {type(pred)}, pred.id: {pred.id} (type: {type(pred.id)})")
            if pred.id in self.node_caches:
                cache = self.node_caches[pred.id]
                if cache is not None:
                    # print(f"        ✅ Found cache from {pred.id} ({len(cache)} layers)")
                    # print(f"           🔍 cache type: {type(cache)}")
                    # print(f"           🔍 is tuple: {isinstance(cache, tuple)}, is list: {isinstance(cache, list)}")
                    # if isinstance(cache, (tuple, list)) and len(cache) > 0:
                    #     print(f"           📐 Cache dimensions: Layer 0 Key={cache[0][0].shape}, Value={cache[0][1].shape}")
                    sharer_caches.append(cache)
                    # Use GCN edge weights if available
                    edge_weight = self.gcn.get_edge_weight(pred.id, node.id) if hasattr(self, 'gcn') else 1.0
                    # print(f"           ⚖️  Edge weight from {pred.id} to {node.id}: {edge_weight}")
                    # print(f"           📏 edge_weight type: {type(edge_weight)}, is scalar: {not hasattr(edge_weight, 'shape')}")
                    edge_weights.append(edge_weight)
                    collected.append((pred.id, cache, float(edge_weight)))
            else:
                print(f"        ❌ No cache from {pred.id}")
        
        if not sharer_caches:
            print(f"   ⚠️ No predecessor caches available")
            return None
        
        print(f"\n   📊 Collected {len(sharer_caches)} caches from predecessors")
        print(f"      • Raw edge_weights: {edge_weights}")
        
        # Normalize edge weights
        total_weight = sum(edge_weights)
        edge_weights = [w / total_weight for w in edge_weights]
        normed = [(pid, cache, w/total_weight) for pid, cache, w in collected]
        print(f"      • Normalized edge_weights: {edge_weights} (sum={sum(edge_weights):.2f})")
        
        # Fuse caches using learnable fusion
        print(f"   🧪 Fusing {len(sharer_caches)} caches with weights {edge_weights}")
        # print(f"   📊 Cache fusion breakdown (BEFORE fusion):")
        # for i, pred in enumerate(node.spatial_predecessors):
        #     if pred.id in self.node_caches and self.node_caches[pred.id]:
        #         cache = self.node_caches[pred.id]
        #         if isinstance(cache, tuple):
        #             k_shape = cache[0][0].shape
        #             v_shape = cache[0][1].shape
        #             print(f"      - Cache {i+1} from {pred.id}: weight={edge_weights[i]:.2f}")
        #             print(f"        Key shape: {k_shape}, Value shape: {v_shape}")
        #         else:
        #             print(f"      - Cache {i+1} from {pred.id}: weight={edge_weights[i]:.2f}, layers={len(cache)}")
        print(f"   🔍 [DIMENSIONS] Before cache_fuser call:")
        print(f"      • sharer_caches: list of {len(sharer_caches)} caches")
        for i, (pid, cache, w) in enumerate(normed):
            k_shape, v_shape = cache[0][0].shape, cache[0][1].shape
            print(f"  - Cache {i+1} from {pid}: lenght={len(cache)} weight={w:.2f}")
            print(f"    Key shape: {k_shape}, Value shape: {v_shape}")


        
        # if sharer_caches and isinstance(sharer_caches[0], tuple):
        #     print(f"        - Each cache: {len(sharer_caches[0])} layers (tuple of (key, value) pairs)")
        #     print(f"        - Layer 0 Key: {sharer_caches[0][0][0].shape}")
        #     print(f"        - Layer 0 Value: {sharer_caches[0][0][1].shape}")
        # print(f"      • edge_weights: {edge_weights} (length={len(edge_weights)})")
        
        fused = self.cache_fuser(sharer_caches, edge_weights)
        
        if fused and isinstance(fused, tuple):
            print(f"   📊 Cache fusion result (AFTER fusion):")
            # for i, cache in enumerate(fused):
            #     k_shape, v_shape = cache[0][0].shape, cache[0][1].shape
            #     print(f"  - Cache {i+1} from {pid}: lenght={len(cache)} weight={w:.2f}")
            #     print(f"    Key shape: {k_shape}, Value shape: {v_shape}")
                
            
            print(f"      - Fused cache: {len(fused)} layers")
            print(f"      - Key shape: {fused[0][0].shape}, Value shape: {fused[0][1].shape}")
            print(f"      - ✅ Dimension unchanged (weighted average, not concatenation)")
        
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
        node_ids = ', '.join(self.nodes.keys())
        print(f"   ✅ Set graph reference on {len(self.nodes)} nodes: [{node_ids}]")
        
        # Call parent's arun
        print(f"   ➡️  Calling parent Graph.arun()...")
        return await super().arun(input, num_rounds, max_tries, max_time)
