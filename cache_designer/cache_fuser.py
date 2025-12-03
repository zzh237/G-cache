"""
Cache Fuser Module for CacheDesigner
Implements layer-wise KV-cache fusion with gating and edge weights
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

class CacheFuser(nn.Module):
    """Fuses KV-caches from multiple Sharer agents into a Receiver agent"""
    
    def __init__(self, hidden_dim: int, num_layers: int, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Layer gates: learnable binary gates for each layer
        self.layer_gates_alpha = nn.Parameter(torch.zeros(num_layers))
        
        # Alignment MLPs for each layer
        self.align_mlps_k = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.align_mlps_v = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Residual fusion MLPs
        self.fusion_mlps_k = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.fusion_mlps_v = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Edge weight scaling factors per layer
        self.beta_layers = nn.Parameter(torch.ones(num_layers))
        
    def gumbel_sigmoid(self, alpha: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """Gumbel-Sigmoid for differentiable binary gates"""
        return torch.sigmoid(alpha / tau)
    
    def align_cache(self, sharer_cache: Tuple[torch.Tensor, torch.Tensor], 
                    layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align Sharer cache to Receiver dimension"""
        k, v = sharer_cache
        aligned_k = self.align_mlps_k[layer_idx](k)
        aligned_v = self.align_mlps_v[layer_idx](v)
        return aligned_k, aligned_v
    
    def aggregate_sharers(self, aligned_caches: List[Tuple[torch.Tensor, torch.Tensor]], 
                         edge_weights: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate multiple Sharer caches with edge weights"""
        if not aligned_caches:
            return None, None
        
        weighted_k = sum(w * k for (k, v), w in zip(aligned_caches, edge_weights))
        weighted_v = sum(w * v for (k, v), w in zip(aligned_caches, edge_weights))
        
        return weighted_k / len(aligned_caches), weighted_v / len(aligned_caches)
    
    def fuse_layer(self, receiver_cache: Tuple[torch.Tensor, torch.Tensor],
                   aggregated_cache: Tuple[torch.Tensor, torch.Tensor],
                   layer_idx: int, gate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse aggregated Sharer cache with Receiver cache"""
        r_k, r_v = receiver_cache
        a_k, a_v = aggregated_cache
        
        # Concatenate and fuse
        delta_k = self.fusion_mlps_k[layer_idx](torch.cat([r_k, a_k], dim=-1))
        delta_v = self.fusion_mlps_v[layer_idx](torch.cat([r_v, a_v], dim=-1))
        
        # Residual connection with gating
        fused_k = r_k + gate * delta_k
        fused_v = r_v + gate * delta_v
        
        return fused_k, fused_v
    
    def forward(self, receiver_caches: List[Tuple[torch.Tensor, torch.Tensor]],
                sharer_caches_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                edge_weights: List[List[float]],
                tau: float = 0.5) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            receiver_caches: List of (K, V) for each layer of Receiver
            sharer_caches_list: List of [List of (K, V) for each Sharer] per layer
            edge_weights: List of [weights for each Sharer] per layer
            tau: Temperature for Gumbel-Sigmoid
        
        Returns:
            Fused caches for each layer
        """
        fused_caches = []
        
        for layer_idx in range(self.num_layers):
            gate = self.gumbel_sigmoid(self.layer_gates_alpha[layer_idx], tau)
            
            if not sharer_caches_list[layer_idx]:
                fused_caches.append(receiver_caches[layer_idx])
                continue
            
            # Align all Sharer caches
            aligned = [self.align_cache(sc, layer_idx) for sc in sharer_caches_list[layer_idx]]
            
            # Aggregate with edge weights
            agg_k, agg_v = self.aggregate_sharers(aligned, edge_weights[layer_idx])
            
            # Fuse with Receiver
            fused = self.fuse_layer(receiver_caches[layer_idx], (agg_k, agg_v), layer_idx, gate)
            fused_caches.append(fused)
        
        return fused_caches
