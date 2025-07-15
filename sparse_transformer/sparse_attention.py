"""
Sparse attention implementation with efficient masking and caching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseMultiHeadAttention(nn.Module):
    """
    Multi‐head attention with aggressive sparse patterns.
    Uses F.scaled_dot_product_attention for efficiency when possible.
    - Cluster 0: focused local
    - Cluster 3: diffuse attention (wider local)
    - Cluster 1: strided attention
    - Cluster 2: global attention with anchor points
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True, mask_subset="0123"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.dropout       = dropout
        self.head_dim      = embedding_dim // num_heads
        self.mask_subset   = mask_subset  # For ablation study
        self.use_sdpa      = True  # Can be toggled for profiling comparison
        
        # Parse active clusters for ablation
        self.active_clusters = set(int(c) for c in mask_subset)

        assert self.head_dim * num_heads == embedding_dim, \
            "embedding_dim must be divisible by num_heads"

        # Filter cluster assignments based on active clusters for ablation
        if num_heads == 8:
            full_counts = {0: 1, 3: 2, 1: 3, 2: 2} # Original aggressive
        elif num_heads == 6:
            full_counts = {0: 1, 3: 1, 1: 2, 2: 2} # Adapted for 6 heads
        elif num_heads == 4:
            full_counts = {0: 1, 3: 1, 1: 1, 2: 1}
        elif num_heads == 2:
            full_counts = {0:1, 1:1} # Minimal: local and one sparse type
        elif num_heads == 1:
            full_counts = {0:1} # Purely local
        else:
            # Fallback: distribute as evenly as possible, prioritizing defined types
            full_counts = {0:0, 1:0, 2:0, 3:0}
            order_of_preference = [0, 1, 2, 3] # Prioritize local, then sparse types
            
            # Ensure each defined cluster type gets at least one head if num_heads allows
            active_clusters_for_fallback = [c for c in order_of_preference if c in self.window_sizes or c in self.strides or c == 2]

            heads_to_assign = num_heads
            for cluster_type in active_clusters_for_fallback:
                if heads_to_assign > 0:
                    full_counts[cluster_type] = 1
                    heads_to_assign -=1
            
            # Distribute remaining heads according to preference
            cluster_idx_ptr = 0
            while heads_to_assign > 0:
                target_cluster = active_clusters_for_fallback[cluster_idx_ptr % len(active_clusters_for_fallback)]
                full_counts[target_cluster] +=1
                heads_to_assign -=1
                cluster_idx_ptr +=1
        
        # Filter based on active clusters for ablation study
        self.cluster_head_counts = {}
        total_active_heads = 0
        
        for cluster_id, count in full_counts.items():
            if cluster_id in self.active_clusters:
                self.cluster_head_counts[cluster_id] = count
                total_active_heads += count
        
        # If no active clusters match, default to cluster 0 (focused-local)
        if not self.cluster_head_counts:
            self.cluster_head_counts = {0: num_heads}
            total_active_heads = num_heads
        
        # Redistribute heads if we have fewer than num_heads due to filtering
        if total_active_heads < num_heads:
            remaining_heads = num_heads - total_active_heads
            # Distribute remaining heads among active clusters
            active_cluster_list = list(self.cluster_head_counts.keys())
            for i in range(remaining_heads):
                cluster_to_add = active_cluster_list[i % len(active_cluster_list)]
                self.cluster_head_counts[cluster_to_add] += 1
        
        if sum(self.cluster_head_counts.values()) != num_heads:
            # This can happen if num_heads < number of primary cluster types we want to assign 1 to.
            # Re-normalize if sum is off but heads were assigned.
            print(f"Warning: Cluster head counts sum {sum(self.cluster_head_counts.values())} does not match num_heads {num_heads}. Adjusting.")
            # Simple greedy reduction or addition - this part might need refinement for optimal distribution
            diff = num_heads - sum(self.cluster_head_counts.values())
            # For now, if diff is non-zero, it's a misconfiguration or too few heads for the strategy.
            # We'll proceed with the possibly mismatched sum, or could raise error.
            # For robust operation, ensure the sum matches.
            # A simple fix if sum < num_heads: add to first cluster type.
            if diff > 0 and 0 in self.cluster_head_counts: self.cluster_head_counts[0] += diff
            # If sum > num_heads (less likely with above logic but good to be aware)
            # assert sum(self.cluster_head_counts.values()) == num_heads, \
            #    f"Final cluster head counts sum {sum(self.cluster_head_counts.values())} != num_heads {num_heads}"


        self.q_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # self.scale is not needed if F.scaled_dot_product_attention is used,
        # but keep for manual path.
        self.scale = math.sqrt(self.head_dim)


        self._mask_cache = {} # Store masks on CPU, move to device on demand


        self.window_sizes = {
            0: 16,
            3: 32,
            1: 64, # Used with stride
            # Cluster 2 (global) doesn't use window_size in the same way
        }
        self.strides = {
            # Cluster 0: no stride (local)
            3: 4,  # For cluster 3, local window is primary, stride is secondary if combined
            1: 8,
            2: 16  # Strided component for global attention
        }
        self.global_anchors = [
            0.0, 0.1, 0.2, 0.3, 0.382, 0.5, 0.618, 0.7, 0.8, 0.9, 1.0
        ]

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _create_cluster_mask(self, seq_len, cluster_idx, device):
        """
        Build non-causal attention mask for a cluster. True means ATTENTION IS ALLOWED.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Cluster 0: Local window
        if cluster_idx == 0:
            window = self.window_sizes[0]
            for i in range(seq_len):
                start = max(0, i - window // 2)
                end = min(seq_len, i + window // 2 + 1)
                mask[i, start:end] = True
        
        # Cluster 3: Wider Local window (+ optional stride, but primarily local)
        elif cluster_idx == 3:
            window = self.window_sizes[3]
            # Stride for this cluster is more of a secondary thought, prioritize local
            # stride_val = self.strides.get(3,1) 
            for i in range(seq_len):
                start = max(0, i - window // 2)
                end = min(seq_len, i + window // 2 + 1)
                mask[i, start:end] = True
                # if stride_val > 1:
                #     strided_indices = torch.arange(0, seq_len, stride_val, device=device)
                #     mask[i, strided_indices] = True


        # Cluster 1: Strided attention (+ local window)
        elif cluster_idx == 1:
            window = self.window_sizes[1] # Local window component
            stride_val = self.strides[1]
            for i in range(seq_len):
                # Local part
                start = max(0, i - window // 2)
                end = min(seq_len, i + window // 2 + 1)
                mask[i, start:end] = True
                # Strided part
                if stride_val > 1 and seq_len >= stride_val : # check seq_len for stride
                    strided_indices = torch.arange(0, seq_len, stride_val, device=device)
                    mask[i, strided_indices] = True
        
        # Cluster 2: Global anchors + strided
        elif cluster_idx == 2:
            stride_val = self.strides[2]
            for i in range(seq_len):
                # Global anchors
                for ratio in self.global_anchors:
                    idx = min(seq_len - 1, int(ratio * (seq_len -1 ))) # Ensure index is within bounds
                    mask[i, idx] = True
                # Strided part
                if stride_val > 1 and seq_len >= stride_val:
                    strided_indices = torch.arange(0, seq_len, stride_val, device=device)
                    mask[i, strided_indices] = True
        
        # DO NOT MAKE CAUSAL HERE. Causality is applied globally later.
        return mask

    def _get_or_create_mask(self, seq_len, cluster_idx, device):
        # Caching masks on CPU and moving to device as needed
        # For simplicity in this refactor, let's keep device in cache key if _create_cluster_mask uses it.
        # Or ensure _create_cluster_mask creates on CPU. Let's assume CPU cache for now.
        cpu_device = torch.device('cpu')
        cache_key = (cluster_idx, seq_len) # Use (seq_len) only if cluster_idx implies fixed structure
        
        if cache_key not in self._mask_cache:
            # Create on CPU
            self._mask_cache[cache_key] = self._create_cluster_mask(seq_len, cluster_idx, cpu_device)
        
        return self._mask_cache[cache_key].to(device)


    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        """
        Args:
            query: [B, L, E]
            key: [B, S, E]  (S=L for self-attention)
            value: [B, S, E] (S=L for self-attention)
            attn_mask: Optional [Lq, Lkv] or [B, Lq, Lkv] boolean mask, True if PREVENT attention.
                       Usually the causal mask.
            key_padding_mask: Optional [B, S] boolean mask, True if PADDED.
            need_weights: If True, return attention probabilities (forces manual path).
        
        Returns:
            if need_weights=False:  → [B, L, E] (context)
            if need_weights=True:   → ([B, L, E], [B, H, L, S]) (context, attention_probs)
        """
        bsz, seq_len_q, _ = query.size()
        seq_len_kv = key.size(1)
        device = query.device

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._shape(q, seq_len_q, bsz) # [B, H, Lq, Dk]
        k = self._shape(k, seq_len_kv, bsz) # [B, H, Lkv, Dk]
        v = self._shape(v, seq_len_kv, bsz) # [B, H, Lkv, Dv]

        # Use F.scaled_dot_product_attention if available, enabled, and not needing weights
        use_sdpa = (hasattr(F, 'scaled_dot_product_attention') and 
                   self.use_sdpa and not need_weights)

        if use_sdpa:
            # 1. Create static sparse masks (allowed, non-causal) per head: [H, Lq, Lkv]
            #    True means attention is allowed by the static pattern.
            static_sparse_masks_allowed = torch.zeros(self.num_heads, seq_len_q, seq_len_kv, dtype=torch.bool, device=device)
            current_head = 0
            # Ensure cluster_head_counts sums to num_heads
            actual_assigned_heads = 0
            for cluster_idx_key in self.cluster_head_counts.keys(): # Iterate over defined clusters
                 num_cluster_heads = self.cluster_head_counts.get(cluster_idx_key, 0)
                 if num_cluster_heads == 0 or current_head >= self.num_heads:
                     continue
                 
                 # Ensure we don't exceed self.num_heads
                 heads_for_this_cluster = min(num_cluster_heads, self.num_heads - current_head)
                 if heads_for_this_cluster <= 0: continue

                 # _get_or_create_mask returns [Lq, Lkv] mask
                 cluster_mask_pattern = self._get_or_create_mask(seq_len_kv, cluster_idx_key, device) # Assuming Lq=Lkv for patterns
                 if seq_len_q != seq_len_kv: # Adjust if cross-attention with different lengths
                     cluster_mask_pattern = cluster_mask_pattern[:seq_len_q, :seq_len_kv]


                 static_sparse_masks_allowed[current_head : current_head + heads_for_this_cluster, :, :] = cluster_mask_pattern.unsqueeze(0)
                 current_head += heads_for_this_cluster
                 actual_assigned_heads += heads_for_this_cluster
            
            if actual_assigned_heads < self.num_heads:
                 # If some heads didn't get any specific sparse mask (e.g. due to cluster_head_counts issue or too few heads)
                 # default them to allow all (full attention) before causal/padding.
                 static_sparse_masks_allowed[actual_assigned_heads : self.num_heads, :, :] = True


            # 2. Create causal mask (allowed): [1, 1, Lq, Lkv] (True means allowed)
            #    Only makes sense if Lq == Lkv (self-attention)
            causal_mask_allowed = torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=device)
            if seq_len_q == seq_len_kv: # Apply causal mask only for self-attention contexts
                causal_mask_allowed = torch.tril(causal_mask_allowed)
            causal_mask_allowed = causal_mask_allowed.unsqueeze(0).unsqueeze(0) # For head and batch broadcasting

            # 3. Combine sparse and causal masks (True means allowed)
            #    [1,H,Lq,Lkv] & [1,1,Lq,Lkv] -> [1,H,Lq,Lkv]
            combined_mask_allowed = static_sparse_masks_allowed.unsqueeze(0) & causal_mask_allowed
            
            # 4. Convert to SDPA format (True means MASK OUT / IGNORE)
            #    attn_mask for SDPA: [B,H,Lq,Lkv] or broadcastable
            final_sdpa_attn_mask = ~combined_mask_allowed # [1,H,Lq,Lkv] (True means MASK OUT)
                                                        # This will broadcast across batch dimension for SDPA.

            # 5. Incorporate external attn_mask (e.g. causal mask from a block)
            #    attn_mask is [Lq,Lkv] or [B,Lq,Lkv], True means MASK OUT
            if attn_mask is not None:
                if attn_mask.dim() == 2: # [Lq, Lkv]
                    attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0) # -> [1,1,Lq,Lkv]
                elif attn_mask.dim() == 3: # [B,Lq,Lkv]
                    attn_mask_expanded = attn_mask.unsqueeze(1) # -> [B,1,Lq,Lkv]
                else:
                    raise ValueError(f"attn_mask has unexpected dimensions: {attn_mask.dim()}")
                final_sdpa_attn_mask = final_sdpa_attn_mask | attn_mask_expanded


            # 6. Incorporate key_padding_mask ([B,S], True if PADDED)
            if key_padding_mask is not None:
                # padding_mask_for_scores: [B,1,1,S] (True if key is padded/masked)
                padding_mask_for_scores = key_padding_mask.unsqueeze(1).unsqueeze(2)
                # Expand final_sdpa_attn_mask to batch size if it's 1 and needs to match padding mask
                if final_sdpa_attn_mask.size(0) == 1 and padding_mask_for_scores.size(0) > 1:
                    final_sdpa_attn_mask = final_sdpa_attn_mask.expand(padding_mask_for_scores.size(0), -1, -1, -1)
                
                final_sdpa_attn_mask = final_sdpa_attn_mask | padding_mask_for_scores
            
            # F.scaled_dot_product_attention handles scaling and dropout.
            # is_causal flag is False because causality is baked into final_sdpa_attn_mask.
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=final_sdpa_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False # Causality is part of final_sdpa_attn_mask
            )
            attn_probs = None # Not returned by SDPA default path
        
        else: # Manual path (need_weights=True or SDPA not available)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            # Construct final_mask_inverted [B, H, Lq, Lkv] (True = MASK OUT)
            # Logic is identical to SDPA mask generation path
            static_sparse_masks_allowed = torch.zeros(self.num_heads, seq_len_q, seq_len_kv, dtype=torch.bool, device=device)
            current_head = 0
            actual_assigned_heads = 0
            for cluster_idx_key in self.cluster_head_counts.keys():
                 num_cluster_heads = self.cluster_head_counts.get(cluster_idx_key, 0)
                 if num_cluster_heads == 0 or current_head >= self.num_heads: continue
                 heads_for_this_cluster = min(num_cluster_heads, self.num_heads - current_head)
                 if heads_for_this_cluster <=0: continue
                 
                 cluster_mask_pattern = self._get_or_create_mask(seq_len_kv, cluster_idx_key, device)
                 if seq_len_q != seq_len_kv:
                     cluster_mask_pattern = cluster_mask_pattern[:seq_len_q, :seq_len_kv]
                 static_sparse_masks_allowed[current_head : current_head + heads_for_this_cluster, :, :] = cluster_mask_pattern.unsqueeze(0)
                 current_head += heads_for_this_cluster
                 actual_assigned_heads += heads_for_this_cluster

            if actual_assigned_heads < self.num_heads:
                 static_sparse_masks_allowed[actual_assigned_heads : self.num_heads, :, :] = True


            causal_mask_allowed = torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=device)
            if seq_len_q == seq_len_kv:
                causal_mask_allowed = torch.tril(causal_mask_allowed)
            causal_mask_allowed = causal_mask_allowed.unsqueeze(0).unsqueeze(0)

            combined_mask_allowed = static_sparse_masks_allowed.unsqueeze(0) & causal_mask_allowed
            final_mask_inverted = ~combined_mask_allowed.expand(bsz, -1, -1, -1) # [B,H,Lq,Lkv]

            # Incorporate external attn_mask (True = MASK OUT)
            if attn_mask is not None:
                if attn_mask.dim() == 2: # [Lq, Lkv]
                    attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0) # -> [1,1,Lq,Lkv]
                elif attn_mask.dim() == 3: # [B,Lq,Lkv]
                    attn_mask_expanded = attn_mask.unsqueeze(1) # -> [B,1,Lq,Lkv]
                else:
                    raise ValueError(f"attn_mask has unexpected dimensions: {attn_mask.dim()}")
                final_mask_inverted = final_mask_inverted | attn_mask_expanded


            if key_padding_mask is not None:
                padding_mask_for_scores = key_padding_mask.unsqueeze(1).unsqueeze(2) # [B,1,1,S]
                final_mask_inverted = final_mask_inverted | padding_mask_for_scores
            
            attn_scores = attn_scores.masked_fill(final_mask_inverted, torch.finfo(attn_scores.dtype).min)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).reshape(bsz, seq_len_q, self.embedding_dim)
        out = self.out_proj(context)

        if need_weights:
            if use_sdpa: # Should not happen if logic is correct, but as a safeguard
                print("Warning: need_weights=True but F.scaled_dot_product_attention was used; returning None for probs.")
                return out, None 
            return out, attn_probs
        return out

class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, mask_subset="0123"):
        super().__init__()
        self.self_attn = SparseMultiHeadAttention(d_model, nhead, dropout=dropout, mask_subset=mask_subset)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None): # Removed attn_mask, this layer doesn't pass causal mask
        # Pre-norm
        norm_src = self.norm1(src)
        
        # Self-attention block
        # SparseMultiHeadAttention handles its own causal masking combined with sparse patterns.
        # Only pass key_padding_mask if available.
        src2 = self.self_attn(norm_src, norm_src, norm_src, key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)

        # Feedforward block
        norm_src_ff = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(norm_src_ff))))
        src = src + self.dropout2(src2)
        return src



