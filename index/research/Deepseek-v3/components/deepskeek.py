import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# Dummy RoPE application implementation for self-containment
def apply_rope(x, cos, sin):
    # Assumes x shape: [batch, seq_len, num_heads, head_dim]
    # Standard rotary embedding rotation logic
    return x # Simplified placeholder for structural layout

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d=7168, n_h=128, d_h=128, d_c=512, d_c_prime=1536, d_R_h=64):
        super().__init__()
        self.n_h = n_h        # Number of attention heads (128)
        self.d_h = d_h        # Dimension per head (128)
        self.d_c = d_c        # KV compression dimension (512)
        self.d_c_prime = d_c_prime # Query compression dimension (1536)
        self.d_R_h = d_R_h    # Decoupled RoPE dimension (64)
        
        # --- KV Compression Down/Up Projections ---
        self.W_DKV = nn.Linear(d, d_c, bias=False)
        self.W_UK = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_UV = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_KR = nn.Linear(d, d_R_h, bias=False) # Decoupled Key RoPE matrix
        self.compr_kv_norm = RMSNorm(d_c)
        
        # --- Query Compression Down/Up Projections ---
        self.W_DQ = nn.Linear(d, d_c_prime, bias=False)
        self.W_UQ = nn.Linear(d_c_prime, n_h * d_h, bias=False)
        self.W_QR = nn.Linear(d_c_prime, n_h * d_R_h, bias=False) # Decoupled Query RoPE matrix
        self.compr_q_norm = RMSNorm(d_c_prime)
        
        # --- Output Projection ---
        self.W_O = nn.Linear(n_h * d_h, d, bias=False)
        
        self.scale = 1.0 / ((d_h + d_R_h) ** 0.5)

    def forward(self, h_t, cos=None, sin=None):
        B, T, _ = h_t.shape
        
        # 1. Compress and reconstruct KV
        c_t_KV = self.compr_kv_norm(self.W_DKV(h_t)) # [B, T, d_c]
        k_t_C = self.W_UK(c_t_KV).view(B, T, self.n_h, self.d_h)
        v_t_C = self.W_UV(c_t_KV).view(B, T, self.n_h, self.d_h)
        
        # Decouple Key positional track
        k_t_R = self.W_KR(h_t).view(B, T, 1, self.d_R_h).expand(-1, -1, self.n_h, -1)
        if cos is not None and sin is not None:
            k_t_R = apply_rope(k_t_R, cos, sin)
            
        # 2. Compress and reconstruct Q
        c_t_Q = self.compr_q_norm(self.W_DQ(h_t)) # [B, T, d_c_prime]
        q_t_C = self.W_UQ(c_t_Q).view(B, T, self.n_h, self.d_h)
        
        # Decouple Query positional track
        q_t_R = self.W_QR(c_t_Q).view(B, T, self.n_h, self.d_R_h)
        if cos is not None and sin is not None:
            q_t_R = apply_rope(q_t_R, cos, sin)
            
        # 3. Concatenate structural and positional tracks
        # q_t,i = [q^C_t,i ; q^R_t,i] | k_t,i = [k^C_t,i ; k^R_t,i]
        q = torch.cat([q_t_C, q_t_R], dim=-1).transpose(1, 2) # [B, n_h, T, d_h + d_R_h]
        k = torch.cat([k_t_C, k_t_R], dim=-1).transpose(1, 2) # [B, n_h, T, d_h + d_R_h]
        v = v_t_C.transpose(1, 2)                             # [B, n_h, T, d_h]
        
        # 4. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, n_h, T, T]
        
        # Causal mask application
        mask = torch.triu(torch.full((T, T), float('-inf'), device=h_t.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(1)
        
        attn_weights = F.softmax(scores, dim=-1)
        o = torch.matmul(attn_weights, v) # [B, n_h, T, d_h]
        
        # 5. Output projection
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(o)


class DeepSeekMoE(nn.Module):
    def __init__(self, d=7168, N_s=1, N_r=256, K_r=8, d_exp=2048):
        super().__init__()
        self.N_s = N_s      # Number of shared experts (1)
        self.N_r = N_r      # Number of routed experts (256)
        self.K_r = K_r      # Activated routed experts per token (8)
        
        # Shared Experts (Dense Feed-Forward Networks)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d_exp, bias=False),
                nn.SiLU(),
                nn.Linear(d_exp, d, bias=False)
            ) for _ in range(N_s)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d_exp, bias=False),
                nn.SiLU(),
                nn.Linear(d_exp, d, bias=False)
            ) for _ in range(N_r)
        ])
        
        # Router settings
        self.router_centroids = nn.Parameter(torch.randn(N_r, d))
        self.register_buffer('bias', torch.zeros(N_r)) # For Aux-Loss-Free Load Balancing
        self.bias_update_speed = 0.001

    def forward(self, u_t, training=True):
        B, T, C = u_t.shape
        tokens = u_t.view(-1, C) # Flatten to [B * T, C]
        
        # 1. Compute Shared Experts Output
        shared_out = torch.zeros_like(tokens)
        for expert in self.shared_experts:
            shared_out += expert(tokens)
            
        # 2. Compute Routing Affinity Matrix via Sigmoid
        # s_i,t = Sigmoid(u_t^T * e_i)
        norm_tokens = F.normalize(tokens, dim=-1)
        norm_centroids = F.normalize(self.router_centroids, dim=-1)
        affinity = torch.sigmoid(torch.matmul(norm_tokens, norm_centroids.t())) # [B*T, N_r]
        
        # 3. Auxiliary-Loss-Free Routing Logic (Bias applied only for routing decisions)
        routing_scores = affinity + self.bias.unsqueeze(0)
        topk_scores, topk_indices = torch.topk(routing_scores, self.K_r, dim=-1)
        
        # 4. Gating Calculation (Normalized strictly from raw original affinity metrics)
        raw_gating = affinity.gather(dim=-1, index=topk_indices) # [B*T, K_r]
        gating_weights = raw_gating / (raw_gating.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 5. Route processing loops
        routed_out = torch.zeros_like(tokens)
        for idx in range(self.N_r):
            # Find tokens routed to expert idx
            mask = (topk_indices == idx)
            if not mask.any():
                continue
                
            token_indices, choice_positions = torch.where(mask)
            expert_inputs = tokens[token_indices]
            expert_outputs = self.routed_experts[idx](expert_inputs)
            
            # Apply corresponding gated normalization factor
            weight = gating_weights[token_indices, choice_positions].unsqueeze(-1)
            routed_out[token_indices] += expert_outputs * weight
            
        # 6. Adjust dynamic balancing biases during real-time training step executions
        if training:
            with torch.no_grad():
                # Track frequency loads across current batch
                expert_load = torch.bincount(topk_indices.view(-1), minlength=self.N_r).float()
                target_load = (B * T * self.K_r) / self.N_r
                
                # Overloaded items see dynamic deductions; underloaded targets earn adjustments
                self.bias[expert_load > target_load] -= self.bias_update_speed
                self.bias[expert_load < target_load] += self.bias_update_speed

        # Final layout synthesis mapping original dimensional metrics
        output = shared_out + routed_out
        return output.view(B, T, C)


class DeepSeekV3TransformerBlock(nn.Module):
    def __init__(self, d=7168):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = MultiHeadLatentAttention(d=d)
        
        self.ffn_norm = RMSNorm(d)
        self.moe = DeepSeekMoE(d=d)

    def forward(self, h_t, training=True):
        # Sub-Layer 1: Attention Residual Hook
        h_t = h_t + self.attn(self.attn_norm(h_t))
        
        # Sub-Layer 2: Feed Forward Mixture of Experts Residual Hook
        h_t = h_t + self.moe(self.ffn_norm(h_t), training=training)
        return h_t