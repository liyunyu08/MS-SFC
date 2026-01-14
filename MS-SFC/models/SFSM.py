import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------
# Helpers
# -----------------------
class LayerNormChannel(nn.Module):
    """LayerNorm over channel dimension for 4D tensors [B,C,H,W]"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        # x: [B,C,H,W] -> LN over channel dim
        b,c,h,w = x.shape
        x2 = x.permute(0,2,3,1).contiguous()  # [B,H,W,C]
        x2 = self.ln(x2)
        return x2.permute(0,3,1,2).contiguous() # [B,C,H,W]

def add_coord_channels(x):
    """
    Append two coordinate channels (x_coord, y_coord) normalized to [-1,1]
    x: [B,C,H,W]
    returns [B, C+2, H, W]
    """
    b, c, h, w = x.shape
    device = x.device
    # range -1 .. 1
    ys = torch.linspace(-1., 1., steps=h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(-1., 1., steps=w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, xs, ys], dim=1)

# -----------------------
# CrossAttentionEnhanced
# -----------------------
class CrossAttentionEnhanced(nn.Module):
    """
    Cross-attention where Q from 'query' domain attends to K/V from 'context' domain.
    Improvements:
    - LayerNorm on inputs
    - optional coordinate channels appended
    - depthwise conv before linear projections for local context
    - learnable temperature per head
    - returns updated query (residual), plus attention map if requested
    """
    def __init__(self, dim, num_heads=1, use_coords=True, proj_drop=0.0, bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_coords = use_coords

        # Normalization to align domains
        self.norm_q = LayerNormChannel(dim + (2 if use_coords else 0))
        self.norm_kv = LayerNormChannel(dim + (2 if use_coords else 0))

        # Depthwise conv to inject local context while preserving channels
        self.dw_q = nn.Conv2d(dim + (2 if use_coords else 0), dim + (2 if use_coords else 0),
                              kernel_size=3, padding=1, groups=dim + (2 if use_coords else 0), bias=False)
        self.dw_kv = nn.Conv2d(dim + (2 if use_coords else 0), dim + (2 if use_coords else 0),
                               kernel_size=3, padding=1, groups=dim + (2 if use_coords else 0), bias=False)

        # Linear projections (1x1 convs)
        self.q_proj = nn.Conv2d(dim + (2 if use_coords else 0), dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim + (2 if use_coords else 0), dim * 2, kernel_size=1, bias=bias)

        # output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # learnable temperature per head
        self.log_tau = nn.Parameter(torch.zeros(num_heads))

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context, return_attn=False):
        """
        query: [B, C, H, W]  (will be Q)
        context: [B, C, H, W] (will be K,V)  -- can be different H/W, we will adapt via interpolation if needed
        """
        b, c_q, hq, wq = query.shape
        b2, c_k, hk, wk = context.shape
        assert b == b2, "batch mismatch"

        # optionally add coords
        if self.use_coords:
            query_in = add_coord_channels(query)
            context_in = add_coord_channels(context)
        else:
            query_in = query
            context_in = context

        # if spatial sizes mismatch, bilinear interpolate context to query size for attention
        if (hk, wk) != (hq, wq):
            context_in = F.interpolate(context_in, size=(hq, wq), mode='bilinear', align_corners=False)

        # Normalize
        qn = self.norm_q(query_in)
        kn = self.norm_kv(context_in)

        # Depthwise conv for local info
        qd = self.dw_q(qn)
        kd = self.dw_kv(kn)

        # Projections
        q = self.q_proj(qd)                     # [B, dim, H, W]
        kv = self.kv_proj(kd)                   # [B, 2*dim, H, W]
        k, v = torch.chunk(kv, 2, dim=1)        # each [B, dim, H, W]

        # reshape for channel-attention: treat channels as 'tokens', positions as feature dim
        # rearrange to [B, heads, C_per_head, (H*W)]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 normalize across spatial dim to compute cosine-like similarity (as your original)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # attention over channels: (C x C) per head, but implemented as matrix multiply
        # logits: [b, head, c, c]
        logits = torch.matmul(q, k.transpose(-2, -1))  # dot over spatial dim -> channel affinity

        # scale by learned temperature
        tau = (self.log_tau).view(1, self.num_heads, 1, 1).exp() + 1e-6
        logits = logits * tau

        attn = F.softmax(logits, dim=-1)  # softmax over key-channels

        out = torch.matmul(attn, v)  # [b, head, c, (h*w)]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=hq, w=wq)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        # residual connection to query (keep original channel scale)
        out = out + query

        if return_attn:
            # return attn aggregated over heads (mean)
            attn_mean = attn.mean(dim=1)  # [b, c, c]
            return out, attn_mean
        else:
            return out

# -----------------------
# FuseBlockEnhanced
# -----------------------
class SFSM(nn.Module):
    """
    Two-way cross attention + competitive gating fusion.
    Steps:
      - Align (optionally) via small convs
      - fre <- CrossAttn(fre, spa) + fre
      - spa <- CrossAttn(spa, fre) + spa
      - competitive gating: compute 2-channel logits per (b, c, h, w), apply softmax across modality axis
      - final = gate_fre * fre + gate_spa * spa
    """
    def __init__(self, channels, attn_heads=1, use_coords=True, reduced_channels=None):
        super().__init__()
        self.channels = channels
        rc = reduced_channels or max(16, channels // 4)
        # light alignment convs
        self.align_fre = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.align_spa = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

        # cross attention modules
        self.attn_fre_from_spa = CrossAttentionEnhanced(dim=channels, num_heads=attn_heads, use_coords=use_coords)
        self.attn_spa_from_fre = CrossAttentionEnhanced(dim=channels, num_heads=attn_heads, use_coords=use_coords)

        # competitive gating: produce 2 logits per channel at each spatial location -> softmax across modality axis
        # We'll produce shape [B, 2, C, H, W]
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * channels, rc, kernel_size=1, bias=False),
            nn.BatchNorm2d(rc),
            nn.ReLU(inplace=True),
            nn.Conv2d(rc, 2 * channels, kernel_size=1)  # no BN, produce raw logits
        )

        # optional final fuse conv
        self.fuse_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fre, spa):
        """
        fre, spa: [B, C, H, W] (H/W may differ; attention module handles interpolation of context)
        """
        # align
        fre_a = self.align_fre(fre)
        spa_a = self.align_spa(spa)

        # cross-attn: fre queries spa
        fre_up = self.attn_fre_from_spa(fre_a, spa_a)  # [B,C,Hf,Wf] (Hf == fre H)
        fre = fre + fre_up  # residual

        # cross-attn: spa queries fre (note: attn module will interpolate fre -> spa size if needed)
        spa_up = self.attn_spa_from_fre(spa_a, fre_a)  # [B,C,Hs,Ws]
        spa = spa + spa_up

        # ensure same spatial for gating: upsample the smaller to larger
        # choose target size = fre spatial (could choose other)
        target_h, target_w = fre.shape[2], fre.shape[3]
        if (spa.shape[2], spa.shape[3]) != (target_h, target_w):
            spa_resized = F.interpolate(spa, size=(target_h, target_w), mode='bilinear', align_corners=False)
            fre_resized = fre
        else:
            spa_resized = spa
            fre_resized = fre

        # compute gating logits
        joint = torch.cat([fre_resized, spa_resized], dim=1)  # [B, 2C, H, W]
        gate_logits = self.gate_conv(joint)  # [B, 2C, H, W]
        # reshape to [B, 2, C, H, W]
        b, two_c, h, w = gate_logits.shape
        gate_logits = gate_logits.view(b, 2, self.channels, h, w)
        gates = F.softmax(gate_logits, dim=1)  # [B,2,C,H,W]
        gate_fre = gates[:, 0]
        gate_spa = gates[:, 1]
        fused = fre_resized * gate_fre + spa_resized * gate_spa
        fused = self.fuse_out(fused)
        return fused
