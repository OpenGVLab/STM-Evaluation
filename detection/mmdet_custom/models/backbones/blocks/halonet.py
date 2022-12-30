import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from timm.models.layers import DropPath, Mlp
from timm.models.layers.halo_attn import rel_logits_1d
from ..meta_arch import MetaArch


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class QueryFreePosEmbedRel(nn.Module):

    def __init__(self, block_size, win_size, num_heads) -> None:
        super().__init__()
        self.block_size = block_size
        self.win_size = win_size
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads))
        self.register_buffer(
            "relative_position_index",
            self._get_relative_position_index(win_size, win_size, block_size,
                                              block_size))

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.block_size**2, self.win_size * self.win_size,
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0).unsqueeze(2)

    def _get_relative_position_index(self, win_h, win_w, block_h, block_w):
        # get pair-wise relative position index for each token inside the window
        '''
        coords = torch.stack(
            torch.meshgrid(
                [torch.arange(win_h), torch.arange(win_w)],
                indexing='ij'))  # 2, Wh, Ww
        '''

        # shimin: for lower version torch, "indexing" arugment is not supported
        coords = torch.stack(
            torch.meshgrid(
                [torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww

        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] *= 2 * win_w - 1
        relative_coords = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        _sh, _sw = (win_h - block_h) // 2, (win_w - block_w) // 2
        relative_coords = relative_coords.reshape(win_h, win_w, win_h, win_w)
        relative_coords = relative_coords[_sh:_sh + block_h,
                                          _sw:_sw + block_w, :, 0:win_w]
        relative_coords = relative_coords.reshape(block_h * block_w,
                                                  win_h * win_w)
        return relative_coords.contiguous()

    def forward(self, _):
        # 1, 4, 1, 49, 169
        # 1, num_heads, 1, block_size ** 2, win_size ** 2
        return self._get_rel_pos_bias()


class QueryRelatedPosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    """

    def __init__(self, block_size, win_size, dim_head, scale):
        """
        Args:
            block_size (int): block size
            win_size (int): neighbourhood window size
            dim_head (int): attention head dim
            scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        self.height_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * scale)
        self.width_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * scale)

    def forward(self, q):
        B, NH, BB, HW, C = q.shape

        q = q.flatten(0, 1)
        B = B * NH

        # relative logits in width dimension.
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, BB, HW, -1)
        # bsz, num_heads, num_blocks ** 2, block_size ** 2, win_size ** 2
        return rel_logits.reshape(B // NH, NH, BB, HW, -1)


class HaloAttn(nn.Module):
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(self,
                 dim,
                 dim_out=None,
                 feat_size=None,
                 stride=1,
                 num_heads=8,
                 dim_head=None,
                 block_size=8,
                 halo_size=3,
                 qk_ratio=1.0,
                 qkv_bias=False,
                 avg_down=False,
                 pos_embed_type='query_free',
                 scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio,
                                                      divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = 1
        use_avg_pool = False

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.kv = nn.Conv2d(dim,
                            self.dim_out_qk + self.dim_out_v,
                            1,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.dim_out_qk, bias=qkv_bias)

        if pos_embed_type == 'query_free':
            self.pos_embed = QueryFreePosEmbedRel(block_size=self.block_size_ds,
                                                  win_size=self.win_size,
                                                  num_heads=num_heads)
        elif pos_embed_type == 'query_related':
            self.pos_embed = QueryRelatedPosEmbedRel(block_size=self.block_size,
                                                     win_size=self.win_size,
                                                     dim_head=self.dim_head_qk,
                                                     scale=self.scale)
        else:
            raise NotImplementedError(pos_embed_type)

        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()
        self.proj = nn.Linear(self.dim_out_v, self.dim_out_v)

        self.H, self.W = None, None
        self.mask = None

    def forward(self, x):
        B, H_origin, W_origin, C = x.shape

        pad_r = (self.block_size - W_origin % self.block_size) % self.block_size
        pad_b = (self.block_size - H_origin % self.block_size) % self.block_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H, W = x.shape[1], x.shape[2]

        assert H % self.block_size == 0
        assert W % self.block_size == 0
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        # unfold
        q = q.reshape(-1, num_h_blocks, self.block_size_ds, num_w_blocks,
                      self.block_size_ds, self.num_heads,
                      self.dim_head_qk).permute(0, 5, 1, 3, 2, 4, 6).contiguous()
        # B, num_heads, num_h_blocks, num_w_blocks, block_size_ds, block_size_ds, dim_head_qk
        q = q.reshape(-1, self.num_heads, num_blocks, self.block_size**2,
                      self.dim_head_qk)
        # B, num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x.permute(0, 3, 1, 2).contiguous())
        kv = F.pad(
            kv,
            [
                self.halo_size,
                self.halo_size,
                self.halo_size,
                self.halo_size,
            ],
        )
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(
            3, self.win_size,
            self.block_size).reshape(-1, self.num_heads,
                                     self.dim_head_qk + self.dim_head_v,
                                     num_blocks,
                                     self.win_size**2).permute(0, 1, 3, 4, 2).contiguous()
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        k = k.reshape(-1, self.num_heads, num_blocks, self.win_size,
                      self.win_size, self.dim_head_qk)
        v = v.reshape(-1, self.num_heads, num_blocks, self.win_size,
                      self.win_size, self.dim_head_v)
        k = k.flatten(3, 4)
        v = v.flatten(3, 4)

        if self.scale_pos_embed:
            attn = (q @ k.transpose(-1, -2) + self.pos_embed()) * self.scale
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2) + self.pos_embed(q)

        max_neg_value = -torch.finfo(attn.dtype).max
        attn.masked_fill_(self.get_mask(H, W, attn.device), max_neg_value)

        # B, num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # B, num_heads, num_blocks, block_size ** 2, dim_head_v
        # fold
        out = out.reshape(-1, self.num_heads, num_h_blocks, num_w_blocks,
                          self.block_size_ds, self.block_size_ds,
                          self.dim_head_qk)
        out = out.permute(0, 2, 4, 3, 5, 1, 6).reshape(B, H, W, self.dim_out_v).contiguous()

        out = out[:, :H_origin, :W_origin, :].contiguous()
        out = self.proj(out)
        # B, H, W, dim_out
        return out

    def get_mask(self, H, W, device):
        if self.H == H and self.W == W and self.mask is not None:
            return self.mask

        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        mask = torch.ones((1, 1, H, W), device=device)
        mask = F.pad(mask, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        mask = mask.unfold(2, self.win_size, self.block_size)
        mask = mask.unfold(3, self.win_size, self.block_size)
        mask = mask.reshape(1, num_blocks, self.win_size * self.win_size)
        mask = mask.unsqueeze(-2)

        # 1, num_blocks, 1, win_size * win_size
        mask = mask.bool()

        self.H = H
        self.W = W
        self.mask = ~mask
        return self.mask


class HaloBlockV2(nn.Module):

    def __init__(self,
                 dim,
                 drop_path,
                 layer_scale_init_value,
                 block_size,
                 halo_size,
                 stage,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 act_layer=nn.GELU,
                 pos_embed_type='query_related',
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm((dim, ))
        self.attn = HaloAttn(dim=dim,
                             dim_out=dim,
                             num_heads=num_heads[stage],
                             block_size=block_size,
                             halo_size=halo_size,
                             pos_embed_type=pos_embed_type)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm((dim, ))
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    @staticmethod
    def pre_stage_transform(x):
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def post_stage_transform(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        # shape: (B, H, W, C)
        shortcut = x
        x = self.attn(self.norm1(x))

        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x))

        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = shortcut + self.drop_path(x)

        return x


@BACKBONES.register_module()
class UnifiedHalonet(MetaArch):
    def __init__(self, *args, **kwargs):
        kwargs['block_type'] = HaloBlockV2
        super().__init__(*args, **kwargs)
