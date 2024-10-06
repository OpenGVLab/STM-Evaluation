import math
import warnings
import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.init import xavier_uniform_, constant_
from timm.models.layers import DropPath
import MultiScaleDeformableAttention as MSDA
# from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDA


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DCNv3Base(nn.Module):
    @staticmethod
    def extra_inputs(x, depths=4, deform_points=9, deform_padding=True):
        b, c, h, w = x.shape
        deform_inputs = []
        if deform_padding:
            padding = int(math.sqrt(deform_points) // 2)
        else:
            padding = int(0)

        # for i in range(sum(self.depths)):
        for i in range(depths):
            spatial_shapes = torch.as_tensor(
                [(h // pow(2, i + 2) + 2 * padding,
                    w // pow(2, i + 2) + 2 * padding)],
                dtype=torch.long, device=x.device)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)),
                    spatial_shapes.prod(1).cumsum(0)[:-1]))
            reference_points = DCNv3Base._get_reference_points(
                [(h // pow(2, i + 2) + 2 * padding,
                    w // pow(2, i + 2) + 2 * padding)],
                device=x.device, padding=padding)
            deform_inputs.append(
                [reference_points, spatial_shapes, level_start_index,
                    (h // pow(2, i + 2), w // pow(2, i + 2))])

        return deform_inputs

    @staticmethod
    def _get_reference_points(spatial_shapes, device, padding=0):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(padding + 0.5, H_ - padding - 0.5,
                               int(H_ - 2 * padding),
                               dtype=torch.float32, device=device),
                torch.linspace(padding + 0.5, W_ - padding - 0.5,
                               int(W_ - 2 * padding),
                               dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]

        return reference_points

    def forward(self, x):
        raise NotImplementedError()


class DCNv3Block(DCNv3Base):
    def __init__(self, dim, drop_path, layer_scale_init_value, stage, total_depth, num_heads,
                 kernel_size=7, deform_points=25, deform_ratio=1.0,
                 dilation_rates=(1,),  deform_padding=True, mlp_ratio=4., drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, offsets_scaler=1.0,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads[stage]
        self.mlp_ratio = mlp_ratio
        self.depth = total_depth
        self.stage = stage

        self.norm1 = norm_layer(dim)
        self.attn = MSDeformAttnGrid_softmax(
            d_model=dim, n_levels=1, n_heads=num_heads[stage],
            n_points=deform_points, ratio=deform_ratio, dilation_rates=dilation_rates,
            padding=deform_padding, dw_ks=kernel_size, offsets_scaler=offsets_scaler)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(
            dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale_init_value > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x_deform_inputs):
        def deform_forward(x):
            n, h, w, c = x.shape
            x = self.attn(
                x.reshape(n, h*w, c),
                reference_points=deform_inputs[0],
                input_flatten=None,
                input_spatial_shapes=deform_inputs[1],
                input_level_start_index=deform_inputs[2],
                input_padding_mask=None).reshape(n, h, w, c)

            return x
        # print(len(x_deform_inputs))
        x = x_deform_inputs[0]
        #deform_inputs = x_deform_inputs[1][self.depth]
        deform_inputs = x_deform_inputs[1]

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = deform_forward(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return (x.permute(0, 3, 1, 2), x_deform_inputs[1])

        shortcut = x
        x = self.norm1(x)
        x = deform_forward(x)
        # x = checkpoint(deform_forward, x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return (x.permute(0, 3, 1, 2), x_deform_inputs[1])  # the returned value will be passed to the next block


class DCNv3SingleResBlock(DCNv3Base):
    def __init__(self, dim, drop_path, layer_scale_init_value, stage, total_depth, num_heads,
                 kernel_size=7, deform_points=25, deform_ratio=1.0,
                 dilation_rates=(1,),  deform_padding=True, mlp_ratio=4., drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, offsets_scaler=1.0,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads[stage]
        self.mlp_ratio = mlp_ratio
        self.depth = total_depth
        self.stage = stage

        self.attn = MSDeformAttnGrid_softmax(
            d_model=dim, n_levels=1, n_heads=num_heads[stage],
            n_points=deform_points, ratio=deform_ratio, dilation_rates=dilation_rates,
            padding=deform_padding, dw_ks=kernel_size, offsets_scaler=offsets_scaler)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(
            dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale_init_value > 0:
            self.layer_scale = True
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x_deform_inputs):
        def deform_forward(x):
            n, h, w, c = x.shape
            x = self.attn(
                x.reshape(n, h*w, c),
                reference_points=deform_inputs[0],
                input_flatten=None,
                input_spatial_shapes=deform_inputs[1],
                input_level_start_index=deform_inputs[2],
                input_padding_mask=None).reshape(n, h, w, c)

            return x
        # print(len(x_deform_inputs))
        x = x_deform_inputs[0]
        #deform_inputs = x_deform_inputs[1][self.depth]
        deform_inputs = x_deform_inputs[1]

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        if not self.layer_scale:
            shortcut = x
            x = deform_forward(x)
            x = self.mlp(self.norm(x))

            x = shortcut + self.drop_path(x)
            return (x.permute(0, 3, 1, 2), x_deform_inputs[1])

        shortcut = x
        x = deform_forward(x)
        x = self.gamma * self.mlp(self.norm(x))

        x = shortcut + self.drop_path(x)
        return (x.permute(0, 3, 1, 2), x_deform_inputs[1])  # the returned value will be passed to the next block


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class MSDeformAttnGrid_softmax(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0, dilation_rates=[1],
                 init_ratio=1.0, padding=True, dw_ks=7, offsets_scaler=1.0):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(
                    d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        assert n_heads % len(dilation_rates) == 0

        self.im2col_step = 128
        self.offsets_scaler = offsets_scaler
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.init_ratio = init_ratio
        self.padding = padding

        self.dw_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, dw_ks, 1,
                      (dw_ks-1)//2, groups=d_model),
            LayerNormProxy(self.d_model),
            nn.GELU())
        self.dilation_rates = dilation_rates
        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)
        self._generate_dilation_grids()
        self._reset_parameters()

    def _generate_dilation_grids(self):
        dilation_rates = self.dilation_rates
        n_heads = self.n_heads
        n_points = self.n_points
        points_list = []
        for rate in dilation_rates:
            kernel_size = int(math.sqrt(n_points))
            y, x = torch.meshgrid(
                torch.linspace((-kernel_size // 2 + 1) * rate, (kernel_size // 2) * rate, kernel_size,
                               dtype=torch.float32),
                torch.linspace((-kernel_size // 2 + 1) * rate, (kernel_size // 2) * rate, kernel_size,
                               dtype=torch.float32))
            points_list.extend([y, x])
        grid = torch.stack(points_list, -1).reshape(-1, len(dilation_rates), 2).\
            repeat(1, n_heads//len(dilation_rates), 1).permute(1, 0, 2)
        self.grid = grid.reshape(1, 1, n_heads, 1, n_points, 2)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.sampling_offsets.bias.data, 0.)
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        device = query.device

        N, Len_q, C = query.shape
        pad_width = int(math.sqrt(self.n_points) // 2)
        H, W = input_spatial_shapes[:, 0], input_spatial_shapes[:, 1]
        H = H - 2 * pad_width
        W = W - 2 * pad_width

        assert input_flatten == None
        if input_flatten == None:
            input_flatten = query
        N, Len_in, C = input_flatten.shape

        query = query.permute(0, 2, 1).reshape(N, C, H, W)
        query = self.dw_conv(query).permute(
            0, 2, 3, 1).reshape(N, Len_q, C)

        # (N, Len-in, d_model)
        value = self.value_proj(input_flatten)

        # padding
        if self.padding:
            value = value.reshape(N, H, W, C)
            value = F.pad(
                value, [0, 0, pad_width, pad_width, pad_width, pad_width])
            value = value.reshape(N, -1, C)
            Len_in = value.size(1)  # update Len_in

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)
        value = value.reshape(N, Len_in, self.n_heads, int(
            self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).reshape(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            N, Len_q,
            self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).reshape(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points)
        self.grid = self.grid.to(device)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:  # 1-stage
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                (self.grid + sampling_offsets) * self.offsets_scaler / \
                offset_normalizer[None, None, None, :, None, :]
        # elif reference_points.shape[-1] == 4:  # 2-stage
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #         + sampling_offsets / self.n_points * \
        #         reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations.contiguous(), attention_weights, self.im2col_step)
        output = self.output_proj(output)

        return output


class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
