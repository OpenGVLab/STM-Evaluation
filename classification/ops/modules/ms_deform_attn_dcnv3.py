# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import einops
from timm.models.layers import trunc_normal_

from ..functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch


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
        self.grid = grid.view(1, 1, n_heads, 1, n_points, 2)

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

        query = query.permute(0, 2, 1).view(N, C, H, W).contiguous()
        query = self.dw_conv(query).permute(
            0, 2, 3, 1).view(N, Len_q, C).contiguous()

        # (N, Len-in, d_model)
        value = self.value_proj(input_flatten)

        # padding
        if self.padding:
            value = value.reshape(N, H, W, C)
            value = F.pad(
                value, [0, 0, pad_width, pad_width, pad_width, pad_width])
            value = value.reshape(N, -1, C).contiguous()
            Len_in = value.size(1)  # update Len_in

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)
        value = value.view(N, Len_in, self.n_heads, int(
            self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
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


class MSDeformAttnGrid_softmax_v1(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=9, ratio=1.0, dilation_rates=[1],
                 init_ratio=1.0, padding=True, dw_ks=3, offsets_scaler=1.0):
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
        self.grid = grid.view(1, 1, n_heads, 1, n_points, 2)

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

        query = query.permute(0, 2, 1).view(N, C, H, W).contiguous()
        query = self.dw_conv(query).permute(
            0, 2, 3, 1).view(N, Len_q, C).contiguous()

        # (N, Len-in, d_model)
        value = self.value_proj(input_flatten)

        # padding
        if self.padding:
            value = value.reshape(N, H, W, C)
            value = F.pad(
                value, [0, 0, pad_width, pad_width, pad_width, pad_width])
            value = value.reshape(N, -1, C).contiguous()
            Len_in = value.size(1)  # update Len_in

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)
        value = value.view(N, Len_in, self.n_heads, int(
            self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
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
                (self.grid / offset_normalizer[None, None, None, :, None, :] + 
                (sampling_offsets / 4).clamp(-1, 1)) * self.offsets_scaler
                
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


class MSDeformAttnGrid_softmax_a(nn.Module):
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
        self.alpha = nn.Parameter(torch.ones(n_heads), requires_grad=True)
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
        self.grid = grid.view(1, 1, n_heads, 1, n_points, 2)

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

        query = query.permute(0, 2, 1).view(N, C, H, W).contiguous()
        query = self.dw_conv(query).permute(
            0, 2, 3, 1).view(N, Len_q, C).contiguous()

        # (N, Len-in, d_model)
        value = self.value_proj(input_flatten)

        # padding
        if self.padding:
            value = value.reshape(N, H, W, C)
            value = F.pad(
                value, [0, 0, pad_width, pad_width, pad_width, pad_width])
            value = value.reshape(N, -1, C).contiguous()
            Len_in = value.size(1)  # update Len_in

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)
        value = value.view(N, Len_in, self.n_heads, int(
            self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
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
                (self.grid * self.alpha[None, None, :, None, None, None] + sampling_offsets) * \
                self.offsets_scaler / offset_normalizer[None, None, None, :, None, :]
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


class MSDeformAttnGrid_softmax_ab(nn.Module):
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
        self.alpha_beta = nn.Parameter(torch.ones(n_heads, 2), requires_grad=True)
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
        self.grid = grid.view(1, 1, n_heads, 1, n_points, 2)

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

        query = query.permute(0, 2, 1).view(N, C, H, W).contiguous()
        query = self.dw_conv(query).permute(
            0, 2, 3, 1).view(N, Len_q, C).contiguous()

        # (N, Len-in, d_model)
        value = self.value_proj(input_flatten)

        # padding
        if self.padding:
            value = value.reshape(N, H, W, C)
            value = F.pad(
                value, [0, 0, pad_width, pad_width, pad_width, pad_width])
            value = value.reshape(N, -1, C).contiguous()
            Len_in = value.size(1)  # update Len_in

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, 8, 64)
        value = value.view(N, Len_in, self.n_heads, int(
            self.ratio * self.d_model) // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q,
            self.n_heads,
            self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
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
                (self.grid * self.alpha_beta[None, None, :, None, None, :] + sampling_offsets) * \
                self.offsets_scaler / offset_normalizer[None, None, None, :, None, :]
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

