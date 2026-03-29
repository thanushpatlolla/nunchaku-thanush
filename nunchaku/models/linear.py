"""
Quantized linear layers for Nunchaku.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ..lora.flux.nunchaku_converter import unpack_lowrank_weight

class SVDQW4A8Linear(nn.Module):
    """
    fake quantization for experimentation. 
    this linear class will get swapped in after weights are loaded in
    NunchakuFluxTransformer2DModelV2
    """
    def __init__(
        self,
        qweight: torch.Tensor,
        wscales: torch.Tensor,
        smooth_factor: torch.Tensor,
        proj_down: torch.Tensor,
        proj_up: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        in_features: int,
        out_features: int,
        act_unsigned: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.act_unsigned = act_unsigned
        self.register_buffer("smooth_factor", smooth_factor)
        self.register_buffer("proj_down", unpack_lowrank_weight(proj_down, down=True).T)
        self.register_buffer("proj_up", unpack_lowrank_weight(proj_up, down=False))
        self.register_buffer("bias", bias)
        self.register_buffer("w_bf16", self._dequantize_weights(qweight, wscales))
      
    def _dequantize_weights(self, qweight, wscales):
        #constants for bits=4, warp_n=128 in NunchakuWeightPacker.pack_weight from packer.py
        #we have to reverse the packer basically, thats what the views and permutes and
        #shifts do
        mem_n = 128
        mem_k = 64 
        num_n_packs = 8
        n_pack_size = 2
        num_n_lanes = 8
        reg_n = 1
        num_k_packs = 1
        k_pack_size = 2
        num_k_lanes = 4
        reg_k = 8
        n_tiles, k_tiles = self.out_features // mem_n, self.in_features // mem_k
        
        w32 = qweight.contiguous().view(self.out_features, -1).view(dtype=torch.int32)
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
        nibbles = (w32.unsqueeze(-1) >> shifts) & 0xF
        
        nibbles = nibbles.reshape(
            n_tiles, 
            k_tiles, 
            num_k_packs, 
            num_n_packs, 
            num_n_lanes,
            num_k_lanes, 
            n_pack_size, 
            k_pack_size, 
            reg_n, 
            reg_k,
        )
        
        nibbles = nibbles.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
        nibbles = nibbles.view(self.out_features, -1)
        nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)
        nibbles = nibbles.to(torch.bfloat16)
        
        num_groups = self.in_features // self.group_size
        
        #using the warp_n=128 values for the constants below from pack_scale in packer.py
        warp_s = 128
        num_s_packs = 1
        num_s_lanes = 32
        s_pack_size = 4
        scale = wscales.reshape(
            self.out_features // warp_s, -1, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2
        )
        scale = scale.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
        scale = scale.view(self.out_features, num_groups)

        w_bf16 = nibbles.reshape(self.out_features, num_groups, self.group_size) * scale.unsqueeze(-1)
        
        return w_bf16.reshape(self.out_features, self.in_features)
    
    def forward(self, x):
        in_shape = x.shape
        lora = x @ self.proj_down @ self.proj_up.T

        x = x / self.smooth_factor
        x = x.reshape(-1, self.in_features)

        amax = x.abs().amax()
        scale = 1.0 if amax == 0 else amax / 127
        x = torch.round(x / scale).clamp(-128, 127) * scale

        out = lora + (x @ self.w_bf16.T).view(*in_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out

    @classmethod
    def from_svdq_linear(cls, layer: "SVDQW4A4Linear"):
        return cls(
            qweight=layer.qweight.data,
            wscales=layer.wscales.data,
            smooth_factor=layer.smooth_factor.data,
            proj_down=layer.proj_down.data,
            proj_up=layer.proj_up.data,
            bias=layer.bias.data if layer.bias is not None else None,
            group_size=layer.group_size,
            in_features=layer.in_features,
            out_features=layer.out_features,
            act_unsigned=layer.act_unsigned,
        )


class FakeQuantFluxAttnProcessor:
    """PyTorch attention processor for SVDQW4A8Linear layers.

    Replaces NunchakuFluxFA2Processor when fused CUDA kernels cannot be used.
    """

    @staticmethod
    def _unpack_rotemb(packed):
        """Inverse of pack_rotemb from nunchaku.models.embeddings.

        packed: [B, M, D] (MMA-packed rotary embeddings)
        returns: [B, M, D//2, 1, 2] where [..., 0]=sin, [..., 1]=cos
        """
        B, M, D = packed.shape
        x = packed.view(B, M // 16, D // 8, 8, 4, 2, 2)
        x = x.permute(0, 1, 2, 5, 3, 4, 6)
        x = x.reshape(B, M // 16, D // 8, 16, 8)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, M, D // 2, 1, 2)
        return x

    @staticmethod
    def _apply_rotary(x, rotemb):
        """Apply rotary embedding from nunchaku raw format.

        x: [B, S, heads, head_dim]
        rotemb: [B_emb, S_emb, head_dim//2, 1, 2]  (sin=0, cos=1)
        """
        S = x.shape[1]
        cos = rotemb[0, :S, :, 0, 1]  # [S, head_dim//2]
        sin = rotemb[0, :S, :, 0, 0]  # [S, head_dim//2]

        x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D/2]
        cos = cos[None, :, None, :]  # [1, S, 1, D/2]
        sin = sin[None, :, None, :]
        out_r = x_r.float() * cos - x_i.float() * sin
        out_i = x_r.float() * sin + x_i.float() * cos
        return torch.stack([out_r, out_i], dim=-1).flatten(-2).to(x.dtype)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        B, S, C = hidden_states.shape

        qkv = attn.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, attn.heads, attn.head_dim)
        k = k.view(B, S, attn.heads, attn.head_dim)
        v = v.view(B, S, attn.heads, attn.head_dim)
        q = attn.norm_q(q)
        k = attn.norm_k(k)

        if isinstance(image_rotary_emb, tuple):
            rotemb_img = self._unpack_rotemb(image_rotary_emb[0])
        elif image_rotary_emb is not None:
            rotemb_img = self._unpack_rotemb(image_rotary_emb)
        else:
            rotemb_img = None

        if rotemb_img is not None:
            q = self._apply_rotary(q, rotemb_img)
            k = self._apply_rotary(k, rotemb_img)

        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            S_ctx = encoder_hidden_states.shape[1]
            qkv_ctx = attn.add_qkv_proj(encoder_hidden_states)
            q_c, k_c, v_c = qkv_ctx.chunk(3, dim=-1)
            q_c = q_c.view(B, S_ctx, attn.heads, attn.head_dim)
            k_c = k_c.view(B, S_ctx, attn.heads, attn.head_dim)
            v_c = v_c.view(B, S_ctx, attn.heads, attn.head_dim)
            q_c = attn.norm_added_q(q_c)
            k_c = attn.norm_added_k(k_c)

            rotemb_txt = self._unpack_rotemb(image_rotary_emb[1])
            q_c = self._apply_rotary(q_c, rotemb_txt)
            k_c = self._apply_rotary(k_c, rotemb_txt)

            q = torch.cat([q_c, q], dim=1)
            k = torch.cat([k_c, k], dim=1)
            v = torch.cat([v_c, v], dim=1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = out.to(q.dtype)

        if encoder_hidden_states is not None:
            S_ctx = encoder_hidden_states.shape[1]
            enc_out, img_out = out[:, :S_ctx], out[:, S_ctx:]
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_out)
            return img_out, enc_out
        else:
            return attn.to_out(out)


def replace_with_fake_quant(model: nn.Module) -> nn.Module:
    """Replace SVDQW4A4Linear layers with fake-quantized W4A8 and patch fused ops.

    Call after from_pretrained so weights are already loaded. Builds replacement
    layers on the same device as the source. Caller should handle device
    placement beforehand (e.g. pipeline.to("cpu") then
    pipeline.enable_sequential_cpu_offload() after).
    """
    from .transformers.transformer_flux_v2 import NunchakuFluxAttention

    for name, child in model.named_children():
        if isinstance(child, SVDQW4A4Linear):
            setattr(model, name, SVDQW4A8Linear.from_svdq_linear(child))
        else:
            replace_with_fake_quant(child)

    if isinstance(model, NunchakuFluxAttention):
        model.processor = FakeQuantFluxAttnProcessor()

    return model


class SVDQW4A4Linear(nn.Module):
    """
    `SVDQuant <paper_svdquant_>`_ W4A4 quantized linear layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    rank : int, optional
        SVD low-rank dimension. Default is 32.
    bias : bool, optional
        If True, adds a learnable bias. Default is True.
    precision : {'int4', 'nvfp4'}, optional
        Quantization precision data type ('int4' or 'nvfp4'). Default is 'int4'.
    act_unsigned : bool, optional
        If True, use unsigned activation quantization (int4 only). Default is False.
    torch_dtype : torch.dtype, optional
        Parameter dtype. Default is torch.bfloat16.
    device : str or torch.device or None, optional
        Device for parameters. Default is CPU.

    Attributes
    ----------
    in_features : int
    out_features : int
    rank : int
    precision : str
        'int4' or 'nvfp4'.
    group_size : int
        64 for int4, 16 for nvfp4.
    qweight : nn.Parameter
        Packed quantized weights, shape (out_features, in_features // 2), dtype int8.
    bias : nn.Parameter or None
        Bias tensor.
    wscales : nn.Parameter
        Weight scales, shape (in_features // group_size, out_features).
        Dtype: bfloat16/float16 (int4), float8_e4m3fn (nvfp4).
    smooth_factor : nn.Parameter
        Smoothing factors, shape (in_features,).
    smooth_factor_orig : nn.Parameter
        Original smoothing factors, shape (in_features,). (Unused)
    proj_down : nn.Parameter
        Packed low-rank down projection, shape (in_features, rank), dtype bfloat16/float16.
    proj_up : nn.Parameter
        Packed low-rank up projection, shape (out_features, rank), dtype bfloat16/float16.
    wtscale : float or None
        Global weight scale (nvfp4 only).
    wcscales : nn.Parameter or None
        Channel-wise weight scale (nvfp4 only), shape (out_features,), dtype float8_e4m3fn.
    act_unsigned : bool
        If True, input activations are unsigned (int4 only).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(SVDQW4A4Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.precision = precision
        self.torch_dtype = torch_dtype

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )

        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )

        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
            )
            self.wtscale = 1.0
        else:
            self.wtscale = None
            self.wcscales = None

        self.act_unsigned = act_unsigned

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        """
        Create an SVDQW4A4Linear from a standard nn.Linear. The weight and bias are dummy tensors.

        Parameters
        ----------
        linear : nn.Linear
            Source linear layer.
        **kwargs
            Additional init arguments.

        Returns
        -------
        SVDQW4A4Linear
        """
        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=linear.weight.device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with 16-bit input. It will call :meth:`quantize` and :meth:`forward_quant`.

        Parameters
        ----------
        x : torch.Tensor, shape (B, S, in_features), dtype float16 or bfloat16
            Input tensor.
        output : torch.Tensor or None, optional
            Optional output buffer.

        Returns
        -------
        torch.Tensor, shape (B, S, out_features)
            Output tensor.

        Notes
        -----
        B: batch size, S: sequence length
        """
        batch_size, seq_len, channels = x.shape
        x = x.reshape(batch_size * seq_len, channels)
        if output is None:
            output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        output = output.reshape(batch_size, seq_len, -1)
        return output

    def quantize(self, x: torch.Tensor, pad_size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input to 4-bit and compute low-rank hidden states. It will call :func:`~nunchaku.ops.quantize.svdq_quantize_w4a4_act_fuse_lora_cuda`.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_features), dtype float16 or bfloat16
            Input tensor.
        pad_size : int, optional
            Batch padding size. Default is 256.

        Returns
        -------
        quantized_x : torch.Tensor
            Quantized input, shape (pad_size * ceil(N / pad_size), in_features // 2), dtype uint8.
        ascales : torch.Tensor
            Activation scales, shape (in_features // group_size,), dtype float8_e4m3fn for nvfp4 and input dtype for int4.
        lora_act_out : torch.Tensor
            Low-rank hidden states, shape (pad_size * ceil(N / pad_size), rank), dtype float32.

        Notes
        -----
        N: batch size
        """
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x, lora_down=self.proj_down, smooth=self.smooth_factor, fp4=self.precision == "nvfp4", pad_size=pad_size
        )
        return quantized_x, ascales, lora_act_out

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-quantized input. It will call :func:`~nunchaku.ops.gemm.svdq_gemm_w4a4_cuda`.

        Parameters
        ----------
        quantized_x : torch.Tensor
            Quantized input, shape (N, in_features // 2), dtype uint8.
        ascales : torch.Tensor
            Activation scales, shape (in_features // group_size,), dtype float8_e4m3fn for nvfp4 and input dtype for int4.
        lora_act : torch.Tensor
            Low-rank hidden states, shape (N, rank), dtype float32.
        output : torch.Tensor or None, optional
            Optional output buffer.

        Returns
        -------
        torch.Tensor
            Output tensor, shape (N, out_features), dtype bfloat16/float16 for int4 and float8_e4m3fn for nvfp4.

        Notes
        -----
        N: batch size
        """
        if output is None:
            output = torch.empty(
                quantized_x.shape[0], self.out_features, dtype=self.proj_up.dtype, device=quantized_x.device
            )

        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        return output

    def __repr__(self):
        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


class AWQW4A16Linear(nn.Module):
    """
    `AWQ <paper_awq_>`_ W4A16 quantized linear layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool, optional
        If True, adds learnable bias. Default is True.
    group_size : int, optional
        Quantization group size. Default is 64.
    torch_dtype : torch.dtype, optional
        Parameter dtype. Default is torch.bfloat16.
    device : str or torch.device or None, optional
        Device for parameters. Default is CPU.

    Attributes
    ----------
    in_features : int
    out_features : int
    group_size : int
    qweight : nn.Parameter
        Packed quantized weights, shape (out_features // 4, in_features // 2), dtype int32.
    bias : nn.Parameter or None
        Bias tensor.
    wscales : nn.Parameter
        Weight scales, shape (in_features // group_size, out_features), dtype float16 or bfloat16.
    wzeros : nn.Parameter
        Weight zero points, shape (in_features // group_size, out_features), dtype float16 or bfloat16.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(AWQW4A16Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(out_features // 4, in_features // 2, dtype=torch.int32, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AWQW4A16Linear.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_features)
            Input tensor.

        Returns
        -------
        torch.Tensor, shape (N, out_features)
            Output tensor.

        Notes
        -----
        N: batch size
        """
        output = awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self.qweight,
            scaling_factors=self.wscales,
            zeros=self.wzeros,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )
        if self.bias is not None:
            view_shape = [1] * (output.ndim - 1) + [-1]
            output.add_(self.bias.view(view_shape))
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Create an uninitialized AWQW4A16Linear from a standard nn.Linear.

        Parameters
        ----------
        linear : nn.Linear
            Source linear layer.
        group_size : int, optional
            Quantization group size.
        torch_dtype : torch.dtype, optional
            Parameter dtype.
        device : str, optional
            Device for parameters.

        Returns
        -------
        AWQW4A16Linear
        """
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def __repr__(self):
        return f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size})"
