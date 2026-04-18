import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GridConfig:
    nx: int = 172
    ny: int = 79
    lx: float = 0.260
    ly: float = 0.120
    nu: float = 1e-4
    u_inlet: float = 0.1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, use_bn: bool = False):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, use_bn: bool = False):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size, use_bn)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel_size, use_bn)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        size = x.size()
        x_pooled, indices = self.pool(x)
        return x_pooled, x, indices, size


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, use_bn: bool = False):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = ConvBlock(in_ch * 2, in_ch, kernel_size, use_bn)
        self.conv2 = ConvBlock(in_ch, out_ch, kernel_size, use_bn)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, indices: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = self.unpool(x, indices, output_size=size)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(self.w_g(g_up) + self.w_x(x)))
        return x * alpha


class DecoderBlockAttn(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.attn = AttentionGate(f_g=in_ch, f_l=in_ch, f_int=max(1, in_ch // 2))
        self.conv1 = ConvBlock(in_ch * 2, in_ch, kernel_size)
        self.conv2 = ConvBlock(in_ch, out_ch, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, indices: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x_up = self.unpool(x, indices, output_size=size)
        skip_attn = self.attn(x_up, skip)
        x = torch.cat([skip_attn, x_up], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        bottleneck_hw: Tuple[int, int] = (10, 4),
    ):
        super().__init__()
        self.embed = PatchEmbedding(channels, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.proj_back = nn.Linear(embed_dim, channels)
        self.norm = nn.LayerNorm(channels)

        seq_len = bottleneck_hw[0] * bottleneck_hw[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens, h, w = self.embed(x)
        n_tokens = h * w
        if n_tokens != self.pos_embed.shape[1]:
            raise ValueError(
                f"Token mismatch: expected {self.pos_embed.shape[1]}, got {n_tokens}. "
                "Check input_hw and pooling depth."
            )
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.proj_back(tokens)
        tokens = self.norm(tokens)
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return out


class UNetBase(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, kernel_size: int = 5, filters=None):
        super().__init__()
        if filters is None:
            filters = [8, 16, 32, 64]
        self.name = "UNet-Base"

        self.enc1 = EncoderBlock(in_channels, filters[0], kernel_size)
        self.enc2 = EncoderBlock(filters[0], filters[1], kernel_size)
        self.enc3 = EncoderBlock(filters[1], filters[2], kernel_size)
        self.enc4 = EncoderBlock(filters[2], filters[3], kernel_size)

        self.bottleneck = nn.Sequential(
            ConvBlock(filters[3], filters[3], kernel_size),
            ConvBlock(filters[3], filters[3], kernel_size),
        )

        self.decoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderBlock(filters[3], filters[2], kernel_size),
                        DecoderBlock(filters[2], filters[1], kernel_size),
                        DecoderBlock(filters[1], filters[0], kernel_size),
                        DecoderBlock(filters[0], filters[0], kernel_size),
                        nn.Conv2d(filters[0], 1, kernel_size=1),
                    ]
                )
                for _ in range(out_channels)
            ]
        )

    def _encode(self, x: torch.Tensor):
        x1, s1, i1, sz1 = self.enc1(x)
        x2, s2, i2, sz2 = self.enc2(x1)
        x3, s3, i3, sz3 = self.enc3(x2)
        x4, s4, i4, sz4 = self.enc4(x3)
        z = self.bottleneck(x4)
        skips = [(s1, i1, sz1), (s2, i2, sz2), (s3, i3, sz3), (s4, i4, sz4)]
        return z, skips

    def _decode(self, z: torch.Tensor, skips, decoder) -> torch.Tensor:
        s1, i1, sz1 = skips[0]
        s2, i2, sz2 = skips[1]
        s3, i3, sz3 = skips[2]
        s4, i4, sz4 = skips[3]
        d = decoder[0](z, s4, i4, sz4)
        d = decoder[1](d, s3, i3, sz3)
        d = decoder[2](d, s2, i2, sz2)
        d = decoder[3](d, s1, i1, sz1)
        d = decoder[4](d)
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skips = self._encode(x)
        outputs = [self._decode(z, skips, dec) for dec in self.decoders]
        return torch.cat(outputs, dim=1)


class UNetAttention(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, kernel_size: int = 5, filters=None):
        super().__init__()
        if filters is None:
            filters = [8, 16, 32, 64]
        self.name = "UNet-Attention"

        self.enc1 = EncoderBlock(in_channels, filters[0], kernel_size)
        self.enc2 = EncoderBlock(filters[0], filters[1], kernel_size)
        self.enc3 = EncoderBlock(filters[1], filters[2], kernel_size)
        self.enc4 = EncoderBlock(filters[2], filters[3], kernel_size)

        self.bottleneck = nn.Sequential(
            ConvBlock(filters[3], filters[3], kernel_size),
            ConvBlock(filters[3], filters[3], kernel_size),
        )

        self.decoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderBlockAttn(filters[3], filters[2], kernel_size),
                        DecoderBlockAttn(filters[2], filters[1], kernel_size),
                        DecoderBlockAttn(filters[1], filters[0], kernel_size),
                        DecoderBlockAttn(filters[0], filters[0], kernel_size),
                        nn.Conv2d(filters[0], 1, kernel_size=1),
                    ]
                )
                for _ in range(out_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, s1, i1, sz1 = self.enc1(x)
        x2, s2, i2, sz2 = self.enc2(x1)
        x3, s3, i3, sz3 = self.enc3(x2)
        x4, s4, i4, sz4 = self.enc4(x3)
        z = self.bottleneck(x4)

        outputs = []
        for dec in self.decoders:
            d = dec[0](z, s4, i4, sz4)
            d = dec[1](d, s3, i3, sz3)
            d = dec[2](d, s2, i2, sz2)
            d = dec[3](d, s1, i1, sz1)
            d = dec[4](d)
            outputs.append(d)

        return torch.cat(outputs, dim=1)


class UNetTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 5,
        filters=None,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        input_hw: Tuple[int, int] = (172, 79),
    ):
        super().__init__()
        if filters is None:
            filters = [8, 16, 32, 64]
        self.name = "UNet-Transformer"

        self.enc1 = EncoderBlock(in_channels, filters[0], kernel_size)
        self.enc2 = EncoderBlock(filters[0], filters[1], kernel_size)
        self.enc3 = EncoderBlock(filters[1], filters[2], kernel_size)
        self.enc4 = EncoderBlock(filters[2], filters[3], kernel_size)

        # Light pre-transform refinement to stabilize tokenization without changing architecture semantics.
        self.pre_transform = nn.Sequential(
            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.GELU(),
        )
        self.pre_transform_gain = nn.Parameter(torch.tensor(0.10))

        bottleneck_h = input_hw[0] // 16
        bottleneck_w = input_hw[1] // 16

        self.bottleneck = TransformerBottleneck(
            channels=filters[3],
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            bottleneck_hw=(bottleneck_h, bottleneck_w),
        )

        self.decoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderBlockAttn(filters[3], filters[2], kernel_size),
                        DecoderBlockAttn(filters[2], filters[1], kernel_size),
                        DecoderBlockAttn(filters[1], filters[0], kernel_size),
                        DecoderBlockAttn(filters[0], filters[0], kernel_size),
                        nn.Conv2d(filters[0], 1, kernel_size=1),
                    ]
                )
                for _ in range(out_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, s1, i1, sz1 = self.enc1(x)
        x2, s2, i2, sz2 = self.enc2(x1)
        x3, s3, i3, sz3 = self.enc3(x2)
        x4, s4, i4, sz4 = self.enc4(x3)

        # Keep the original transformer bottleneck but feed a softly refined representation.
        x4_refined = x4 + torch.tanh(self.pre_transform_gain) * self.pre_transform(x4)
        z = self.bottleneck(x4_refined)

        outputs = []
        for dec in self.decoders:
            d = dec[0](z, s4, i4, sz4)
            d = dec[1](d, s3, i3, sz3)
            d = dec[2](d, s2, i2, sz2)
            d = dec[3](d, s1, i1, sz1)
            d = dec[4](d)
            outputs.append(d)

        return torch.cat(outputs, dim=1)


class PhysicsInformedLossV2(nn.Module):
    """
    Improved PINN residual module for DeepCFD:
    - finite-difference residuals via fixed conv kernels
    - robust interior masking (2-cell margin)
    - scaled residuals to reduce term imbalance
    - richer outlet conditions: dUx/dx=0, dUy/dx=0, dp/dx=0
    - pressure gauge term to remove null-space drift
    """

    def __init__(self, grid: GridConfig = GridConfig()):
        super().__init__()
        self.grid = grid
        dx = grid.lx / (grid.nx - 1)
        dy = grid.ly / (grid.ny - 1)

        k_dx = torch.zeros(1, 1, 3, 1)
        k_dx[0, 0, 0, 0] = -1.0 / (2.0 * dx)
        k_dx[0, 0, 2, 0] = 1.0 / (2.0 * dx)
        self.register_buffer("kernel_dx", k_dx)

        k_dy = torch.zeros(1, 1, 1, 3)
        k_dy[0, 0, 0, 0] = -1.0 / (2.0 * dy)
        k_dy[0, 0, 0, 2] = 1.0 / (2.0 * dy)
        self.register_buffer("kernel_dy", k_dy)

        k_d2x = torch.zeros(1, 1, 3, 1)
        k_d2x[0, 0, 0, 0] = 1.0 / (dx**2)
        k_d2x[0, 0, 1, 0] = -2.0 / (dx**2)
        k_d2x[0, 0, 2, 0] = 1.0 / (dx**2)
        self.register_buffer("kernel_d2x", k_d2x)

        k_d2y = torch.zeros(1, 1, 1, 3)
        k_d2y[0, 0, 0, 0] = 1.0 / (dy**2)
        k_d2y[0, 0, 0, 1] = -2.0 / (dy**2)
        k_d2y[0, 0, 0, 2] = 1.0 / (dy**2)
        self.register_buffer("kernel_d2y", k_d2y)

    def _deriv(self, f: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kh, kw = kernel.shape[-2], kernel.shape[-1]
        pad_h = kh // 2
        pad_w = kw // 2
        f_padded = F.pad(f, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        return F.conv2d(f_padded, kernel, padding=0)

    def ddx(self, f: torch.Tensor) -> torch.Tensor:
        return self._deriv(f, self.kernel_dx)

    def ddy(self, f: torch.Tensor) -> torch.Tensor:
        return self._deriv(f, self.kernel_dy)

    def d2dx2(self, f: torch.Tensor) -> torch.Tensor:
        return self._deriv(f, self.kernel_d2x)

    def d2dy2(self, f: torch.Tensor) -> torch.Tensor:
        return self._deriv(f, self.kernel_d2y)

    def compute_masks(self, x_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        bc = x_input[:, 1:2, :, :]

        fluid_mask = (bc > 0.5) & (bc < 1.5)
        wall_mask = (bc > 1.5) & (bc < 2.5)
        obstacle_mask = (bc > -0.5) & (bc < 0.5)
        inlet_mask = (bc > 2.5) & (bc < 3.5)
        outlet_mask = (bc > 3.5) & (bc < 4.5)
        noslip_mask = wall_mask | obstacle_mask

        interior = fluid_mask.clone()
        interior[:, :, :2, :] = False
        interior[:, :, -2:, :] = False
        interior[:, :, :, :2] = False
        interior[:, :, :, -2:] = False

        return {
            "fluid": fluid_mask.float(),
            "interior": interior.float(),
            "noslip": noslip_mask.float(),
            "inlet": inlet_mask.float(),
            "outlet": outlet_mask.float(),
        }

    @staticmethod
    def masked_mse(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x.square() * mask).sum() / (mask.sum() + eps)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x * mask).sum() / (mask.sum() + eps)

    def forward(self, output: torch.Tensor, x_input: torch.Tensor, weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        ux = output[:, 0:1]
        uy = output[:, 1:2]
        p = output[:, 2:3]

        masks = self.compute_masks(x_input)
        fluid = masks["fluid"]
        interior = masks["interior"]

        dux_dx = self.ddx(ux)
        dux_dy = self.ddy(ux)
        duy_dx = self.ddx(uy)
        duy_dy = self.ddy(uy)
        dp_dx = self.ddx(p)
        dp_dy = self.ddy(p)

        lap_ux = self.d2dx2(ux) + self.d2dy2(ux)
        lap_uy = self.d2dx2(uy) + self.d2dy2(uy)

        div = dux_dx + duy_dy
        res_x = ux * dux_dx + uy * dux_dy + dp_dx - self.grid.nu * lap_ux
        res_y = ux * duy_dx + uy * duy_dy + dp_dy - self.grid.nu * lap_uy

        u_scale = torch.sqrt(self.masked_mean(ux.square() + uy.square(), fluid) + 1e-8)
        p_scale = torch.sqrt(self.masked_mean(p.square(), fluid) + 1e-8)
        l_ref = max(self.grid.lx, self.grid.ly)

        div_scale = (u_scale / l_ref).detach() + 1e-8
        mom_scale = ((u_scale.square() + p_scale) / l_ref).detach() + 1e-8

        cont_loss = self.masked_mse(div / div_scale, interior)
        mom_x_loss = self.masked_mse(res_x / mom_scale, interior)
        mom_y_loss = self.masked_mse(res_y / mom_scale, interior)
        mom_loss = mom_x_loss + mom_y_loss

        noslip = masks["noslip"]
        inlet = masks["inlet"]
        outlet = masks["outlet"]

        bc_loss = 0.0
        if noslip.sum() > 0:
            bc_loss = bc_loss + self.masked_mse(ux, noslip) + self.masked_mse(uy, noslip)
        if inlet.sum() > 0:
            bc_loss = bc_loss + self.masked_mse(ux - self.grid.u_inlet, inlet) + self.masked_mse(uy, inlet)
        if outlet.sum() > 0:
            bc_loss = (
                bc_loss
                + self.masked_mse(dux_dx, outlet)
                + self.masked_mse(duy_dx, outlet)
                + self.masked_mse(dp_dx, outlet)
            )

        p_gauge = self.masked_mean(p, fluid).square()

        total = (
            weights["continuity"] * cont_loss
            + weights["momentum"] * mom_loss
            + weights["boundary"] * bc_loss
            + weights.get("pressure_gauge", 0.05) * p_gauge
        )

        return {
            "continuity": cont_loss,
            "momentum": mom_loss,
            "boundary": bc_loss,
            "pressure_gauge": p_gauge,
            "total_physics": total,
            "masks": masks,
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str) -> nn.Module:
    key = name.lower()
    if key == "base":
        return UNetBase()
    if key == "attention":
        return UNetAttention()
    if key == "transformer":
        return UNetTransformer()
    raise ValueError(f"Unknown model name: {name}")
 
 
 
 
