import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch, kernel_size)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel_size)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.relu(self.w_g(g_up) + self.w_x(x))
        alpha = self.psi(attn)
        return x * alpha


class DecoderBlockAttn(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=5):
        super().__init__()
        self.attn = AttentionGate(in_ch, skip_ch, max(1, skip_ch // 2))
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch, kernel_size)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel_size)

    def forward(self, x, skip):
        skip_att = self.attn(x, skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip_att, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HardConstraintHead(nn.Module):
    def __init__(self, enforce_obstacle=True, zero_mean_p=True):
        super().__init__()
        self.enforce_obstacle = enforce_obstacle
        self.zero_mean_p = zero_mean_p

    def forward(self, y_hat, x_in):
        out = y_hat
        fluid_mask = None
        if self.enforce_obstacle:
            region = x_in[:, 1:2]
            fluid_mask = (region > 0.5).float()
            out = out * fluid_mask

        if self.zero_mean_p:
            p = out[:, 2:3]
            if fluid_mask is not None:
                denom = fluid_mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
                p_mean = (p * fluid_mask).sum(dim=(1, 2, 3), keepdim=True) / denom
            else:
                p_mean = p.mean(dim=(1, 2, 3), keepdim=True)
            out = torch.cat([out[:, 0:2], p - p_mean], dim=1)
            if fluid_mask is not None:
                out = out * fluid_mask

        return out


class UNetBase(nn.Module):
    name = "UNet-Base"

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, filters=(8, 16, 32, 32), enforce_obstacle=True, zero_mean_p=True):
        super().__init__()
        f = list(filters)

        self.enc1 = EncoderBlock(in_channels, f[0], kernel_size)
        self.enc2 = EncoderBlock(f[0], f[1], kernel_size)
        self.enc3 = EncoderBlock(f[1], f[2], kernel_size)
        self.enc4 = EncoderBlock(f[2], f[3], kernel_size)

        self.bottleneck = nn.Sequential(ConvBlock(f[3], f[3], kernel_size), ConvBlock(f[3], f[3], kernel_size))

        def build_decoder():
            return nn.ModuleList([
                DecoderBlock(f[3], f[3], f[2], kernel_size),
                DecoderBlock(f[2], f[2], f[1], kernel_size),
                DecoderBlock(f[1], f[1], f[0], kernel_size),
                DecoderBlock(f[0], f[0], f[0], kernel_size),
                nn.Conv2d(f[0], 1, kernel_size=1),
            ])

        self.decoders = nn.ModuleList([build_decoder() for _ in range(out_channels)])
        self.head = HardConstraintHead(enforce_obstacle, zero_mean_p)

    def forward(self, x):
        x_in = x
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        z = self.bottleneck(x4)

        outs = []
        for dec in self.decoders:
            d = dec[0](z, s4)
            d = dec[1](d, s3)
            d = dec[2](d, s2)
            d = dec[3](d, s1)
            d = dec[4](d)
            outs.append(d)

        y = torch.cat(outs, dim=1)
        return self.head(y, x_in)


class UNetAttention(nn.Module):
    name = "UNet-Attention"

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, filters=(8, 16, 32, 32), enforce_obstacle=True, zero_mean_p=True):
        super().__init__()
        f = list(filters)

        self.enc1 = EncoderBlock(in_channels, f[0], kernel_size)
        self.enc2 = EncoderBlock(f[0], f[1], kernel_size)
        self.enc3 = EncoderBlock(f[1], f[2], kernel_size)
        self.enc4 = EncoderBlock(f[2], f[3], kernel_size)

        self.bottleneck = nn.Sequential(ConvBlock(f[3], f[3], kernel_size), ConvBlock(f[3], f[3], kernel_size))

        def build_decoder():
            return nn.ModuleList([
                DecoderBlockAttn(f[3], f[3], f[2], kernel_size),
                DecoderBlockAttn(f[2], f[2], f[1], kernel_size),
                DecoderBlockAttn(f[1], f[1], f[0], kernel_size),
                DecoderBlockAttn(f[0], f[0], f[0], kernel_size),
                nn.Conv2d(f[0], 1, kernel_size=1),
            ])

        self.decoders = nn.ModuleList([build_decoder() for _ in range(out_channels)])
        self.head = HardConstraintHead(enforce_obstacle, zero_mean_p)

    def forward(self, x):
        x_in = x
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        z = self.bottleneck(x4)

        outs = []
        for dec in self.decoders:
            d = dec[0](z, s4)
            d = dec[1](d, s3)
            d = dec[2](d, s2)
            d = dec[3](d, s1)
            d = dec[4](d)
            outs.append(d)

        y = torch.cat(outs, dim=1)
        return self.head(y, x_in)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, channels, modes1, modes2, activation=True):
        super().__init__()
        self.spec = SpectralConv2d(channels, channels, modes1, modes2)
        self.w = nn.Conv2d(channels, channels, 1)
        self.activation = activation

    def forward(self, x):
        y = self.spec(x) + self.w(x)
        if self.activation:
            y = F.gelu(y)
        return y


class FNOBottleneck(nn.Module):
    def __init__(self, channels, modes1=4, modes2=3, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(FNOBlock(channels, modes1, modes2, activation=(i < num_layers - 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNetFNO(nn.Module):
    name = "UNet-FNO"

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, filters=(8, 16, 32, 32), fno_modes=(4, 3), fno_layers=4, enforce_obstacle=True, zero_mean_p=True):
        super().__init__()
        f = list(filters)

        self.enc1 = EncoderBlock(in_channels, f[0], kernel_size)
        self.enc2 = EncoderBlock(f[0], f[1], kernel_size)
        self.enc3 = EncoderBlock(f[1], f[2], kernel_size)
        self.enc4 = EncoderBlock(f[2], f[3], kernel_size)

        self.bottleneck = FNOBottleneck(f[3], modes1=fno_modes[0], modes2=fno_modes[1], num_layers=fno_layers)

        def build_decoder():
            return nn.ModuleList([
                DecoderBlockAttn(f[3], f[3], f[2], kernel_size),
                DecoderBlockAttn(f[2], f[2], f[1], kernel_size),
                DecoderBlockAttn(f[1], f[1], f[0], kernel_size),
                DecoderBlockAttn(f[0], f[0], f[0], kernel_size),
                nn.Conv2d(f[0], 1, kernel_size=1),
            ])

        self.decoders = nn.ModuleList([build_decoder() for _ in range(out_channels)])
        self.head = HardConstraintHead(enforce_obstacle, zero_mean_p)

    def forward(self, x):
        x_in = x
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        z = self.bottleneck(x4)

        outs = []
        for dec in self.decoders:
            d = dec[0](z, s4)
            d = dec[1](d, s3)
            d = dec[2](d, s2)
            d = dec[3](d, s1)
            d = dec[4](d)
            outs.append(d)

        y = torch.cat(outs, dim=1)
        return self.head(y, x_in)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
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

    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, bottleneck_hw, embed_dim=128, num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        h, w = bottleneck_hw
        self.proj_in = nn.Conv2d(channels, embed_dim, 1)
        self.proj_out = nn.Linear(embed_dim, channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(channels)
        self.hw = (h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        if (h, w) != self.hw:
            raise ValueError(f"Expected bottleneck {self.hw}, got {(h, w)}")
        t = self.proj_in(x).flatten(2).transpose(1, 2)
        t = t + self.pos_embed
        for blk in self.blocks:
            t = blk(t)
        t = self.proj_out(t)
        t = self.norm(t)
        return t.transpose(1, 2).reshape(b, c, h, w)


class UNetTransformer(nn.Module):
    name = "UNet-Transformer"

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, filters=(8, 16, 32, 32), input_hw=(172, 79), embed_dim=128, num_heads=4, num_transformer_layers=2, enforce_obstacle=True, zero_mean_p=True):
        super().__init__()
        f = list(filters)

        self.enc1 = EncoderBlock(in_channels, f[0], kernel_size)
        self.enc2 = EncoderBlock(f[0], f[1], kernel_size)
        self.enc3 = EncoderBlock(f[1], f[2], kernel_size)
        self.enc4 = EncoderBlock(f[2], f[3], kernel_size)

        bh, bw = input_hw[0] // 16, input_hw[1] // 16
        self.bottleneck = TransformerBottleneck(f[3], (bh, bw), embed_dim=embed_dim, num_heads=num_heads, num_layers=num_transformer_layers)

        def build_decoder():
            return nn.ModuleList([
                DecoderBlockAttn(f[3], f[3], f[2], kernel_size),
                DecoderBlockAttn(f[2], f[2], f[1], kernel_size),
                DecoderBlockAttn(f[1], f[1], f[0], kernel_size),
                DecoderBlockAttn(f[0], f[0], f[0], kernel_size),
                nn.Conv2d(f[0], 1, kernel_size=1),
            ])

        self.decoders = nn.ModuleList([build_decoder() for _ in range(out_channels)])
        self.head = HardConstraintHead(enforce_obstacle, zero_mean_p)

    def forward(self, x):
        x_in = x
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        z = self.bottleneck(x4)

        outs = []
        for dec in self.decoders:
            d = dec[0](z, s4)
            d = dec[1](d, s3)
            d = dec[2](d, s2)
            d = dec[3](d, s1)
            d = dec[4](d)
            outs.append(d)

        y = torch.cat(outs, dim=1)
        return self.head(y, x_in)


class PhysicsInformedLoss(nn.Module):
    def __init__(self, nx=172, ny=79, lx=0.260, ly=0.120, nu=1e-4, u_inlet=0.1):
        super().__init__()
        self.nu = nu
        self.u_inlet = u_inlet
        dx = lx / (nx - 1)
        dy = ly / (ny - 1)

        k_dx = torch.zeros(1, 1, 3, 1)
        k_dx[0, 0, 0, 0] = -1.0 / (2.0 * dx)
        k_dx[0, 0, 2, 0] = 1.0 / (2.0 * dx)
        self.register_buffer("kernel_dx", k_dx)

        k_dy = torch.zeros(1, 1, 1, 3)
        k_dy[0, 0, 0, 0] = -1.0 / (2.0 * dy)
        k_dy[0, 0, 0, 2] = 1.0 / (2.0 * dy)
        self.register_buffer("kernel_dy", k_dy)

        k_d2x = torch.zeros(1, 1, 3, 1)
        k_d2x[0, 0, 0, 0] = 1.0 / dx**2
        k_d2x[0, 0, 1, 0] = -2.0 / dx**2
        k_d2x[0, 0, 2, 0] = 1.0 / dx**2
        self.register_buffer("kernel_d2x", k_d2x)

        k_d2y = torch.zeros(1, 1, 1, 3)
        k_d2y[0, 0, 0, 0] = 1.0 / dy**2
        k_d2y[0, 0, 0, 1] = -2.0 / dy**2
        k_d2y[0, 0, 0, 2] = 1.0 / dy**2
        self.register_buffer("kernel_d2y", k_d2y)

    def _deriv(self, f, kernel):
        kh, kw = kernel.shape[-2], kernel.shape[-1]
        f_pad = F.pad(f, (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate")
        return F.conv2d(f_pad, kernel)

    def ddx(self, f):
        return self._deriv(f, self.kernel_dx)

    def ddy(self, f):
        return self._deriv(f, self.kernel_dy)

    def d2dx2(self, f):
        return self._deriv(f, self.kernel_d2x)

    def d2dy2(self, f):
        return self._deriv(f, self.kernel_d2y)

    def region_masks(self, x_in):
        r = x_in[:, 1:2]
        fluid = (r > 0.5).float()
        obstacle = (r < 0.5).float()
        wall = ((r > 1.5) & (r < 2.5)).float()
        inlet = ((r > 2.5) & (r < 3.5)).float()
        outlet = (r > 3.5).float()
        noslip = ((obstacle + wall) > 0.5).float()

        interior = fluid.clone()
        interior[:, :, :1, :] = 0
        interior[:, :, -1:, :] = 0
        interior[:, :, :, :1] = 0
        interior[:, :, :, -1:] = 0

        return {
            "fluid": fluid,
            "interior": interior,
            "obstacle": obstacle,
            "wall": wall,
            "noslip": noslip,
            "inlet": inlet,
            "outlet": outlet,
        }

    @staticmethod
    def _masked_mean(x, mask):
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    def continuity(self, ux, uy, mask):
        div = self.ddx(ux) + self.ddy(uy)
        return self._masked_mean(div**2, mask)

    def momentum(self, ux, uy, p, mask):
        dux_dx = self.ddx(ux)
        dux_dy = self.ddy(ux)
        duy_dx = self.ddx(uy)
        duy_dy = self.ddy(uy)
        dp_dx = self.ddx(p)
        dp_dy = self.ddy(p)
        lap_ux = self.d2dx2(ux) + self.d2dy2(ux)
        lap_uy = self.d2dx2(uy) + self.d2dy2(uy)
        res_x = ux * dux_dx + uy * dux_dy + dp_dx - self.nu * lap_ux
        res_y = ux * duy_dx + uy * duy_dy + dp_dy - self.nu * lap_uy
        return self._masked_mean(res_x**2, mask) + self._masked_mean(res_y**2, mask)

    def boundary(self, ux, uy, masks):
        loss = ux.new_tensor(0.0)
        noslip = masks["noslip"]
        if noslip.sum() > 0:
            loss = loss + self._masked_mean(ux**2, noslip) + self._masked_mean(uy**2, noslip)
        inlet = masks["inlet"]
        if inlet.sum() > 0:
            loss = loss + self._masked_mean((ux - self.u_inlet) ** 2, inlet)
            loss = loss + self._masked_mean(uy**2, inlet)
        outlet = masks["outlet"]
        if outlet.sum() > 0:
            loss = loss + self._masked_mean(self.ddx(ux) ** 2, outlet)
        return loss

    def forward(self, y_hat, x_in, weights=None):
        if weights is None:
            weights = {"continuity": 1.0, "momentum": 0.1, "boundary": 1.0}

        ux = y_hat[:, 0:1]
        uy = y_hat[:, 1:2]
        p = y_hat[:, 2:3]
        m = self.region_masks(x_in)

        l_cont = self.continuity(ux, uy, m["interior"])
        l_mom = self.momentum(ux, uy, p, m["interior"])
        l_bc = self.boundary(ux, uy, m)

        total = weights["continuity"] * l_cont + weights["momentum"] * l_mom + weights["boundary"] * l_bc
        return {"continuity": l_cont, "momentum": l_mom, "boundary": l_bc, "total_physics": total}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_type, **kwargs):
    key = model_type.lower()
    if key == "base":
        return UNetBase(**kwargs)
    if key in ["attn", "attention"]:
        return UNetAttention(**kwargs)
    if key == "fno":
        return UNetFNO(**kwargs)
    if key in ["trans", "transformer"]:
        return UNetTransformer(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")
