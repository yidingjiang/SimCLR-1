import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert (
            num_patches > MIN_NUM_PATCHES
        ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class NoiseTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ExemplarTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        num_noise_token=2,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        assert (
            num_patches > MIN_NUM_PATCHES
        ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"

        self.patch_size = patch_size

        self.num_noise_token = num_noise_token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + num_noise_token, dim)
        )
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.embedding_to_patch = nn.Linear(dim, patch_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = NoiseTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.to_latent = nn.Identity()

    def forward(self, img, noise_token, mask=None):
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        x = torch.cat((noise_token, x), dim=1)
        x += self.pos_embedding[
            :, : (n + self.num_noise_token)
        ]  # TODO: figure out what's going on here
        x = self.dropout(x)

        x = self.transformer(x, mask)
        x = x[:, 2:]
        x = self.embedding_to_patch(x)

        # TODO: project back to 3 d
        return rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(self.num_patches ** 0.5),
            p1=p,
            p2=p,
        )


#########################################################################################################
#########################################################################################################


class GeometricAugmentor(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        # num_noise_token=2,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        # assert (
        #     num_patches > MIN_NUM_PATCHES
        # ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"

        self.patch_size = patch_size

        # Positional encoding
        band_width = 0.75
        m, n = np.meshgrid(np.arange(num_patches), np.arange(width))
        coord = np.stack((m, n), axis=-1)
        coord = np.reshape(x, (-1, 2))
        self.X = torch.from_numpy(coord).float()  # (p, p, 2)
        W = np.random.normal(loc=0, scale=band_width, size=(dim // 2, 2))
        self.W = torch.from_numpy(W).float()

        # Random Transformation
        # self.I = torch.from_numpy(np.array([[1, 0], [0, 1]])).float()
        self.A_mean = torch.from_numpy(np.array([[1, 0], [0, 1]])).float()
        self.b_mean = torch.from_numpy(np.array([0, 0])).float()
        self.A_std = torch.from_numpy(np.array([[1, 1], [1, 1]])).float()
        self.b_std = torch.from_numpy(np.array([1, 1])).float()

        # self.num_noise_token = num_noise_token
        # self.pos_embedding = nn.Parameter(
        #     torch.randn(1, num_patches + num_noise_token, dim)
        # )
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.embedding_to_patch = nn.Linear(dim, patch_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = NoiseTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.to_latent = nn.Identity()

    def _compute_embedding(self, batch_size):
        A_noise = torch.randn(
            batch_size, 2, 2
        )  # convert noise to transformation matrices
        b_noise = torch.randn(batch_size, 2)  # convert noise to offset
        A = self.A_mean + F.softplus(self.A_std) * A_noise  # (b, 2, 2)
        b = self.b_mean + F.softplus(self.b_std) * b_noise  # (b, 2)

        X = repeat(self.X, "() x y d -> b x y d", b=batch_size)
        X_t = torch.einsum("bxyd,bdo->bxyo", X, A) + rearrange(b, "b d -> b 1 1 d")

        proj = torch.einsum("od,bxyd->bxyo", self.W, X_t)
        z_cos = torch.sqrt(2.0 / (self.dim // 2)) * torch.cos(proj)
        z_sin = torch.sqrt(2.0 / (self.dim // 2)) * torch.sin(proj)
        Z = torch.cat((z_cos, z_sin), axis=-1)  # (b, x, y, d)
        # z_cos = norm * np.sqrt(2) * np.cos(W @ (A self.coord.T + b))
        # z_sin = norm * np.sqrt(2) * np.sin(W @ (A self.coord.T + b))
        # Z = np.concatenate((z_cos, z_sin), axis=0)
        # Z = np.reshape(Z, (2 * d_model, length, width))
        return Z

    def forward(self, img, noise_token, mask=None):
        del noise_token
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        pos_embedding = self._compute_embedding(b)
        pos_embedding = rearrange(pos_embedding, "b h w d -> b (h w) d")
        # x = self.dropout(x)

        x = self.transformer(x, mask)
        x = x[:, 2:]
        x = self.embedding_to_patch(x)

        return rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(self.num_patches ** 0.5),
            p1=p,
            p2=p,
        )


class SplitTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            modules = [
                Residual(
                    PreNorm(
                        dim,
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    )
                ),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ]
            self.layers.append(nn.ModuleList(modules))

    def forward(self, x, pos_embedding, mask=None):
        for attn, ff in self.layers:
            x = attn(x, pos_embedding=pos_embedding, mask=mask)
            x = ff(x)
        return x


class SplitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, pos_only=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos_only = pos_only

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qk_pos = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, pos_embedding, mask=None):
        b, n, _, h = *x.shape, self.heads

        if not self.pos_only:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

            qk_pos = self.to_qk_pos(pos_embedding).chunk(2, dim=-1)
            q_pos, k_pos = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qk_pos)

            dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
            # TODO: this can be changed to indentity-ish operation
            dots_pos = torch.einsum("bhid,bhjd->bhij", q_pos, k_pos) * self.scale
            dots = (dots + dots_pos) / 2.
        else:
            qk_pos = self.to_qk_pos(pos_embedding).chunk(2, dim=-1)
            q_pos, k_pos = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qk_pos)
            # TODO: this can be changed to indentity-ish operation
            dots = torch.einsum("bhid,bhjd->bhij", q_pos, k_pos) * self.scale

        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out
