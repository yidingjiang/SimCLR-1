import numpy as np
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
        band_with=1.0,
        variance_init=0.0,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_dim = patch_dim
        self.num_patches = num_patches

        self.patch_size = patch_size

        # Positional encoding
        band_width = band_with
        m, n = np.meshgrid(np.arange(image_size // patch_size), np.arange(image_size // patch_size))
        coord = np.stack((m, n), axis=-1)
        coord = np.reshape(coord, (-1, 2))
        W = np.random.normal(loc=0, scale=band_width, size=(dim // 2, 2))
        self.W = nn.Parameter(torch.from_numpy(W).float(), requires_grad=False)
        self.X = nn.Parameter(torch.from_numpy(coord).float(), requires_grad=False)  # (p*p, 2)
#         self.X = torch.from_numpy(coord).float().to(self.W.device)

        # Random Transformation
        # self.I = torch.from_numpy(np.array([[1, 0], [0, 1]])).float()
        var = variance_init
        self.A_mean = torch.nn.Parameter(torch.from_numpy(np.array([[1, 0], [0, 1]])).float())
        self.b_mean = torch.nn.Parameter(torch.from_numpy(np.array([0.0, 0.0])).float())
        self.A_std = torch.nn.Parameter(torch.from_numpy(np.array([[var, var], [var, var]])).float())
        self.b_std = torch.nn.Parameter(torch.from_numpy(np.array([var, var])).float())

        self.dim = dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.embedding_to_patch = nn.Linear(dim, patch_dim)
        self.dropout = nn.Dropout(emb_dropout)

        X_base = torch.unsqueeze(self.X, 0)
        self.base_pos_embedding = nn.Parameter(self._compute_embedding(X_base))
        self.transformer = SplitTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, self.base_pos_embedding
        )

        self.to_latent = nn.Identity()

    def _compute_embedding(self, coordinates):
        """"Compute the position encoding for a batch of coordinates."""
        proj = torch.einsum("od,bpd->bpo", self.W, coordinates)
        z_cos = (2.0 / (self.dim // 2))**0.5 * torch.cos(proj)
        z_sin = (2.0 / (self.dim // 2))**0.5 * torch.sin(proj)
        Z = torch.cat((z_cos, z_sin), axis=-1)  # (b, x, y, d)
        return Z

    def _compute_transformed_embedding(self, batch_size):
        """Transform and compute the position encoding for a batch of coordinates."""
        A_noise = torch.randn(
            batch_size, 2, 2
        ).to(self.W.device)  # convert noise to transformation matrices
        b_noise = torch.randn(batch_size, 2).to(self.W.device)  # convert noise to offset
        A = self.A_mean + F.softplus(self.A_std) * A_noise  # (b, 2, 2)
        b = self.b_mean + F.softplus(self.b_std) * b_noise  # (b, 2)
        X = repeat(torch.unsqueeze(self.X, 0), "() p d -> b p d", b=batch_size)
        X_t = torch.einsum("bpd,bdo->bpo", X, A) + rearrange(b, "b d -> b 1 d")
        return self._compute_embedding(X_t)

    def forward(self, img, noise_token, mask=None, pre_embedding=False):
        del noise_token
        p = self.patch_size

        if p == 1:
            x = rearrange(img, "b c h w -> b (h w) c")
        else:
            x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        pos_embedding = self._compute_transformed_embedding(b)
        # pos_embedding = rearrange(pos_embedding, "b h w d -> b (h w) d")
        # x = self.dropout(x)

        x = self.transformer(x, pos_embedding, mask)
        if pre_embedding:
            return x

        x = self.embedding_to_patch(x)
        return rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(self.num_patches ** 0.5),
            p1=p,
            p2=p,
        )


class SplitTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, base_pos_embedding):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            modules = [
                PreNorm(
                    dim,
                    SplitAttention(dim, base_pos_embedding, heads=heads, dim_head=dim_head, dropout=dropout, pos_only=True),
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

    def __init__(self, dim, base_pos_embedding, heads=8, dim_head=64, dropout=0.0, pos_only=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos_only = pos_only

        self.base_pos_embedding = base_pos_embedding
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q_pos = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_pos = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, pos_embedding, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        print('q size', q.size())
        if not self.pos_only:
            qk_pos = self.to_qk_pos(pos_embedding).chunk(2, dim=-1)
            q_pos, k_pos = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qk_pos)

            dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
            # TODO: this can be changed to indentity-ish operation
            dots_pos = torch.einsum("bhid,bhjd->bhij", q_pos, k_pos) * self.scale
            dots = (dots + dots_pos) / 2.
        else:
            print('base pos size 1', self.base_pos_embedding.size())
            base_pos_embedding = repeat(self.base_pos_embedding, "() n d -> b n d", b=b)
            print('base pos size 2', base_pos_embedding.size())
            print('pos emb', pos_embedding.size())
            dots = torch.einsum("bid,bjd->bij", pos_embedding, base_pos_embedding) * self.scale
            dots = repeat(dots.unsqueeze(1), "b () i j -> b h i j", h=h)
#             pos_embedding = rearrange(pos_embedding, "b (n h) d -> b h n d", h=h)
#             print('pos embedding size 2', pos_embedding.size())
#             base_pos_embedding = rearrange(pos_embedding, "b (n h) d -> b h n d", h=h)
#             dots = torch.einsum("bhid,bhjd->bhij", pos_embedding, base_pos_embedding) * self.scale
#             qk_pos = [self.to_q_pos(pos_embedding), self.to_k_pos(base_pos_embedding)]
#             q_pos, k_pos = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qk_pos)
#             dots = torch.einsum("bhid,bhjd->bhij", q_pos, k_pos) * self.scale

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


#########################################################################################################
#########################################################################################################

class ExplicitGeometricAugmentor(nn.Module):
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
        band_width=1.0,
        variance_init=0.0,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_dim = patch_dim
        self.num_patches = num_patches

        self.patch_size = patch_size

        # Positional encoding
        m, n = np.meshgrid(np.arange(image_size // patch_size), np.arange(image_size // patch_size))
        coord = np.stack((m, n), axis=-1)
        coord = np.reshape(coord, (-1, 2))
        self.X = nn.Parameter(torch.from_numpy(coord).float(), requires_grad=False)  # (p*p, 2)
        self.band_width = nn.Parameter(torch.tensor(band_width), requires_grad=True)

        # Random Transformation
        # self.I = torch.from_numpy(np.array([[1, 0], [0, 1]])).float()
        var = variance_init
        self.A_mean = torch.nn.Parameter(torch.from_numpy(np.array([[1, 0], [0, 1]])).float())
        self.b_mean = torch.nn.Parameter(torch.from_numpy(np.array([0.0, 0.0])).float())
        self.A_std = torch.nn.Parameter(torch.from_numpy(np.array([[var, var], [var, var]])).float())
        self.b_std = torch.nn.Parameter(torch.from_numpy(np.array([var, var])).float())

        self.dim = dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.embedding_to_patch = nn.Linear(dim, patch_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ExplicitSplitTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, self.X, self.band_width
        )

        self.to_latent = nn.Identity()

    def _compute_transformed_coord(self, batch_size):
        """Transform and compute the position encoding for a batch of coordinates."""
        A_noise = torch.randn(
            batch_size, 2, 2
        ).to(self.X.device)  # convert noise to transformation matrices
        b_noise = torch.randn(batch_size, 2).to(self.X.device)  # convert noise to offset
        A = self.A_mean + F.softplus(self.A_std) * A_noise  # (b, 2, 2)
        b = self.b_mean + F.softplus(self.b_std) * b_noise  # (b, 2)
        X = repeat(torch.unsqueeze(self.X, 0), "() p d -> b p d", b=batch_size)
        X_t = torch.einsum("bpd,bdo->bpo", X, A) + rearrange(b, "b d -> b 1 d")
        return X_t

    def forward(self, img, noise_token, mask=None, pre_embedding=False):
        del noise_token
        p = self.patch_size

        if p == 1:
            x = rearrange(img, "b c h w -> b (h w) c")
        else:
            x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        pos_embedding = self._compute_transformed_coord(b)
        # pos_embedding = rearrange(pos_embedding, "b h w d -> b (h w) d")
        # x = self.dropout(x)

        x = self.transformer(x, pos_embedding, mask)
        if pre_embedding:
            return x

        x = self.embedding_to_patch(x)
        return rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(self.num_patches ** 0.5),
            p1=p,
            p2=p,
        )


class ExplicitSplitTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, base_pos_embedding, band_width, pos_only=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            modules = [
                PreNorm(
                    dim,
                    ExplicitSplitAttention(dim, base_pos_embedding, band_width, heads=heads, dim_head=dim_head, dropout=dropout, pos_only=pos_only),
                ),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ]
            self.layers.append(nn.ModuleList(modules))

    def forward(self, x, pos_embedding, mask=None):
        for attn, ff in self.layers:
            x = attn(x, pos_embedding=pos_embedding, mask=mask)
            x = ff(x)
        return x


class ExplicitSplitAttention(nn.Module):

    def __init__(self, dim, base_pos_embedding, band_width, heads=8, dim_head=64, dropout=0.0, pos_only=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos_only = pos_only

        self.base_pos_embedding = base_pos_embedding
        self.band_width = band_width
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q_pos = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_pos = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, pos_embedding, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        base_pos_embedding = repeat(self.base_pos_embedding.unsqueeze(0), "() n d -> b n d", b=b)
        print('pos_embedding', pos_embedding.size())
        print('base_pos_embedding', base_pos_embedding.size())
        pe = repeat(pos_embedding.unsqueeze(1), 'b () i j -> b r i j', r=n)
        bpe = repeat(base_pos_embedding.unsqueeze(2), 'b i () j -> b i r j', r=n)
        diff = pe - bpe
        print('diff', diff.size())
        dots = -torch.sum(diff**2, dim=-1) / self.band_width
#         dots = torch.exp(diff)
        print('dots', dots.size())
        print('v', v.size())
        dots = repeat(dots.unsqueeze(1), "b () i j-> b h i j", h=h)
        print('dots', dots.size())

        if not self.pos_only:
            dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
            dots = (dots + dots_pos) / 2.

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