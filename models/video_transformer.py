import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNTransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer + adaLN(cond)

    x:           (S, B, D)
    cond_tokens: (S, B, cond_dim)  # 각 토큰 위치마다 cond 벡터
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        cond_dim: int,
        mod_scale: float = 0.1,
        mod_clip: float = 5.0,
    ):
        """
        mod_scale : cond로부터 나온 gamma, beta 에 곱해줄 스케일 (기본 0.1)
        mod_clip  : modulation 클리핑 범위. None 이면 클리핑 안 함.
        """
        super().__init__()

        # 기본 TransformerEncoderLayer 구성과 동일
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,  # (S, B, D)
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        # adaLN용 scale/shift 생성기 (attn 앞, ff 앞 각각)
        self.mod1 = nn.Linear(cond_dim, 2 * d_model)
        self.mod2 = nn.Linear(cond_dim, 2 * d_model)

        # modulation 강도 및 클리핑 파라미터
        self.mod_scale = mod_scale
        self.mod_clip = mod_clip

    def apply_adaLN(self, x, cond_tokens, ln, mod_linear):
        """
        x:           (S, B, D)
        cond_tokens: (S, B, cond_dim)
        ln:          LayerNorm(D)
        mod_linear:  Linear(cond_dim -> 2*D)
        """
        x_norm = ln(x)  # (S, B, D)

        # cond_tokens -> (S, B, 2D)
        mod = mod_linear(cond_tokens)
        gamma, beta = mod.chunk(2, dim=-1)  # (S, B, D), (S, B, D)

        # modulation을 너무 세게 걸지 않도록 스케일/클리핑
        # 기본은 identity 근처에서 작은 perturbation:
        #   gamma ≈ 1, beta ≈ 0
        gamma = 1.0 + self.mod_scale * gamma
        beta = self.mod_scale * beta

        if self.mod_clip is not None:
            # gamma는 identity(1.0)를 중심으로 제한
            gamma = torch.clamp(
                gamma,
                1.0 - self.mod_clip,
                1.0 + self.mod_clip,
            )
            beta = torch.clamp(beta, -self.mod_clip, self.mod_clip)

        return gamma * x_norm + beta

    def forward(self, x, cond_tokens):
        """
        x:           (S, B, D)
        cond_tokens: (S, B, cond_dim)
        """
        # --- Self-Attention block with adaLN ---
        x_mod1 = self.apply_adaLN(x, cond_tokens, self.norm1, self.mod1)
        attn_out, _ = self.self_attn(
            x_mod1, x_mod1, x_mod1, need_weights=False
        )
        x = x + self.dropout1(attn_out)

        # --- FFN block with adaLN ---
        x_mod2 = self.apply_adaLN(x, cond_tokens, self.norm2, self.mod2)
        ff = self.linear2(self.dropout_ff(F.gelu(self.linear1(x_mod2))))
        x = x + self.dropout2(ff)

        return x


class VideoTransformerPredictor(nn.Module):
    def __init__(
        self,
        img_size: int = 864,
        patch_size: int = 16,
        in_channels: int = 2,   # [ice_filled, sea_mask]
        cond_dim: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_frames: int = 8,    # history frame 최대 개수 (input_len)
        land_value: float = 0.0,
    ):
        super().__init__()

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.max_frames = max_frames
        self.land_value = land_value

        self.H_patches = img_size // patch_size
        self.W_patches = img_size // patch_size
        self.num_patches_per_frame = self.H_patches * self.W_patches

        # 1) Patch embedding: [얼음 + sea_mask] → patch 단위 feature
        self.patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2) 시간/공간 임베딩 (cond는 adaLN으로 주입)
        self.time_embed = nn.Embedding(max_frames, d_model)
        self.spatial_embed = nn.Embedding(self.num_patches_per_frame, d_model)

        # 3) adaLN Transformer encoder
        self.layers = nn.ModuleList(
            [
                AdaLNTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    cond_dim=cond_dim,
                    # 필요하면 여기서 mod_scale, mod_clip 바꿔도 된다.
                    # mod_scale=0.1,
                    # mod_clip=5.0,
                )
                for _ in range(n_layers)
            ]
        )

        # 4) 디코더: 마지막 프레임 token feature → full-res 프레임
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                d_model,
                128,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # [-1, 1]
        )

    def forward(self, ice, cond, mask):
        """
        ice:  (B, T, 1, H, W)
        cond: (B, T, cond_dim)  # 과거 T-1 프레임 + 타깃 프레임의 ERA5
        mask: (B, T, 1, H, W)
        return:
            pred_full: (B, 1, H, W)
            ice_target: (B, 1, H, W)
            mask_target: (B, 1, H, W)
        """
        B, T, C, H, W = ice.shape
        device = ice.device

        # 과거 프레임 개수 (마지막 프레임 하나는 target)
        T_in = T - 1
        assert (
            T_in <= self.max_frames
        ), "increase max_frames if you use more history frames"

        # --- 입력/타겟 분리 ---
        ice_in = ice[:, :T_in]      # (B, T_in, 1, H, W)
        cond_hist = cond[:, :T_in]  # (B, T_in, cond_dim)  ← 과거 프레임 cond
        cond_target = cond[:, -1]   # (B, cond_dim)       ← 타깃 프레임 cond
        mask_in = mask[:, :T_in]    # (B, T_in, 1, H, W)

        ice_target = ice[:, -1]     # (B, 1, H, W)
        mask_target = mask[:, -1]   # (B, 1, H, W)

        # --- NaN → 0 채우고 sea_mask 채널 추가 ---
        ice_filled = torch.nan_to_num(ice_in, nan=0.0)   # (B, T_in, 1, H, W)
        sea_mask = (mask_in == 0).float()                # (B, T_in, 1, H, W), sea=1, land=0
        x_in = torch.cat([ice_filled, sea_mask], dim=2)  # (B, T_in, 2, H, W)

        # --- Patch embedding ---
        # (B, T_in, 2, H, W) → (B*T_in, 2, H, W)
        x_bt = x_in.view(B * T_in, self.in_channels, H, W)
        patch_feat = self.patch_embed(x_bt)              # (B*T_in, D, H', W')
        _, D, H_p, W_p = patch_feat.shape
        assert H_p == self.H_patches and W_p == self.W_patches

        # 다시 (B, T_in, D, H', W') → (B, T_in, N, D)
        N = H_p * W_p
        patch_feat = patch_feat.view(B, T_in, D, H_p, W_p)
        patch_feat = patch_feat.permute(0, 1, 3, 4, 2).contiguous()  # (B, T_in, H', W', D)
        patch_feat = patch_feat.view(B, T_in, N, D)                  # (B, T_in, N, D)

        # --- 시간 임베딩 ---
        time_ids = torch.arange(T_in, device=device)                 # (T_in,)
        time_emb = self.time_embed(time_ids)                         # (T_in, D)
        time_emb = time_emb[None, :, None, :].expand(B, T_in, N, D)  # (B, T_in, N, D)

        # --- 공간 임베딩 ---
        spatial_ids = torch.arange(N, device=device)                 # (N,)
        spatial_emb = self.spatial_embed(spatial_ids)                # (N, D)
        spatial_emb = spatial_emb[None, None, :, :].expand(
            B, T_in, N, D
        )  # (B, T_in, N, D)

        # --- 최종 토큰 시퀀스 (cond는 여기선 더하지 않음; adaLN에서 사용) ---
        tokens = patch_feat + time_emb + spatial_emb                 # (B, T_in, N, D)

        # (B, T_in, N, D) → (S, B, D), S = T_in * N
        S = T_in * N
        tokens = tokens.view(B, S, D)                                # (B, S, D)
        tokens = tokens.permute(1, 0, 2).contiguous()                # (S, B, D)

        # --- cond를 각 토큰 위치로 확장해서 (S, B, cond_dim) 만들기 ---
        # 1) 과거 프레임 cond (hist)
        cond_hist_tokens = cond_hist[:, :, None, :].expand(
            B, T_in, N, self.cond_dim
        )                                                            # (B, T_in, N, cond_dim)
        cond_hist_tokens = cond_hist_tokens.reshape(
            B, S, self.cond_dim
        )                                                            # (B, S, cond_dim)

        # 2) 타깃 프레임 cond (target) → 모든 토큰에 공통으로 주입
        cond_target_tokens = cond_target[:, None, None, :].expand(
            B, T_in, N, self.cond_dim
        )                                                            # (B, T_in, N, cond_dim)
        cond_target_tokens = cond_target_tokens.reshape(
            B, S, self.cond_dim
        )                                                            # (B, S, cond_dim)

        # 3) 두 정보를 합쳐서 최종 cond_tokens
        cond_tokens = cond_hist_tokens + cond_target_tokens          # (B, S, cond_dim)
        cond_tokens = cond_tokens.permute(1, 0, 2).contiguous()      # (S, B, cond_dim)

        # --- adaLN Transformer 통과 ---
        h = tokens
        for layer in self.layers:
            h = layer(h, cond_tokens)   # (S, B, D)

        # --- 마지막 input frame에 해당하는 토큰 feature만 추출 ---
        h = h.permute(1, 0, 2).contiguous()                          # (B, S, D)
        h = h.view(B, T_in, N, D)                                    # (B, T_in, N, D)
        last_frame_feat = h[:, -1]                                   # (B, N, D)

        # (B, N, D) → (B, D, H', W')
        last_frame_feat = last_frame_feat.view(
            B, H_p, W_p, D
        ).permute(0, 3, 1, 2).contiguous()

        # --- 디코더: full-res 예측 ---
        pred_full = self.decoder(last_frame_feat)                    # (B, 1, H, W)

        # --- 육지 픽셀은 land_value로 강제 세팅 ---
        sea = (mask_target == 0).float()
        land = 1.0 - sea
        pred_full = pred_full * sea + self.land_value * land

        # target도 같이 반환 (loss 계산용)
        return pred_full, ice_target, mask_target

    @staticmethod
    def masked_mse_loss(pred, target, mask):
        """
        pred:   (B, 1, H, W)
        target: (B, 1, H, W)  [-1,0,1] + NaN on land
        mask:   (B, 1, H, W)  1=land, 0=sea

        바다(=mask==0) 위치에서만 MSE 계산.
        """
        # NaN → 0
        target_filled = torch.nan_to_num(target, nan=0.0)

        sea = (mask == 0).float()
        valid = sea > 0.5

        diff_sq = (pred - target_filled) ** 2

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        loss = diff_sq[valid].mean()
        return loss
