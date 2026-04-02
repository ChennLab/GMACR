import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. 时间嵌入模块 (Sinusoidal Positional Embedding + MLP)
# ============================================================

# --- ConvBlock 和 UpBlock 保持不变 ---
class ConvBlock(nn.Module):
    """基础卷积块 (保持不变)"""
    def __init__(self, in_channels, out_channels, use_norm=True, use_activation=True, **kwargs):
        super().__init__()
        padding_mode = 'reflect' if kwargs.get('kernel_size', 3) > 1 else 'zeros'
        padding = kwargs.get('padding', 1)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_norm, padding_mode=padding_mode, **kwargs)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else None
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else None
    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.activation: x = self.activation(x)
        return x

class SinusoidalPosEmb(nn.Module):
    """生成正弦位置嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 嵌入的目标维度

    def forward(self, time): # time 是一个形状为 [B] 的张量
        device = time.device
        half_dim = self.dim // 2
        # 计算嵌入频率: 1 / (10000^(2i / dim))
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 将时间广播并乘以频率
        embeddings = time[:, None] * embeddings[None, :] # [B, half_dim]
        # 计算 sin 和 cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # [B, dim]
        # 如果 dim 是奇数，最后一个维度补零 (虽然通常 dim 是偶数)
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0,1))
        return embeddings

class TimeEmbeddingMLP(nn.Module):
    """处理正弦嵌入的小型 MLP"""
    def __init__(self, time_embed_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = time_embed_dim # 默认输出维度与输入相同
        if hidden_dim is None:
            hidden_dim = time_embed_dim * 4 # 常用扩展因子

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(), # Swish 激活函数
            nn.Linear(hidden_dim, out_dim)
        )
        self.sinusoidal_emb = SinusoidalPosEmb(time_embed_dim)

    def forward(self, t): # t 是原始时间步 [B]
        # 1. 获取正弦嵌入
        sin_emb = self.sinusoidal_emb(t)
        # 2. 通过 MLP 处理
        time_embedding = self.mlp(sin_emb)
        return time_embedding

# ============================================================
# 2. 自适应组归一化 (AdaGN)
# ============================================================
class AdaptiveGroupNorm(nn.Module):
    """自适应组归一化 (FiLM Layer)"""
    def __init__(self, num_groups, num_channels, cond_embed_dim, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        # GroupNorm 本身，不学习仿射参数 gamma 和 beta
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        # 线性层，用于从条件嵌入预测 gamma 和 beta
        self.projection = nn.Linear(cond_embed_dim, num_channels * 2) # 预测 gamma 和 beta

    def forward(self, x, cond_emb):
        # 1. 标准 GroupNorm
        normalized_x = self.group_norm(x)

        # 2. 预测 gamma 和 beta
        gamma_beta = self.projection(cond_emb) # [B, num_channels * 2]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1) # 分割为 [B, C] 和 [B, C]

        # 3. 调整形状并应用仿射变换
        # 形状变为 [B, C, 1, 1] 以便广播到 [B, C, H, W]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return normalized_x * gamma + beta

# ============================================================
# 3. 条件残差块 (Conditioned Residual Block)
# ============================================================
class ConditionedResBlock(nn.Module):
    """集成了 AdaGN 的时间/标签条件残差块"""
    def __init__(self, in_channels, out_channels, time_embed_dim, label_embed_dim,
                 num_groups=32, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.label_embed_dim = label_embed_dim
        self.cond_embed_dim = time_embed_dim + label_embed_dim # 合并后的条件维度

        # 第一个卷积层 + AdaGN + SiLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveGroupNorm(num_groups, out_channels, self.cond_embed_dim)
        self.act1 = nn.SiLU()

        # 可选的 Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 第二个卷积层 + AdaGN + SiLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(num_groups, out_channels, self.cond_embed_dim)
        self.act2 = nn.SiLU()

        # 残差连接的 1x1 卷积 (如果输入输出通道数不同)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb, label_emb):
        # 合并时间和标签嵌入
        cond_emb = torch.cat([time_emb, label_emb], dim=1) # [B, time_dim + label_dim]

        # 主路径
        h = self.conv1(x)
        h = self.norm1(h, cond_emb) # 使用 AdaGN
        h = self.act1(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h, cond_emb) # 使用 AdaGN
        h = self.act2(h)

        # 添加残差连接
        return h + self.residual_conv(x)

# ============================================================
# 4. 自注意力模块 (Self-Attention)
# ============================================================
class SelfAttention(nn.Module):
    """空间自注意力模块 (带 GroupNorm)"""
    def __init__(self, in_channels, num_groups=32, num_heads=4):
        super().__init__()
        if in_channels % num_heads != 0:
            # 确保通道数可以被头数整除，如果不能，选择一个能整除的头数
            # 或者调整通道数，这里简单选择单头
             print(f"Warning: in_channels ({in_channels}) not divisible by num_heads ({num_heads}). Using num_heads=1.")
             num_heads = 1

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5 # 缩放因子

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False) # QKV 投影
        self.to_out = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 输出投影

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1) # 分割为 Q, K, V -> List[Tensor[B, C, H, W]]

        # 重塑并调整维度以进行多头注意力计算
        # q, k, v: [B, num_heads, head_dim, H*W]
        q, k, v = map(
            lambda t: t.reshape(B, self.num_heads, self.head_dim, H * W), qkv
        )

        # 计算注意力图: Q^T * K ( scaled )
        # attention: [B, num_heads, H*W, H*W]
        attention_scores = torch.einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)

        # 应用注意力到 V
        # out: [B, num_heads, head_dim, H*W]
        out = torch.einsum('b h i j, b h d j -> b h d i', attention_probs, v)

        # 重塑回原始图像形状
        out = out.reshape(B, C, H, W)
        out = self.to_out(out)

        return out + residual # 添加残差连接

# ============================================================
# 5. 下采样和上采样模块
# ============================================================
class Downsample(nn.Module):
    """下采样模块 (使用 stride=2 的卷积)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """上采样模块 (使用 Upsample + Conv)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

# ============================================================
# 6. 增强版条件生成器
# ============================================================
class ConditionalResidualGenerator(nn.Module):
    """
    增强版条件生成器 (U-Net 结构)，输出残差图
    - 使用 AdaGN 条件残差块
    - 内部注入正弦时间嵌入和标签嵌入
    - 瓶颈处加入自注意力
    """
    def __init__(self,
                 img_channels=1,
                 num_classes=2,
                 img_size=128, # 需要图像尺寸来确定 GroupNorm 的适用性
                 base_features=64, # 基础通道数
                 feature_mults=(1, 2, 4, 8), # 通道数乘数
                 time_embed_dim=256,
                 label_embed_dim=64, # 标签嵌入维度 (可以调整)
                 num_res_blocks=1, # 每个分辨率层级的残差块数量
                 num_groups=32, # GroupNorm 的组数
                 use_attention=True # 是否在瓶颈处使用注意力
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.base_features = base_features
        self.feature_mults = feature_mults
        self.num_res_blocks = num_res_blocks

        # --- 1. 时间嵌入 ---
        self.time_mlp = TimeEmbeddingMLP(time_embed_dim)

        # --- 2. 标签嵌入 ---
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        # --- 3. 初始卷积 ---
        # 注意：初始卷积不进行条件注入，只提取初始特征
        self.init_conv = nn.Conv2d(img_channels, base_features, kernel_size=3, padding=1)

        # --- 4. 构建 U-Net 层 ---
        self.downs = nn.ModuleList() # 编码器层
        self.ups = nn.ModuleList()   # 解码器层
        current_channels = base_features
        num_resolutions = len(feature_mults)

        # === 编码器 (Encoder) ===
        for i, mult in enumerate(feature_mults):
            out_channels = base_features * mult
            is_last_res = (i == num_resolutions - 1)

            # 添加残差块
            for _ in range(num_res_blocks):
                self.downs.append(ConditionedResBlock(
                    current_channels, out_channels, time_embed_dim, label_embed_dim, num_groups
                ))
                current_channels = out_channels

            # 如果不是最后一层，添加下采样
            if not is_last_res:
                self.downs.append(Downsample(current_channels, current_channels))

        # === 瓶颈层 (Bottleneck) ===
        self.mid_block1 = ConditionedResBlock(
            current_channels, current_channels, time_embed_dim, label_embed_dim, num_groups
        )
        if use_attention:
             # 检查通道数是否适合 GroupNorm 和 Attention Heads
             if current_channels % num_groups != 0:
                 print(f"Warning: Bottleneck channels ({current_channels}) not divisible by num_groups ({num_groups}). GN might error.")
             self.mid_attention = SelfAttention(current_channels, num_groups=num_groups)
        else:
             self.mid_attention = nn.Identity()
        self.mid_block2 = ConditionedResBlock(
            current_channels, current_channels, time_embed_dim, label_embed_dim, num_groups
        )

        # === 解码器 (Decoder) ===
        for i, mult in reversed(list(enumerate(feature_mults))):
            out_channels = base_features * mult
            is_first_res = (i == 0)

            # 添加上采样 (如果不是瓶颈层之后的第一组)
            if i != num_resolutions - 1:
                self.ups.append(Upsample(current_channels, current_channels))

            # 添加残差块 (注意通道数变化)
            # 输入通道数 = 当前通道数 (来自上采样或瓶颈) + 跳跃连接通道数
            skip_channels = base_features * feature_mults[i]
            decoder_in_channels = current_channels + skip_channels
            for _ in range(num_res_blocks + 1): # 解码器通常比编码器多一个块来处理融合后的特征
                 self.ups.append(ConditionedResBlock(
                     decoder_in_channels if _ == 0 else out_channels, # 第一个块输入是融合后的通道数
                     out_channels,
                     time_embed_dim, label_embed_dim, num_groups
                 ))
                 current_channels = out_channels
                 decoder_in_channels = out_channels # 后续块的输入通道就是输出通道

        # --- 5. 最终输出层 ---
        self.final_norm = nn.GroupNorm(num_groups, base_features) # 对最终特征进行归一化
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_features, img_channels, kernel_size=1) # 1x1 卷积输出
        self.final_tanh = nn.Tanh() # 限制输出范围到 [-1, 1]

    def forward(self, x, t, target_label):
        # 1. 计算时间嵌入
        time_emb = self.time_mlp(t) # [B, time_embed_dim]

        # 2. 计算标签嵌入
        label_emb = self.label_embedding(target_label) # [B, label_embed_dim]

        # 3. 初始卷积
        h = self.init_conv(x)

        # --- 4. 编码器 (修正 Skip Connection 保存逻辑) ---
        skips = [] # 初始化空的 skips 列表
        encoder_block_idx = 0
        for i, mult in enumerate(self.feature_mults):
             is_last_res = (i == len(self.feature_mults) - 1)
             # 应用 ResBlocks
             for _ in range(self.num_res_blocks):
                 res_block_module = self.downs[encoder_block_idx]
                 h = res_block_module(h, time_emb, label_emb)
                 encoder_block_idx += 1
             # *** 在下采样之前保存 h 到 skips ***
             skips.append(h)

             # 应用下采样 (如果不是最后一层)
             if not is_last_res:
                 downsample_module = self.downs[encoder_block_idx]
                 h = downsample_module(h)
                 encoder_block_idx += 1
        # skips 列表现在应该包含每个编码器层级在下采样前的输出，顺序从高分辨率到低分辨率

        # 5. 瓶颈层
        h = self.mid_block1(h, time_emb, label_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb, label_emb)

        # --- 6. 解码器 (使用修正后的 skips 列表) ---
        ups_module_indices = [idx for idx, m in enumerate(self.ups) if isinstance(m, Upsample)]
        res_block_indices = [idx for idx, m in enumerate(self.ups) if isinstance(m, ConditionedResBlock)]

        ups_idx_counter = 0
        res_idx_counter = 0

        # 从最低分辨率开始向上解码
        for i, mult in reversed(list(enumerate(self.feature_mults))):
            # --- 应用上采样 (如果不是最低分辨率层) ---
            if i != len(self.feature_mults) - 1:
                # 重要: 假设 __init__ 中 Upsample 和 ResBlock 的添加顺序是固定的
                # 例如，每个层级先加 Upsample, 再加 ResBlocks
                # 需要确保 ups_module_indices 和 res_block_indices 的顺序与 __init__ 一致
                upsample_module_global_idx = ups_module_indices[ups_idx_counter]
                h = self.ups[upsample_module_global_idx](h)
                ups_idx_counter += 1

            # --- 拼接跳跃连接 ---
            # 从 skips 列表末尾弹出对应编码器层级的输出
            skip_feature = skips.pop() # 现在 pop 出来的是正确的 skip connection
            # 检查并调整空间尺寸 (理论上应该匹配，除非有 stride/padding 问题)
            if h.shape[2:] != skip_feature.shape[2:]:
                 skip_feature = F.interpolate(skip_feature, size=h.shape[2:], mode='bilinear', align_corners=True)
            # 进行拼接
            h = torch.cat([h, skip_feature], dim=1)

            # --- 应用残差块 ---
            # 第一个残差块处理拼接后的 h
            res_block_module_global_idx = res_block_indices[res_idx_counter]
            # 调试信息 (可选):
            # print(f"Decoder Level i={i}, Block 0 Input: h.shape={h.shape}, Expected InChannels={self.ups[res_block_module_global_idx].in_channels}")
            h = self.ups[res_block_module_global_idx](h, time_emb, label_emb)
            res_idx_counter += 1

            # 后续残差块处理上一个块的输出 h
            for _ in range(self.num_res_blocks):
                res_block_module_global_idx = res_block_indices[res_idx_counter]
                # 调试信息 (可选):
                # print(f"Decoder Level i={i}, Block {_ + 1} Input: h.shape={h.shape}, Expected InChannels={self.ups[res_block_module_global_idx].in_channels}")
                h = self.ups[res_block_module_global_idx](h, time_emb, label_emb)
                res_idx_counter += 1

        # 7. 最终输出
        # 调试信息 (可选):
        # print(f"Final Norm Input: h.shape={h.shape}, Expected InChannels={self.base_features}")
        h = self.final_norm(h)
        h = self.final_act(h)
        residual_map = self.final_conv(h)
        residual_map = self.final_tanh(residual_map)

        return residual_map



# --- 判别器 ConditionalPatchDiscriminator 保持不变 ---
class ConditionalPatchDiscriminator(nn.Module):
    """条件判别器 (无需修改)"""
    def __init__(self, img_channels=1, num_classes=2, features=[64, 128, 256, 512]):
        super().__init__()
        in_channels = img_channels + num_classes
        self.num_classes = num_classes
        self.initial = ConvBlock(in_channels, features[0], kernel_size=4, stride=2, padding=1, use_norm=False, use_activation=True)
        self.conv1 = ConvBlock(features[0], features[1], kernel_size=4, stride=2, padding=1, use_norm=True, use_activation=True)
        self.conv2 = ConvBlock(features[1], features[2], kernel_size=4, stride=2, padding=1, use_norm=True, use_activation=True)
        self.conv3 = ConvBlock(features[2], features[3], kernel_size=4, stride=1, padding=1, use_norm=True, use_activation=True)
        self.final_conv = nn.Conv2d(features[3], 1, kernel_size=4, stride=1, padding=1, bias=True, padding_mode='reflect')
    def forward(self, x, label):
        label_one_hot = nn.functional.one_hot(label, num_classes=self.num_classes).float()
        label_map = label_one_hot.view(label_one_hot.size(0), self.num_classes, 1, 1)
        label_map = label_map.expand(-1, -1, x.size(2), x.size(3))
        x_conditioned = torch.cat((x, label_map), dim=1)
        x = self.initial(x_conditioned)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_conv(x)
        return x


class DiffusionHelper:
    """
    辅助类，用于处理扩散模型的前向加噪过程和时间步采样。
    """
    def __init__(self, T=100, beta_start=0.0001, beta_end=0.02, device="cuda"):
        """
        初始化扩散参数。

        Args:
            T (int): 总扩散步数。
            beta_start (float): beta 调度的起始值。
            beta_end (float): beta 调度的结束值。
            device (str or torch.device): 计算设备。
        """
        self.T = T
        self.device = device

        # 线性 beta 调度
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # 预计算前向过程所需的项
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

    def sample_timesteps(self, n):
        """
        采样 n 个时间步 (从 1 到 T-1，因为 t=0 通常不加噪)。
        注意：这里我们采样从 0 到 T-1，在 forward_process 中处理 t=0 的情况。
        """
        # 返回 [0, T-1] 之间的整数
        return torch.randint(0, self.T, (n,), device=self.device, dtype=torch.long)

    def forward_process(self, x0, t):
        """
        执行前向加噪过程。

        Args:
            x0 (Tensor): 原始清晰图像 [B, C, H, W]。
            t (Tensor): 时间步长 [B]。

        Returns:
            tuple: (xt, noise)
                   xt: 加噪后的图像 [B, C, H, W]。
                   noise: 添加的高斯噪声 [B, C, H, W]。
        """
        # 获取对应时间步 t 的预计算值，并调整形状以进行广播
        # t 的索引对应 alpha_bars 的索引
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]

        # 采样高斯噪声
        noise = torch.randn_like(x0, device=self.device)

        # 计算加噪后的图像 xt
        # 注意处理 t=0 的情况，此时 alpha_bar[0] = alpha[0]
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise