import torch
import torch.nn as nn

# --- Supervised Contrastive Loss (SupCon) ---
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """计算 SupCon loss
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
                      在监督场景下，n_views=1，所以 shape 是 [bsz, feat_dim].
                      需要确保输入的 features 已经 L2 归一化。
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                  has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 如果 features 是 [bsz, feat_dim]，增加一个 n_views 维度
        if len(features.shape) < 3:
            features = features.unsqueeze(1) # -> [bsz, 1, feat_dim]

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 无监督 SimCLR 的情况 (mask 对角线为 1)
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # [bsz, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask_{i,j}=1 if labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device) # [bsz, bsz]
        else:
            # 使用了自定义 mask
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # n_views (这里是 1)
        # contrast_feature 是将所有 view 拉平后的特征 [bsz * n_views, feat_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # [bsz, feat_dim]

        if self.contrast_mode == 'one':
            # SimCLR 对比模式 'one' (这里用不上)
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # SupCon 对比模式 'all'
            anchor_feature = contrast_feature # 每个样本都作为 anchor
            anchor_count = contrast_count # 每个样本有 contrast_count 个 view (这里是 1)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算相似度矩阵 [bsz * n_views, bsz * n_views]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # [bsz, bsz]

        # 调整 mask 适应 [bsz * n_views, bsz * n_views] 的结构
        # 在监督场景 n_views=1 时，mask 已经是 [bsz, bsz]
        # labels [bsz, 1], mask [bsz, bsz]

        # mask-out self-contrast cases (对角线置 0)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        ) # 对角线为 0，其余为 1
        mask = mask * logits_mask # 保留同类样本的 mask，且排除自身

        # 计算 log_prob
        exp_logits = torch.exp(logits) * logits_mask # 分母项：排除自身的 exp(sim)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9) # 加上 epsilon 防止 log(0)

        # 计算 mean log-likelihood over positive
        # mask 现在是正样本对的指示器 (不含自身)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9) # SupCon 原论文公式，对每个 anchor 的所有正样本取平均
        # 改为类似 InfoNCE 的形式，只考虑一个正样本（但 SupCon 应该用上面的）
        # 这里我们用 SupCon 的标准形式：所有正样本的 log-likelihood 之和 / 正样本数量

        # 避免除以零（如果没有正样本对，loss 为 0）
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1)[mask_sum > 0] / mask_sum[mask_sum > 0]


        # 总损失
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # SupCon 原论文公式
        # 简化：不带 base_temperature 缩放
        loss = - mean_log_prob_pos

        # SupCon 可能对批次中没有正样本对（只有一个类别的样本，或某个样本没有其他同类样本）的 anchor 计算出 NaN
        # 需要处理这种情况，通常是取有效的 anchor 的平均损失
        loss = loss.mean() if len(loss) > 0 else torch.tensor(0.0).to(device) # 取有效 anchor 损失的平均值


        return loss