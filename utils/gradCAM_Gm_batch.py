import numpy as np
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt # 如果在其他地方需要绘图，保留此导入

# --- 钩子需要小心管理 ---
# 不要将钩子定义为全局函数并追加到全局列表中
# 明确传递列表或在需要时使用类结构

# --- 注册钩子并存储数据的函数 ---
def register_cam_hooks(target_layer, activations_list, gradients_list):
    """
    为目标层注册前向和后向钩子，将激活值和梯度存储到列表中。
    """
    # 前向钩子：在前向传播后触发，存储模块的输出
    fwd_handle = target_layer.register_forward_hook(lambda m, inp, out: activations_list.append(out.detach()))
    # 后向钩子：在计算 w.r.t. 模块输出的梯度后触发，存储梯度
    # 使用 register_full_backward_hook 获取 w.r.t. 层输出的梯度
    bwd_handle = target_layer.register_full_backward_hook(lambda m, grad_inp, grad_out: gradients_list.append(grad_out[0].detach()))
    # 返回钩子句柄，用于后续移除钩子
    return fwd_handle, bwd_handle

# --- 批处理 Grad-CAM 函数 ---
def grad_CAM_map_2D_batch(model, input_tensor, res_p_batch, target_layer, target_class_index=None):
    """
    高效的批处理二维图像 Grad-CAM 计算。

    Args:
        model (torch.nn.Module): 分类模型。
        input_tensor (torch.Tensor): 输入图像批次 (B, C, H, W)。
                                     必须 require_grad=True。
        target_layer (torch.nn.Module): 目标卷积层模块。
        target_class_index (int 或 torch.Tensor, 可选): Grad-CAM 的目标类别索引。
                                                            如果为 None，使用批次中每个样本的预测类别索引。
                                                            如果是张量 (B,)，为每个样本使用指定的索引。

    Returns:
        torch.Tensor: Grad-CAM 热力图批次 (B, 1, H, W)，缩放到 [0, 1]。
                      requires_grad 为 False。
    """
    model.eval()  # 确保模型处于评估模式 (不启用 dropout, batchnorm 使用全局统计量等)

    batch_size, channels, height, width = input_tensor.shape

    # 用于存储当前批次调用中的激活值和梯度的列表
    activations = []
    gradients = []

    # --- 1. 注册钩子并执行前向传播 ---
    # 注册钩子 - 它们在触发时会填充上面的列表
    fwd_handle, bwd_handle = register_cam_hooks(target_layer, activations, gradients)

    # 前向传播
    # 在调用此函数之前，请确保 input_tensor.requires_grad=True
    # 这对于内部的反向传播至关重要
    # 直接将 input_tensor 传递给模型
    # 假设您的模型返回输出和特征 (如果只返回输出，忽略 _ )
    model_output, _, _ = model(input_tensor, res_p_batch)

    # --- 2. 计算目标梯度并执行反向传播 ---
    model.zero_grad() # 清除模型参数的梯度，以便进行新的反向传播

    # 为反向传播选择目标输出 (logit/分数)
    if target_class_index is None:
        # 使用批次中每个样本的预测类别
        predicted_classes = model_output.argmax(dim=1)
        # 收集与预测类别对应的 logit
        # 形状 (B,)
        target_outputs = model_output[torch.arange(batch_size, device=input_tensor.device), predicted_classes]
    elif isinstance(target_class_index, int):
        # 为批次中所有样本使用固定的目标类别索引
        # 形状 (B,)
        target_outputs = model_output[:, target_class_index]
    elif torch.is_tensor(target_class_index) and target_class_index.shape == (batch_size,):
        # 使用为每个样本提供的目标类别索引 (形状为 B 的张量)
        # 形状 (B,)
        target_outputs = model_output[torch.arange(batch_size, device=input_tensor.device), target_class_index]
    else:
        raise ValueError("无效的 target_class_index。必须是 int, 形状为 (B,) 的张量, 或 None。")

    # 从选择的目标输出之和执行反向传播
    # 这个反向传播将触发已注册的钩子
    # 确保 target_outputs 需要梯度 (如果 model_output 需要梯度，并且 model_output 如果 input_tensor 需要梯度，则 target_outputs 也需要)
    # 我们需要钩子捕获的 w.r.t. 目标层特征的梯度。
    # 但是我们仍然需要从模型输出开始一个反向传播。
    # 我们需要计算 w.r.t. 'model_output' 的 'target_outputs' 梯度。
    # 这本质上是选取 w.r.t. 完整输出的梯度的一部分。
    # 附加到 target_layer 的钩子将捕获 w.r.t. target_layer 输出所需的梯度。
    # 所以，我们需要从 'target_outputs' 反向传播。
    # 这需要 target_outputs 需要梯度，如果 model_output 需要梯度，它就满足。
    # 为整个批次反向传播 target_outputs 的总和。
    # 这等同于为每个样本的目标类别反向传播 1。
    # 标准 Grad-CAM 通常不需要 retain_graph=True。
    # 标准 Grad-CAM 通常不需要 create_graph=True。
    target_outputs.sum().backward(retain_graph=False)

    # --- 在反向传播后立即移除钩子 ---
    # 这是为了防止钩子在后续的批次或训练步骤中被重复触发
    fwd_handle.remove()
    bwd_handle.remove()

    # --- 3. 获取激活值和梯度，计算热力图 ---
    # 钩子应该已经被触发并填充了列表
    # 列表现在应该各自包含一个张量，代表整个批次
    if not activations or not gradients:
        # 如果 target_layer 不在导致 model_output 计算图的路径上，或者钩子没有触发，可能会发生这种情况。
        raise RuntimeError(f"钩子未捕获目标层 {target_layer} 的激活值或梯度。请检查模型路径和目标层的连接性。")

    activation_batch = activations[0] # 形状 (B, channels, H_layer, W_layer)
    gradient_batch = gradients[0]   # 形状 (B, channels, H_layer, W_layer) # w.r.t. target_layer 输出的梯度

    # 清空列表以释放内存并防止在下次调用时出现问题
    activations.clear()
    gradients.clear()

    # 计算每个通道在整个批次中的全局平均梯度
    # 形状 (B, channels, 1, 1)
    global_avg_gradients = torch.mean(gradient_batch, dim=[2, 3], keepdim=True)

    # 计算激活值的加权和 (B, channels, H_layer, W_layer) * (B, channels, 1, 1) -> (B, channels, H_layer, W_layer)
    # 在通道维度求和 -> (B, H_layer, W_layer)
    heatmap_batch = torch.sum(activation_batch * global_avg_gradients, dim=1)

    # 应用 ReLU，去除负值
    heatmap_batch = torch.relu(heatmap_batch) # 形状 (B, H_layer, W_layer)

    # 将热力图上采样到原始图像大小
    heatmap_batch = heatmap_batch.unsqueeze(1) # 添加通道维度: (B, 1, H_layer, W_layer)
    heatmap_batch = F.interpolate(heatmap_batch,
                                  size=(height, width), # 上采样到原始图像的高和宽
                                  mode='bilinear', # 或 'bicubic'
                                  align_corners=False) # 如果输入的空间尺寸是 2xN+1，则设置为 True

    # --- 4. 叠加和最终归一化 ---
    # 对上采样后的热力图进行独立归一化到 [0, 1] (模仿Code 2中的第一次归一化)
    # 虽然Code 2的第一次归一化是在上采样之前，这里为了逻辑顺畅放在上采样后
    heatmap_normalized = torch.zeros_like(heatmap_batch)  # (B, 1, H, W)
    epsilon_norm = 1e-8
    for i in range(batch_size):
        min_val = heatmap_batch[i].min().item()
        max_val = heatmap_batch[i].max().item()
        if max_val == min_val:
            heatmap_normalized[i] = torch.zeros_like(heatmap_batch[i])
        else:
            heatmap_normalized[i] = (heatmap_batch[i] - min_val) / (max_val - min_val + epsilon_norm)

    # 将热力图通道复制到与输入图像通道数一致
    # heatmap_normalized 是 (B, 1, H, W), input_tensor 是 (B, C, H, W)
    if heatmap_normalized.shape[1] != input_tensor.shape[1]:
        heatmap_colored = heatmap_normalized.repeat(1, input_tensor.shape[1], 1, 1)  # (B, C, H, W)
    else:
        heatmap_colored = heatmap_normalized  # (B, C, H, W)

    # 执行叠加 (简单相加)
    added_result = heatmap_colored + input_tensor  # (B, C, H, W)

    # 对叠加结果应用 ReLU (模仿 Code 2)
    added_result_relu = torch.relu(added_result)  # (B, C, H, W)

    # 对叠加并应用 ReLU 后的结果进行最终独立归一化到 [0, 1]
    final_overlay_normalized = torch.zeros_like(added_result_relu)  # (B, C, H, W)
    # 使用相同的 epsilon_norm
    for i in range(batch_size):
        min_val = added_result_relu[i].min().item()
        max_val = added_result_relu[i].max().item()
        if max_val == min_val:
            final_overlay_normalized[i] = torch.zeros_like(added_result_relu[i])
        else:
            final_overlay_normalized[i] = (added_result_relu[i] - min_val) / (max_val - min_val + epsilon_norm)

    # 返回分离（detached）的最终结果张量
    return final_overlay_normalized.detach()  # 形状 (B, C, H, W), requires_grad=False

