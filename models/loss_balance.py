# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Model codes for CodeBind
import math

class LossBalancer:
    def __init__(self, loss_names, base_loss_name='info_nce', 
                 momentum=0.9, min_interval=10, max_interval=100, max_steps=40000):
        self.loss_names = loss_names
        self.base_loss_name = base_loss_name
        self.fixed_weight_keys = [base_loss_name, 'loss_rec']
        self.momentum = momentum
        
        # 动态间隔参数
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_steps = max_steps  # 预计在多少步时达到最大间隔
        self.current_interval = min_interval
        
        # 定义不同任务的量级缩放因子 (Target Magnitude relative to Base)
        # 1.0 代表量级接近, 0.1 代表小一个量级, 0.01 代表小两个量级
        self.scale_factors = {
            'loss_vq': 0.01,           # 小两个量级
            'loss_cmcm': 0.1,          # 小一个量级
            'loss_vq_reg': 0.1,        # 小一个量级
            'loss_modal_decomp': 0.1,  # 小一个量级
            'loss_uniform': 1.0,       # 量级接近
        }
        self.max_weight_vq = 10000.0   # vq 权重的上限

        # 状态追踪
        self.running_stats = {name: 1.0 for name in loss_names}
        self.current_weights = {name: 1.0 for name in loss_names}
        self.global_step = 0
        self.last_update_step = 0
        self.is_initialized = False

    def get_weighted_loss(self, raw_losses, is_training=True):
        """
        raw_losses: dict {'loss_name': tensor}
        is_training: bool, 仅在训练模式下更新权重
        """
        # 验证是否有缺失的loss (防止某个batch没计算某些loss导致报错)
        valid_losses = {k: v for k, v in raw_losses.items() if k in self.loss_names}

        if is_training:
            self.global_step += 1
            
            # 1. 强初始化 (First Batch)
            if not self.is_initialized:
                self._initialize_weights(valid_losses)
                self.is_initialized = True
                self.last_update_step = self.global_step
            
            # 2. 动态间隔更新
            elif (self.global_step - self.last_update_step) >= self.current_interval:
                self._update_weights_ema(valid_losses)

                # --- 正弦增长逻辑 ---
                # 计算当前进度 (0.0 到 1.0)
                progress = min(1.0, self.global_step / self.max_steps)
                # 使用 sin(progress * pi/2) 从 0 变化到 1
                # 这样间隔增长在最开始和最末尾都比较平滑
                multiplier = math.sin(progress * (math.pi / 2))
                self.current_interval = int(self.min_interval + (self.max_interval - self.min_interval) * multiplier)
                self.last_update_step = self.global_step

        # 3. 计算加权总和 (Validation 时使用当前的缓存权重)
        total_loss = 0
        weights_to_log = {}
        
        for name, loss_val in valid_losses.items():
            w = self.current_weights.get(name, 1.0)
            total_loss += w * loss_val
            if is_training:
                weights_to_log[f"weight/{name}"] = w
        
        if is_training:
            weights_to_log["weight/update_interval"] = float(self.current_interval)
            
        return total_loss, weights_to_log

    def _calculate_custom_weight(self, name, base_magnitude, current_loss_mag):
        """核心逻辑：根据预设的量级比例计算权重"""
        if name in self.fixed_weight_keys:
            return 1.0
        # 获取缩放因子，如果未定义则默认为 1.0 (保持原有逻辑)
        factor = self.scale_factors.get(name, 1.0)
        # 权重 = (基准量级 * 缩放因子) / 当前 Loss 量级
        new_weight = (base_magnitude * factor) / (current_loss_mag + 1e-8)
        # 特殊处理：loss_vq 的权重上限设定
        if name == 'loss_vq':
            new_weight = min(self.max_weight_vq, new_weight)
            
        return new_weight
    
    def _initialize_weights(self, raw_losses):
        base_val = raw_losses[self.base_loss_name].detach().item()
        for name in self.loss_names:
            if name in raw_losses:
                val = abs(raw_losses[name].detach().item())
                self.running_stats[name] = val
                self.current_weights[name] = self._calculate_custom_weight(name, base_val, val)
                

    def _update_weights_ema(self, raw_losses):
        # 先更新基准
        base_val = raw_losses[self.base_loss_name].detach().item()
        self.running_stats[self.base_loss_name] = (self.momentum * self.running_stats[self.base_loss_name] + 
                                                   (1 - self.momentum) * base_val)
        base_magnitude = self.running_stats[self.base_loss_name]
        
        # 准备打印字符串
        # print_msg = f"\n>>> [Step {self.global_step}] Loss Weights Updated (Interval: {self.current_interval}) <<<"
        # print_msg += f"\n    Base ({self.base_loss_name}) Magnitude: {base_magnitude:.6f}, Weight: 1.0000"

        for name in self.loss_names:
            if name == self.base_loss_name or name not in raw_losses: continue
            
            val = abs(raw_losses[name].detach().item())
            self.running_stats[name] = (self.momentum * self.running_stats[name] + 
                                       (1 - self.momentum) * val)
            
            self.current_weights[name] = self._calculate_custom_weight(name, base_magnitude, self.running_stats[name])

                # 添加到打印信息
                # print_msg += f"\n    - {name:18s} | Mag: {self.running_stats[name]:.6f} | Weight: {new_weight:.6f}"

        # print(print_msg + "\n" + "="*60)
