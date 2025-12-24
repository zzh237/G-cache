"""
详细解释 log_prob 的计算过程
"""
import numpy as np

print("=" * 80)
print("Log Probability 计算详解")
print("=" * 80)

# 模拟采样过程
edges_info = [
    # (edge_id, edge_name, prob, sampled)
    (0, "node_0 → node_0", None, "SKIP"),      # 自连接，跳过
    (1, "node_0 → node_1", 0.277, False),      # 未采样
    (2, "node_0 → node_2", 0.321, True),       # 采样到
    (3, "node_0 → node_3", 0.491, True),       # 采样到
    (4, "node_1 → node_0", 0.277, False),      # 未采样
    (5, "node_1 → node_1", None, "SKIP"),      # 自连接，跳过
    (6, "node_1 → node_2", 0.468, False),      # 未采样
    (7, "node_1 → node_3", 0.269, False),      # 未采样
    (8, "node_2 → node_0", 0.321, False),      # 未采样
    (9, "node_2 → node_1", 0.468, False),      # 未采样
    (10, "node_2 → node_2", None, "SKIP"),     # 自连接，跳过
    (11, "node_2 → node_3", 0.339, False),     # 未采样
    (12, "node_3 → node_0", 0.491, True),      # 采样到
    (13, "node_3 → node_1", 0.269, False),     # 未采样
    (14, "node_3 → node_2", 0.339, False),     # 未采样
    (15, "node_3 → node_3", None, "SKIP"),     # 自连接，跳过
]

print("\n步骤 1: 遍历所有边，计算每条边的 log_prob")
print("-" * 80)

log_probs = [0.0]  # 初始值
total_log_prob_manual = 0.0

for edge_id, edge_name, prob, sampled in edges_info:
    if sampled == "SKIP":
        print(f"边 {edge_id:2d}: {edge_name:20s} | SKIP (不计入 log_prob)")
        continue
    
    if sampled:
        # 采样到这条边 → 记录 log(prob)
        log_p = np.log(prob)
        log_probs.append(log_p)
        total_log_prob_manual += log_p
        print(f"边 {edge_id:2d}: {edge_name:20s} | ✓ SAMPLED   | "
              f"prob={prob:.3f} → log(prob)={log_p:.4f}")
    else:
        # 没采样到这条边 → 记录 log(1 - prob)
        log_p = np.log(1 - prob)
        log_probs.append(log_p)
        total_log_prob_manual += log_p
        print(f"边 {edge_id:2d}: {edge_name:20s} | ✗ REJECTED  | "
              f"prob={prob:.3f} → log(1-prob)={log_p:.4f}")

print("-" * 80)
print(f"\n总 log_prob = sum(log_probs) = {total_log_prob_manual:.4f}")
print(f"log_probs 列表长度: {len(log_probs)} (包含初始的 0.0)")

print("\n" + "=" * 80)
print("关键理解")
print("=" * 80)

print("""
1. 为什么要遍历所有边？
   - 因为这是一个概率模型，需要计算整个采样过程的联合概率
   - 每条边都有一个决策：采样 or 不采样
   - 总概率 = P(边1的决策) × P(边2的决策) × ... × P(边N的决策)
   - 取对数后：log P_total = log P1 + log P2 + ... + log PN

2. 为什么采样到的边用 log(prob)，未采样的用 log(1-prob)？
   - 采样到：说明这次随机采样"成功"了，概率是 prob
   - 未采样：说明这次随机采样"失败"了，概率是 1-prob
   
   例如：
   - 边的概率 prob = 0.3
   - 如果采样到：P(采样成功) = 0.3 → log_prob = log(0.3) = -1.204
   - 如果未采样：P(采样失败) = 0.7 → log_prob = log(0.7) = -0.357

3. 为什么要计算 log_prob？
   - 用于强化学习的 REINFORCE 算法
   - 梯度公式：∇L = -log_prob × reward
   - 如果 reward 高（任务成功），增加采样到的边的概率
   - 如果 reward 低（任务失败），减少采样到的边的概率
""")

print("\n" + "=" * 80)
print("详细计算示例")
print("=" * 80)

print("\n假设只有 3 条边：")
print("-" * 80)

edges_simple = [
    ("边A", 0.8, True),   # 采样到
    ("边B", 0.3, False),  # 未采样
    ("边C", 0.6, True),   # 采样到
]

print("边的信息：")
for name, prob, sampled in edges_simple:
    status = "✓ 采样到" if sampled else "✗ 未采样"
    print(f"  {name}: prob={prob:.1f}, {status}")

print("\n计算过程：")
total = 0.0
for name, prob, sampled in edges_simple:
    if sampled:
        log_p = np.log(prob)
        total += log_p
        print(f"  {name}: log({prob:.1f}) = {log_p:.4f}")
    else:
        log_p = np.log(1 - prob)
        total += log_p
        print(f"  {name}: log(1-{prob:.1f}) = log({1-prob:.1f}) = {log_p:.4f}")

print(f"\n总 log_prob = {total:.4f}")

print("\n等价的概率计算（验证）：")
prob_total = 0.8 * (1 - 0.3) * 0.6
log_prob_total = np.log(prob_total)
print(f"  P(总) = 0.8 × (1-0.3) × 0.6 = 0.8 × 0.7 × 0.6 = {prob_total:.4f}")
print(f"  log P(总) = log({prob_total:.4f}) = {log_prob_total:.4f}")
print(f"  与上面的结果一致！")

print("\n" + "=" * 80)
print("代码对应")
print("=" * 80)

print("""
在 construct_spatial_connection 中：

```python
log_probs = [torch.tensor(0.0, requires_grad=optimized_spatial)]

for edge_logit, edge_mask in zip(self.spatial_logits, self.spatial_masks):
    if edge_mask == 0.0:
        continue  # 跳过，不计入 log_prob
    
    edge_prob = torch.sigmoid(edge_logit / temperature)
    
    if torch.rand(1) < edge_prob:
        # 采样到这条边
        out_node.add_successor(in_node, 'spatial')
        log_probs.append(torch.log(edge_prob))  # ← log(prob)
    else:
        # 没采样到这条边
        log_probs.append(torch.log(1 - edge_prob))  # ← log(1-prob)

return torch.sum(torch.stack(log_probs))  # ← 求和
```

关键点：
1. 遍历所有允许的边（edge_mask != 0）
2. 每条边都贡献一个 log_prob
3. 最后求和得到总 log_prob
4. 用于计算策略梯度：loss = -log_prob × utility
""")

print("\n" + "=" * 80)
print("为什么需要遍历所有边？")
print("=" * 80)

print("""
这是强化学习中策略梯度方法的要求：

1. 策略（Policy）：
   - 策略定义了在每个状态下采取每个动作的概率
   - 这里的"动作"是"选择哪些边"
   - 策略的概率 = P(选择这个图结构)

2. 联合概率：
   - P(图G) = P(边1) × P(边2) × ... × P(边N)
   - 其中 P(边i) = prob_i (如果采样) 或 (1-prob_i) (如果不采样)

3. 对数概率：
   - log P(图G) = log P(边1) + log P(边2) + ... + log P(边N)
   - 这就是为什么要遍历所有边并求和

4. 梯度更新：
   - ∇ log P(图G) 用于更新参数
   - 如果这个图结构导致好的结果，增加 P(图G)
   - 如果导致坏的结果，减少 P(图G)

所以必须遍历所有边，才能正确计算整个图结构的概率！
""")
