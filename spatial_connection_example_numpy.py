"""
详细解释 construct_spatial_connection 的工作原理（纯Python版本）
"""
import numpy as np

# ============================================================================
# 第一步：计算 spatial_logits（边的得分矩阵）
# ============================================================================

# 假设有 4 个节点
N = 4  # 节点数
D = 16  # 特征维度

# GCN + MLP 输出的节点嵌入
np.random.seed(42)
logits = np.random.randn(N, D)  # [4, 16]
print("=" * 80)
print("步骤 1: 计算边的 logits 矩阵")
print("=" * 80)
print(f"节点嵌入 logits 形状: {logits.shape}")
print(f"logits (前3行，前5列):\n{logits[:3, :5]}\n")

# 计算边的得分矩阵（对称矩阵）
spatial_logits_matrix = logits @ logits.T  # [4, 4]
print(f"spatial_logits_matrix = logits @ logits.T")
print(f"形状: {spatial_logits_matrix.shape}")
print(f"矩阵:\n{spatial_logits_matrix}\n")

# 验证对称性
is_symmetric = np.allclose(spatial_logits_matrix, spatial_logits_matrix.T)
print(f"是否对称: {is_symmetric}")
print(f"例如 [0,1] = {spatial_logits_matrix[0,1]:.4f}, [1,0] = {spatial_logits_matrix[1,0]:.4f}\n")

# Min-Max 归一化到 [-1, 1]
def min_max_norm(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1

# 展平并归一化
spatial_logits_flat = spatial_logits_matrix.flatten()  # [16]
spatial_logits_normalized = min_max_norm(spatial_logits_flat)  # [-1, 1]

print(f"展平后的 spatial_logits: {spatial_logits_flat.shape}")
print(f"归一化后的范围: [{spatial_logits_normalized.min():.4f}, {spatial_logits_normalized.max():.4f}]\n")

# ============================================================================
# 第二步：construct_spatial_connection - 采样边
# ============================================================================

print("=" * 80)
print("步骤 2: construct_spatial_connection - 根据概率采样边")
print("=" * 80)

# 模拟 potential_spatial_edges（所有可能的边）
potential_spatial_edges = []
for i in range(N):
    for j in range(N):
        potential_spatial_edges.append([f"node_{i}", f"node_{j}"])

print(f"潜在边的数量: {len(potential_spatial_edges)}")
print(f"前几条边: {potential_spatial_edges[:5]}\n")

# 模拟 spatial_masks（哪些边是允许的）
# 假设对角线不允许自连接，其他都允许
spatial_masks = np.ones(N * N)
for i in range(N):
    spatial_masks[i * N + i] = 0  # 对角线设为 0（不允许自连接）

print(f"spatial_masks (0=不允许, 1=允许):")
print(spatial_masks.reshape(N, N))
print()

# Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 采样参数
temperature = 1.0
optimized_spatial = True

# 开始采样
print("开始采样边...")
print("-" * 80)

sampled_edges = []
log_probs = [0.0]

np.random.seed(123)  # 固定随机种子以便复现

for idx, (potential_connection, edge_logit, edge_mask) in enumerate(
    zip(potential_spatial_edges, spatial_logits_normalized, spatial_masks)
):
    out_node = potential_connection[0]
    in_node = potential_connection[1]
    
    # 跳过不允许的边
    if edge_mask == 0.0:
        print(f"边 {idx:2d}: {out_node} → {in_node} | SKIP (mask=0, 自连接)")
        continue
    
    # 如果不优化且 mask=1，直接添加
    if edge_mask == 1.0 and not optimized_spatial:
        sampled_edges.append((out_node, in_node))
        print(f"边 {idx:2d}: {out_node} → {in_node} | ADD (固定边)")
        continue
    
    # 计算边的概率
    edge_prob = sigmoid(edge_logit / temperature)
    
    # 随机采样
    random_value = np.random.rand()
    
    if random_value < edge_prob:
        sampled_edges.append((out_node, in_node))
        log_probs.append(np.log(edge_prob))
        status = "✓ SAMPLED"
    else:
        log_probs.append(np.log(1 - edge_prob))
        status = "✗ REJECTED"
    
    print(f"边 {idx:2d}: {out_node} → {in_node} | "
          f"logit={edge_logit:6.3f} → prob={edge_prob:.3f} | "
          f"rand={random_value:.3f} | {status}")

total_log_prob = sum(log_probs)
print("-" * 80)
print(f"\n采样结果:")
print(f"  总共采样了 {len(sampled_edges)} 条边")
print(f"  总 log_prob: {total_log_prob:.4f}")
print(f"\n采样的边:")
for edge in sampled_edges:
    print(f"  {edge[0]} → {edge[1]}")

# ============================================================================
# 第三步：可视化概率分布
# ============================================================================

print("\n" + "=" * 80)
print("步骤 3: 概率分布可视化")
print("=" * 80)

# 重新计算所有边的概率（用于可视化）
edge_probs_matrix = sigmoid(spatial_logits_normalized.reshape(N, N) / temperature)

print("边的概率矩阵 (sigmoid(logits)):")
print(edge_probs_matrix)
print()

print("概率矩阵的统计:")
print(f"  最小概率: {edge_probs_matrix.min():.4f}")
print(f"  最大概率: {edge_probs_matrix.max():.4f}")
print(f"  平均概率: {edge_probs_matrix.mean():.4f}")
print(f"  中位数: {np.median(edge_probs_matrix):.4f}")

# ============================================================================
# 第四步：对称性验证
# ============================================================================

print("\n" + "=" * 80)
print("步骤 4: 验证对称性")
print("=" * 80)

print("因为 spatial_logits = Z @ Z.T 是对称的，所以:")
print(f"  P(node_0 → node_1) = {edge_probs_matrix[0, 1]:.4f}")
print(f"  P(node_1 → node_0) = {edge_probs_matrix[1, 0]:.4f}")
print(f"  是否相等: {np.allclose(edge_probs_matrix[0, 1], edge_probs_matrix[1, 0])}")

print("\n注意: 虽然概率是对称的，但采样结果不一定对称！")
print("例如: 可能采样到 node_0 → node_1，但没采样到 node_1 → node_0")

# ============================================================================
# 第五步：不同 temperature 的影响
# ============================================================================

print("\n" + "=" * 80)
print("步骤 5: Temperature 的影响")
print("=" * 80)

test_logit = 2.0
for temp in [0.5, 1.0, 2.0, 5.0]:
    prob = sigmoid(test_logit / temp)
    print(f"Temperature = {temp:.1f}: logit={test_logit:.1f} → prob={prob:.4f}")

print("\n解释:")
print("  - temperature 越小 → 概率越极端（接近 0 或 1）")
print("  - temperature 越大 → 概率越平滑（接近 0.5）")
print("  - temperature = 1.0 是标准的 sigmoid")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("""
1. spatial_logits = Z @ Z.T 是对称矩阵
   - 意味着 P(i→j) = P(j→i)
   
2. construct_spatial_connection 的流程:
   a) 遍历所有潜在的边 (i, j)
   b) 检查 edge_mask: 如果为 0，跳过
   c) 计算概率: prob = sigmoid(logit / temperature)
   d) 随机采样: if rand() < prob, 添加这条边
   e) 记录 log_prob 用于训练
   
3. 采样是随机的:
   - 即使 P(i→j) = P(j→i)，采样结果可能不对称
   - 每次运行结果可能不同
   
4. 用于强化学习:
   - log_probs 用于计算策略梯度
   - 通过 REINFORCE 算法优化边的选择
   
5. 关键代码对应:
   self.spatial_logits = logits @ logits.t()  # 对称矩阵
   edge_prob = torch.sigmoid(edge_logit / temperature)  # 转为概率
   if torch.rand(1) < edge_prob:  # 随机采样
       out_node.add_successor(in_node, 'spatial')  # 添加边
""")
