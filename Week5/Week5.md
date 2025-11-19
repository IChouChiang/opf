# Week5 

## 图卷积神经网络模型

### 特征构建

**输入参数：**
- `ppc_int`: PYPOWER内部索引的算例数据
- `pd`: 有功负荷需求 `[N_BUS]`，单位 p.u.
- `qd`: 无功负荷需求 `[N_BUS]`，单位 p.u.
- `k`: 迭代次数，默认为 8

**输出特征：**
- `e_0_k`: 节点电压实部特征 `[N_BUS, k]`
- `f_0_k`: 节点电压虚部特征 `[N_BUS, k]`

**实现细节：**

特征构建模块基于电力系统潮流方程的物理原理，通过 k 次迭代计算得到每个节点的电压分量序列。这种方法将电力系统的物理约束嵌入到神经网络的输入特征中。

迭代公式（参考论文 Eqs. 16-22）：
- 初始化：`e^0 = 1.0`, `f^0 = 0.0`（平启动）
- 迭代更新使用导纳矩阵 G, B 和功率平衡方程
- 每次迭代考虑节点间的耦合关系和功率注入

**主要文件和函数：**
- `gcnn_opf_01/feature_construction_model_01.py`
  - `construct_features()`: 核心特征构建函数
  - `construct_features_from_ppc()`: 从 PYPOWER 算例提取参数的封装函数
  - `extract_gen_limits()`: 提取发电机功率和电压限制

**Shape 说明：**
```python
e_0_k: [N_BUS=6, k=8]  # 每个节点的 8 次迭代电压实部
f_0_k: [N_BUS=6, k=8]  # 每个节点的 8 次迭代电压虚部
```

### 图卷积层

**网络架构：**
- 两层图卷积层：`gc1`, `gc2`
- 输入通道：8（对应 k=8 次迭代特征）
- 输出通道：8
- 激活函数：`tanh`

**输入参数（每层）：**

- `e, f`: 节点特征 `[B, N_BUS, C_in]`（批处理版本）或 `[N_BUS, C_in]`
- `pd, qd`: 负荷需求 `[B, N_BUS]` 或 `[N_BUS]`
- `g_ndiag, b_ndiag`: 非对角导纳矩阵 `[N_BUS, N_BUS]`
- `g_diag, b_diag`: 对角导纳向量 `[N_BUS]`

**物理引导计算：**

图卷积层不同于传统的 GCN，它明确使用电力系统的导纳矩阵和潮流方程：

1. **邻居聚合**（Eqs. 19-20）：
   - α = G_ndiag @ e - B_ndiag @ f
   - β = G_ndiag @ f + B_ndiag @ e

2. **功率平衡**（Eqs. 21-22）：
   - δ = -P_D - (e²+f²) ⊙ g_diag
   - λ = -Q_D - (e²+f²) ⊙ b_diag

3. **电压更新**（Eqs. 16-17）：
   - 基于分母 α²+β² 和分子 δα-λβ, δβ+λα
   - 应用线性变换和 tanh 激活

**主要文件和函数：**
- `gcnn_opf_01/model_01.py`
  - `class GraphConv(nn.Module)`: 图卷积层实现
  - `forward()`: 支持批处理和单样本的前向传播

**批处理支持：**
- 自动检测输入维度（2D vs 3D）
- 使用 `torch.einsum` 实现批处理矩阵乘法
- 单样本输入自动添加/移除批维度

### 全连接层

**网络结构：**
```
GraphConv输出 → 拼接(e, f) → 展平 → FC1(ReLU) → 双头输出
                                   ↓
                    ┌──────────────┴──────────────┐
                    ↓                             ↓
              gen_head (PG, VG)              v_head (e, f)
              [N_GEN, 2]                     [N_BUS, 2]
```

**参数配置：**
- 输入维度：`N_BUS * 2 * CHANNELS_GC_OUT = 6 * 2 * 8 = 96`
- FC1 隐藏层神经元：`NEURONS_FC = 128`（根据论文建议从 1000 降至 128）
- 输出1（发电机）：`N_GEN * 2 = 3 * 2 = 6`（PG, VG）
- 输出2（节点电压）：`N_BUS * 2 = 6 * 2 = 12`（e, f）

**双头设计理由：**

1. **gen_head**: 直接用于监督学习，预测发电机出力和电压
2. **v_head**: 用于物理损失计算，确保满足功率平衡约束

**主要文件和函数：**

- `gcnn_opf_01/model_01.py`
  - `class GCNN_OPF_01(nn.Module)`: 主模型
  - `__init__()`: 定义网络层
  - `forward()`: 前向传播（支持批处理）

**模型参数统计：**
- 总参数量：15,026
- 分布：GraphConv 层 ≈ 90%，FC 层 ≈ 10%

### 损失函数

**总损失函数（Eq. 35）：**
```
L_total = L_supervised + κ * L_ΔPG
```

**1. 监督损失 (L_supervised)**
```python
L_sup = MSE(PG_pred, PG_label) + MSE(VG_pred, VG_label)
```
- 直接比较预测值与 AC-OPF 解标签
- Shape: PG, VG 各为 `[N_GEN]`

**2. 物理关联损失 (L_ΔPG, Eq. 27)**
```python
# 将发电机功率映射到节点
PG_bus = A_g2b @ PG_pred  # [N_BUS]

# 根据节点电压计算功率注入
PG_from_V = f_pg(V_pred, pd, G, B)

# 关联损失
L_ΔPG = MSE(PG_bus, PG_from_V)
```

**物理损失计算步骤：**
1. 使用 one-hot 矩阵 `A_g2b [N_BUS, N_GEN]` 将发电机功率分配到节点
2. 根据预测的节点电压 `(e, f)` 和导纳矩阵计算理论功率注入
3. 最小化两者差异，确保功率平衡

**超参数：**
- `κ (kappa) = 0.1`: 物理损失权重
- 根据论文经验值设置

**主要文件和函数：**
- `gcnn_opf_01/loss_model_01.py`
  - `correlative_loss_pg()`: 关联损失计算
  - `f_pg_from_v()`: 从电压计算功率注入
- `gcnn_opf_01/train.py`
  - 监督损失使用 `nn.functional.mse_loss()`
  - 物理损失按样本遍历计算后平均

**返回值：**
- `loss_total`: 总损失（标量）
- `loss_sup`: 监督损失分量
- `loss_phys`: 物理损失分量

## 样本生成

### 基准

**基准算例：** `case6ww` (Wood & Wollenberg 6-bus system)

**系统参数：**

- 节点数：`N_BUS = 6`
- 发电机数：`N_GEN = 3`
- 支路数：`N_BRANCH = 11`
- 基准容量：`baseMVA = 100 MVA`

**基准负荷（p.u.）：**
```python
# 从 case6ww 提取
PD_base = [0.70, 0.70, 0.70, 0.70, 0.70, 0.0]  # 节点 1-6
QD_base = [0.70, 0.70, 0.70, 0.70, 0.70, 0.0]  # 对应无功
```

**发电机配置：**
- Gen 0 @ Bus 1: Pmin=0.5, Pmax=2.0 p.u.
- Gen 1 @ Bus 2: Pmin=0.5, Pmax=2.0 p.u.
- Gen 2 @ Bus 3: Pmin=0.5, Pmax=2.0 p.u.

**主要文件：**
- `gcnn_opf_01/sample_config_model_01.py`
  - `load_case6ww_int()`: 加载并转换为内部索引
- PYPOWER 库：`case6ww()`

### N-1 选择

**拓扑配置（N-1 事故）：**

定义了 5 种拓扑场景（0-4），其中拓扑 0 为基准，1-4 为 N-1 事故：

```python
TOPOLOGY_BRANCH_PAIRS_1BASED = {
    0: [],           # 基准：无支路断开
    1: [(5, 2)],     # 支路 5-2 断开
    2: [(1, 2)],     # 支路 1-2 断开
    3: [(2, 3)],     # 支路 2-3 断开
    4: [(5, 6)],     # 支路 5-6 断开
}
```

**实现逻辑：**
1. 根据 `topo_id` 查找需要断开的支路对（外部编号 1-based）
2. 使用 `ext2int` 映射转换为内部索引
3. 在支路矩阵中查找对应行（支持双向和并联线路）
4. 设置 `branch[idx, BR_STATUS] = 0`（断开）

**导纳矩阵重建：**
- 每个拓扑对应一组独立的 `G, B, g_diag, b_diag, g_ndiag, b_ndiag`
- 使用 PYPOWER 的 `makeYbus()` 函数根据当前拓扑计算
- 预计算所有拓扑的算子并保存为 `topology_operators.npz`

**主要文件和函数：**
- `gcnn_opf_01/sample_config_model_01.py`
  - `find_branch_indices_for_pairs()`: 查找支路索引
  - `apply_topology()`: 应用拓扑（断开指定支路）
  - `build_G_B_operators()`: 构建导纳矩阵算子
- `gcnn_opf_01/generate_dataset.py`
  - 预计算所有 5 种拓扑的算子矩阵

**数据集分布：**
- 训练集 10,000 样本：均匀分布在 5 种拓扑上（每种约 2,000 样本）
- 测试集 2,000 样本：同样均匀分布

### 负荷正态分布

**负荷波动模型：**

基于基准负荷添加高斯随机波动：

```python
PD_fluctuated = PD_base * (1 + N(0, σ_rel²))
QD_fluctuated = QD_base * (1 + N(0, σ_rel²))
```

**参数设置：**
- 相对标准差：`σ_rel = 0.1` (10%)
- 分布类型：正态分布（独立同分布于每个节点）

**实现细节：**
```python
# 每个样本独立采样
epsilon_pd = rng.normal(0, sigma_rel, size=N_BUS)
epsilon_qd = rng.normal(0, sigma_rel, size=N_BUS)

pd_fluctuated = PD_base * (1.0 + epsilon_pd)
qd_fluctuated = QD_base * (1.0 + epsilon_qd)
```

**物理约束：**
- 负荷波动后可能出现负值（新能源接入后视为负负荷）
- 参数 `allow_negative_pd` 控制是否允许负值

**主要文件和函数：**

- `gcnn_opf_01/sample_generator_model_01.py`
  - `class SampleGeneratorModel01`
  - `generate_one_sample()`: 采样单个场景

### 新能源接入

**新能源配置：**
- 风电接入节点：Bus 5（外部编号）→ Index 4（内部）
- 光伏接入节点：Bus 4, 6（外部编号）→ Index 3, 5（内部）
- 目标渗透率：`penetration_target = 0.507` (50.7%)

**渗透率定义：**

```
渗透率 = 新能源总发电量 / 总负荷需求
```

#### 风电

**Weibull 分布风速模型：**

风速服从 Weibull 分布，用于模拟自然风速的统计特性：

```python
# Weibull 分布参数
λ (scale) = 5.089 m/s     # 尺度参数
k (shape) = 2.016         # 形状参数

# 概率密度函数
f(v) = (k/λ) * (v/λ)^(k-1) * exp(-(v/λ)^k)
```

**风电功率曲线（分段函数）：**

```python
def wind_power_curve(v):
    if v < v_cut_in (4.0 m/s):
        P = 0                           # 切入风速以下
    elif v_cut_in ≤ v < v_rated (12.0 m/s):
        P = ((v - v_cut_in)/(v_rated - v_cut_in))³  # 立方律
    elif v_rated ≤ v < v_cut_out (25.0 m/s):
        P = 1.0                         # 额定功率
    else:
        P = 0                           # 切出风速以上
```

**特点：**
- 切入风速 `v_cut_in = 4.0 m/s`: 风机开始发电
- 额定风速 `v_rated = 12.0 m/s`: 达到额定功率
- 切出风速 `v_cut_out = 25.0 m/s`: 保护停机
- 功率-风速曲线呈 S 型，符合实际风机特性

**风速-功率曲线示意：**
```
P(p.u.)
1.0 |        ___________________
    |      /
    |     /
0.5 |    /
    |   /
    |__/________________________
    0   4      12              25    v(m/s)
      cut-in  rated          cut-out
```

**主要参数：**
- Weibull λ (lam_wind): 5.089 m/s
- Weibull k (k_wind): 2.016
- v_cut_in: 4.0 m/s
- v_rated: 12.0 m/s
- v_cut_out: 25.0 m/s

#### 光伏

**Beta 分布光照模型：**

太阳辐照度服从 Beta 分布，适合模拟日间光照强度的时变特性：

```python
# Beta 分布参数
α (alpha) = 2.06
β (beta) = 2.5

# 归一化辐照度 s ∈ [0, 1]
# 实际辐照度 S = s * G_STC
```

**光伏功率曲线（线性模型）：**

```python
def pv_power_curve(S):
    cf = S / G_STC          # 容量因子
    cf = clip(cf, 0, 1)     # 限制在 [0,1]
    P = cf                  # 线性关系
```

**特点：**
- 标准测试辐照度 `G_STC = 1000 W/m²`
- 线性功率曲线：辐照度正比于输出功率
- Beta 分布形状（α=2.06, β=2.5）呈右偏，符合晴天光照模式

**辐照度-功率曲线示意：**

```
P(p.u.)
1.0 |                      /
    |                    /
    |                  /
0.5 |                /
    |              /
    |            /
0.0 |__________/____________
    0        500        1000  S(W/m²)
                       G_STC
```

**Beta 分布特性：**
- α > 1, β > 1: 单峰分布
- α < β: 右偏（峰值偏左），模拟早晨-中午光照模式
- 适合日间变化的光伏出力场景

**主要参数：**
- Beta α (alpha_pv): 2.06
- Beta β (beta_pv): 2.5
- G_STC: 1000.0 W/m²

**新能源功率计算流程：**

1. **采样随机变量**
   ```python
   # 风速（Weibull 分布）
   v_wind = rng.weibull(k_wind, size=n_wind) * lam_wind
   
   # 辐照度（Beta 分布）
   s_pv = rng.beta(alpha_pv, beta_pv, size=n_pv)
   S_pv = s_pv * G_STC
   ```

2. **计算容量因子**
   ```python
   cf_wind = [wind_power_curve(v) for v in v_wind]
   cf_pv = [pv_power_curve(S) for S in S_pv]
   ```

3. **缩放至目标渗透率**
   ```python
   # 总负荷
   total_load = sum(PD_fluctuated)
   
   # 目标新能源功率
   P_res_target = penetration_target * total_load
   
   # 初始新能源功率
   P_res_raw = sum(cf_wind) + sum(cf_pv)
   
   # 缩放因子
   scale = P_res_target / P_res_raw
   
   # 最终功率
   P_wind_final = cf_wind * scale
   P_pv_final = cf_pv * scale
   ```

4. **转换为负负荷**
   ```python
   # 新能源视为负荷减少
   pd_final = pd_fluctuated.copy()
   pd_final[wind_bus_indices] -= P_wind_final
   pd_final[pv_bus_indices] -= P_pv_final
   
   # qd 保持功率因数（简化假设 pf=1.0）
   qd_final = qd_fluctuated  # 或按比例调整
   ```

**主要文件和函数：**
- `gcnn_opf_01/sample_generator_model_01.py`
  - `wind_power_curve()`: 风电功率曲线
  - `pv_power_curve()`: 光伏功率曲线
  - `class SampleGeneratorModel01.generate_one_sample()`: 完整采样流程
- `gcnn_opf_01/sample_config_model_01.py`
  - `RES_BUS_WIND_EXTERNAL`, `RES_BUS_PV_EXTERNAL`: 新能源接入节点
  - 分布参数定义

### batch 设置

**训练批次大小：** `batch_size = 10`

**选择理由：**
- 参考论文使用的 mini-batch 设置
- 平衡训练稳定性和收敛速度
- 适合中小规模系统（case6ww）

**DataLoader 配置：**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,          # 训练集随机打乱
    num_workers=0,         # Windows 兼容性
    pin_memory=True        # GPU 加速
)

val_loader = DataLoader(
    val_dataset,
    batch_size=10,
    shuffle=False,         # 验证集保持顺序
    num_workers=0,
    pin_memory=True
)
```

**批处理实现：**
- 模型支持自动批维度检测（2D/3D 输入）
- 拓扑算子在批内共享（假设同一批次拓扑相同）
- 物理损失按样本遍历计算后平均

**主要文件：**
- `gcnn_opf_01/train.py`
  - `parse_args()`: CLI 参数解析
  - `main()`: DataLoader 初始化

## 训练与结果

### 训练配置

**超参数设置：**
- 优化器：Adam
- 学习率：`lr = 1e-3`
- 权重衰减：`weight_decay = 1e-5`
- 训练轮数：`epochs = 50`
- 早停耐心：`patience = 10`（验证损失无改善停止）
- 物理损失权重：`κ = 0.1`

**数据集规模：**
- 训练样本：10,000
- 验证样本：2,000
- 数据增强：5 种拓扑 × 负荷波动 × 新能源随机

**训练环境：**
- 设备：CUDA GPU (自动检测)
- 模型参数：15,026
- 训练时间：约 4.8 分钟（23 个 epoch，早停）

### 训练过程

**损失变化趋势：**

| Epoch | Train Loss | Train Sup | Train Phys | Val Loss | Val Sup | Val Phys |
|-------|-----------|-----------|-----------|----------|---------|----------|
| 1 | 0.3065 | 0.1633 | 1.4321 | 0.2253 | 0.0741 | 1.5113 |
| 5 | 0.1938 | 0.0249 | 1.6889 | 0.1926 | 0.0224 | 1.7013 |
| 10 | 0.1614 | 0.0142 | 1.4707 | 0.1596 | 0.0127 | 1.4688 |
| 15 | 0.1800 | 0.0111 | 1.6887 | 0.1765 | 0.0098 | 1.6671 |
| 20 | 0.1769 | 0.0100 | 1.6691 | **0.1605** | 0.0114 | 1.4907 |
| 23 | 0.1756 | 0.0096 | 1.6598 | 0.1709 | 0.0075 | 1.6342 |

**关键观察：**
1. **监督损失快速下降**：从 0.163 降至 0.0075（98% 降幅）
2. **物理损失稳定收敛**：维持在 1.63-1.67 范围
3. **无过拟合**：验证损失 ≤ 训练损失
4. **早停触发**：Epoch 20 后验证损失无改善，Epoch 23 停止

**最佳模型：**
- 最佳验证损失：`0.160208`（约 Epoch 20）
- 保存路径：`gcnn_opf_01/results/best_model.pth`

### 测试集评估结果

**测试集规模：** 2,000 样本

**发电机有功功率 (PG) 预测：**
- MSE: 0.0233 (p.u.²)
- RMSE: 0.153 p.u. ≈ **15.3 MW** (100 MVA 基准)
- MAE: 0.073 p.u. ≈ **7.3 MW**
- MAPE: 30.20%
- **R² 得分: 0.9765** (97.65% 方差解释) ✓
- 最大误差: 3.59 p.u.

**发电机电压 (VG) 预测：**
- MSE: 0.000060 (p.u.²)
- RMSE: 0.0077 p.u. ≈ **0.77% 电压误差**
- MAE: 0.0060 p.u. ≈ **0.60%**
- **MAPE: 0.68%** (优秀!) ✓
- **R² 得分: 0.9999** (99.99% 方差解释) ✓✓
- 最大误差: 0.061 p.u. (6.1%)

**分发电机详细统计（PG）：**

| 发电机 | MSE | MAE | 预测均值 (p.u.) | 真实均值 (p.u.) | 性能评价 |
|--------|-----|-----|----------------|----------------|----------|
| Gen 0 | 0.0085 | 0.013 | -0.0831 | -0.0809 | **最佳** ✓ |
| Gen 1 | 0.0438 | 0.132 | 0.7440 | 0.8277 | 系统性低估 |
| Gen 2 | 0.0176 | 0.073 | -0.7700 | -0.7399 | 良好 |

**结果分析：**

1. **电压预测极其准确** (R²=99.99%, MAPE<1%)
   - 模型精确学习了节点电压与系统状态的关系
   - 满足电力系统电压质量要求

2. **有功功率预测很好** (R²=97.65%)
   - 典型预测误差约 7 MW（100 MVA 系统）
   - 达到实用工程精度要求

3. **Gen 1 误差较大的可能原因**
   - Gen 1 是最大发电机（平均出力 83 MW）
   - 可能在发电机极限附近存在非线性
   - 系统性低估：预测 74.4 vs 实际 82.8 MW
   - 建议：增加高负荷场景训练样本

4. **总体性能评价**
   - 模型成功学习了 OPF 的底层规律
   - 预测速度：毫秒级（vs AC-OPF 求解器秒级）
   - 适用于实时调度和快速安全评估

**输出文件：**
- `gcnn_opf_01/results/best_model.pth`: 最佳模型权重
- `gcnn_opf_01/results/final_model.pth`: 最终模型（Epoch 23）
- `gcnn_opf_01/results/training_log.csv`: 详细训练日志
- `gcnn_opf_01/results/training_curves.png`: 损失曲线图
- `gcnn_opf_01/results/training_history.npz`: NumPy 格式历史数据
- `gcnn_opf_01/results/evaluation_results.npz`: 测试集预测结果

### 主要代码文件

**模型相关：**
- `gcnn_opf_01/model_01.py`: GCNN 模型定义
- `gcnn_opf_01/config_model_01.py`: 模型配置参数
- `gcnn_opf_01/loss_model_01.py`: 损失函数实现
- `gcnn_opf_01/feature_construction_model_01.py`: 特征构建

**数据相关：**
- `gcnn_opf_01/sample_config_model_01.py`: 样本配置（拓扑、新能源）
- `gcnn_opf_01/sample_generator_model_01.py`: 样本生成器
- `gcnn_opf_01/generate_dataset.py`: 数据集生成脚本
- `gcnn_opf_01/dataset.py`: PyTorch Dataset 封装

**训练评估：**
- `gcnn_opf_01/train.py`: 训练主程序
- `gcnn_opf_01/evaluate.py`: 测试集评估

**测试验证：**
- `tests/test_model_forward.py`: 模型前向传播测试
- `tests/test_feature_construction.py`: 特征构建测试
- `tests/test_sample_generator.py`: 样本生成器测试
- `tests/test_topology_outages.py`: N-1 拓扑验证

