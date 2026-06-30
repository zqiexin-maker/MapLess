# 🗺️ MapLess：基于PPO的无地图视觉强化学习导航系统

> **一句话概括**：智能体只看"当前街景照片 + 目标位置"，无需地图，自己学会在曼哈顿街道中找路。

---

## 一、项目背景与动机

### 我们要解决什么问题？

传统导航高度依赖预构建地图（如 Google Maps）。但现实中：
- 地图可能过时、缺失或无法获取
- 机器人/智能体需要**像人一样**：看一眼当前场景 + 知道目标在哪 → 决定下一步怎么走

**核心挑战**：如何让智能体仅凭视觉感知完成城市级路径导航？

### 数据来源

| 项目 | 内容 |
|------|------|
| 数据集 | Google StreetLearn 曼哈顿街景 |
| 规模 | 55,761 张全景图，LevelDB 存储 |
| 地理范围 | 曼哈顿 [-74.035, -73.935, 40.695, 40.795] |
| 图结构 | 每个节点有 2~4 个相邻节点（街道连接） |

---

## 二、整体框架一览

```
┌─────────────────────────────────────────────────────────┐
│                     MapLess 系统                         │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  数据层   │───▶│   环境层      │───▶│   算法层       │  │
│  │ cache.py │    │environment.py│    │    exp.py     │  │
│  │dataloader│    │              │    │  PPOAgent     │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│                          │                   │           │
│                  ┌───────▼───────────────────▼───────┐  │
│                  │           神经网络层                 │  │
│                  │   agent.py (AgentPosi)              │  │
│                  │   ResNet18 + LSTM + 多任务头         │  │
│                  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 三、文件结构与职责

| 文件 | 职责 | 关键类/函数 |
|------|------|------------|
| `main_.py` | **训练入口**，解析参数，启动训练 | `argparse` |
| `main_test.py` | **测试入口**，消融实验专用 | — |
| `exp.py` | **PPO 训练/测试核心逻辑** | `Exps`, `PPOAgent` |
| `agent.py` | **主 Agent 神经网络** | `AgentPosi` |
| `agent_comp.py` | **7 种消融对比 Agent** | `lonlat_lonlat`, `pics_marker`... |
| `environment.py` | **三类强化学习环境** | `GridEnv`, `StreetEnv`, `StreetPicEnv` |
| `cache.py` | **数据预处理与缓存** | `Cache_DB`, `Landmark` |
| `dataloader.py` | **街景图像加载** | `Picdataset_Realtime`, `Picdataset_Cache` |
| `utils.py` | **工具函数** | Haversine 距离、航向计算、归一化 |
| `draw.py` | **轨迹 + 奖励曲线可视化** | — |
| `visio.py` | **多模型对比图** | — |

---

## 四、三类实验环境

### 4.1 环境层级

```
GridEnv（网格环境）
  └── 4×4 方格，状态 = [agent_x, agent_y, goal_x, goal_y]
  └── 用途：调试算法基础逻辑

StreetEnv（街道环境·无视觉）
  └── 真实曼哈顿图结构，状态 = 归一化经纬度
  └── 用途：消融实验，验证坐标表示的贡献

StreetPicEnv（街道环境·含视觉）⭐ 主实验
  └── 真实图结构 + 街景图像 + Landmark 嵌入
  └── 状态向量：212 维（见下）
```

### 4.2 主实验状态向量（212维）

| 维度 | 内容 | 说明 |
|------|------|------|
| 0–1 | `labels_mark` | 网格离散化标签（16×16） |
| 2–5 | `grids(coor)` | 连续网格坐标 [0,16] |
| 6–9 | `scaler.trans(coor)` | 标准化经纬度 |
| 10 | `heading / 360` | 归一化连续航向 |
| 11 | `heading × 16 / 360` | 离散航向 bin（0–15） |
| 12–111 | `emb_1`（当前位置） | **100维 Landmark 高斯核嵌入** |
| 112–211 | `emb_2`（目标位置） | **100维 Landmark 高斯核嵌入** |

### 4.3 三个地理实验区域

| 区域 | 地理范围 | 约束 | 难度 |
|------|----------|------|------|
| `min` | ~0.4×0.6 km | 起止距离 ≥ 40m | ⭐ 入门 |
| `mid` | ~1.2×1.0 km | 起止距离 ≥ 400m | ⭐⭐ 中等 |
| `max` | ~7.7×11.1 km（全曼哈顿） | 无额外限制 | ⭐⭐⭐ 全域 |

---

## 五、核心创新：Landmark 高斯核嵌入

### 🤔 问题：为什么不直接用 GPS 坐标？

- 绝对经纬度在不同区域数值差异大，难以泛化
- GPS 坐标无法体现"当前位置与周边环境的相对关系"

### 💡 解法：Landmark 参考点嵌入

```
1. 初始化：在实验区域内随机采样 100 个节点作为"地标"

2. 嵌入计算（对每个查询位置 p）：
   for i in range(100):
       d_i = Haversine距离(p, landmark_i)    # 单位：米
       emb_i = exp(-d_i × 0.002)             # 高斯核变换
   
   emb = Z-score 标准化(emb)                  # 零均值单位方差

3. 结果：每个位置 → 100维向量
         "与各地标的远近关系" 作为空间指纹
```

**效果**：将绝对坐标变为相对位置描述，对区域内任意位置都有稳定表示。

---

## 六、神经网络架构（AgentPosi）

```
输入层
  ├─ obs[32:]（19维状态切片）
  └─ views（ResNet18输出的1000维视觉特征，可选）

特征提取
  ├─ 状态编码器：Linear(20,128) → Tanh → Linear(128,128)
  └─ 视觉编码器：Linear(1000,128)（ResNet18后接FC）

特征融合
  └─ concat → Linear(256,128) → Tanh

时序建模
  └─ LSTM(128, hidden=128)  ← 记忆历史轨迹信息

多任务输出头
  ├─ actor(128→4)       策略头：输出4个方向的动作概率
  ├─ critic(128→1)      价值头：估计当前状态价值 V(s)
  ├─ pred_obs(128→2)    辅助头①：预测当前位置经纬度
  ├─ pred_goal(128→2)   辅助头②：预测目标位置经纬度
  └─ pred_sp(128→16)    辅助头③：预测离散航向 bin（分类）
```

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{policy} + 0.5 \cdot \mathcal{L}_{value} - 0.004 \cdot H(\pi) + \mathcal{L}_{aux}$$

$$\mathcal{L}_{aux} = \text{MSE}(\hat{pos}, pos) + \text{MSE}(\hat{goal}, goal) + \text{CE}(\hat{sp}, sp)$$

**设计思路**：辅助任务迫使 LSTM 隐状态中必须编码"我在哪 / 目标在哪 / 我朝哪"三个关键信息，从而提升导航策略的空间感知能力。

---

## 七、PPO 训练流程

### 7.1 关键超参数

| 参数 | 值 | 含义 |
|------|----|------|
| `total_steps` | 500,000 | 总训练步数 |
| `rollout_len` | 1,500 | 每次采样步数 |
| `max_steps` | 150 | 单 episode 最大步数 |
| `learning_rate` | 0.001 | Adam 优化器学习率 |
| `gamma` | 0.995 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ（优势估计平滑） |
| `clip_eps` | 0.2 | PPO 裁剪阈值 |
| `update_times` | 6 | 每次 rollout 后更新轮数 |
| `landmark_num` | 100 | 地标数量 |
| `goal_range` | 5m | 到达目标的判定阈值 |

### 7.2 训练循环（伪代码）

```python
while step < 500_000:
    # Step 1: 采样 rollout
    for t in range(1500):
        action, logprob, value = agent.act(obs, hidden_state)
        next_obs, reward, done = env.step(action)
        buffer.store(obs, action, reward, value, logprob, done)
        step += 1
    
    # Step 2: 计算优势（GAE）
    advantages = compute_gae(last_value, gamma=0.995, lambda=0.95)
    returns = advantages + values
    
    # Step 3: PPO 更新（6轮）
    for _ in range(6):
        ratio = π_new(a|s) / π_old(a|s)
        policy_loss = -min(ratio * adv, clip(ratio, 0.8, 1.2) * adv)
        value_loss  = 0.5 * max(|V - R|², |V_clip - R|²)
        total_loss  = policy_loss + 0.5*value_loss - 0.004*entropy + aux_loss
        optimizer.step(total_loss)
```

### 7.3 奖励函数设计

```python
# 每步奖励 = 距离缩减奖励 + 步数惩罚
reward = (prev_distance - cur_distance) × 0.02
reward += -0.02                    # 鼓励尽快到达

# 到达奖励
if distance < 5m:
    reward += 1.0
    done = True
```

### 7.4 动作执行机制

```
选择动作 k ∈ {0,1,2,3}  →  偏移方向 {0°, 90°, 180°, 270°}
         ↓
获取当前节点所有邻居的真实航向角
         ↓
选取与目标方向角度差最小的邻居  →  移动过去
```

> 这个设计使得动作空间在真实街道图上保持语义一致性："前进/右转/后退/左转"。

---

## 八、消融实验设计（7种 Agent 变体）

| Agent 名称 | 输入模态 | 辅助任务 | 实验目的 |
|------------|----------|----------|----------|
| `lonlat_lonlat` | 绝对经纬度 (4维) | 无 | 纯坐标基线 |
| `lonlat_marker` | Landmark嵌入 (200维) | 无 | 纯 Landmark 基线 |
| `pics_lonlat` | 图像 + 目标坐标 | 无 | 图像 + 坐标 |
| `pics_marker` | 图像 + 目标 Landmark | 无 | 图像 + Landmark |
| `pics_marker_conti` | 图像 + 目标 Landmark | 预测连续坐标 | 连续辅助 |
| `pics_marker_Rconti` | 图像 + 目标 Landmark | 预测相对位移 | 相对辅助 |
| `pics_marker_disc` | 图像 + 目标 Landmark | 预测离散网格 | 离散辅助 |
| **AgentPosi（完整）** | 图像 + 双端 Landmark | 位置+目标+航向 | **最终方案** ⭐ |

### 实验结果（min 区域，任务完成率）

| 配置 | 完成率 |
|------|--------|
| pics + marker（无辅助） | 0.45 |
| pics + marker + pred_sp（航向辅助） | 0.65–0.72 |
| **pics + marker + pred_lonlat（完整）** | **0.67–0.75** ✅ |
| pics + marker + pred_emb | 0.50 |

---

## 九、图像处理流水线

```
全景图（LevelDB）
       │ plyvel 读取 + Protobuf 解析
       ↓
等距柱面投影图（2048×1024）
       │ 根据 agent 当前朝向截取视野
       ↓
透视图（FOV=120°，84×84 像素）
       │ py360convert
       ↓
ResNet18（预训练，去掉最后分类层）
       │ torchvision
       ↓
1000维视觉特征向量
       │ Linear(1000,128) + Tanh
       ↓
128维视觉编码  → 与状态特征 concat → 融合表示
```

**两种加载策略**：

| 模式 | 类名 | 策略 | 适用场景 |
|------|------|------|----------|
| `rt`（实时） | `Picdataset_Realtime` | LRU 缓存(100张)，按需解码 | 内存受限 |
| `cc`（预缓存） | `Picdataset_Cache` | 启动时全量编码为特征向量 | 训练提速 |

---

## 十、系统运行流程（完整版）

```
▶ 启动 main_.py
       │
       ├─ 1. 数据预处理（cache.py）
       │     ├─ 从 LevelDB 读取所有节点的经纬度、邻居关系 → static_info.npy
       │     ├─ 随机采样 100 个 Landmark → landmark.npy
       │     └─ 为每个节点预计算邻居索引 → neigs_index.npy
       │
       ├─ 2. 构建训练组件（exp.py::Exps.build_()）
       │     ├─ 初始化 StreetPicEnv 环境
       │     ├─ 加载图像数据（rt/cc 模式）
       │     └─ 构建 AgentPosi 网络 + Adam 优化器
       │
       ├─ 3. PPO 训练循环（exp.py::Exps.train()）
       │     └─ 采样 → GAE → 多次更新 → 保存 checkpoint
       │
       └─ 4. 测试评估（exp.py::Exps.test()）
             ├─ 采样 9 条轨迹 → draw.py 可视化轨迹图
             ├─ 运行 300 episodes → 计算完成率 + 平均步数
             └─ 结果写入 metric.csv
```

---

## 十一、总结：创新点归纳

| # | 创新点 | 核心思想 |
|---|--------|----------|
| ① | **Landmark 高斯核嵌入** | 用随机参考点将绝对 GPS 转化为相对空间指纹，解决坐标泛化问题 |
| ② | **多任务辅助学习** | 同时预测自身位置、目标位置、航向，引导 LSTM 建立显式空间表示 |
| ③ | **LSTM 时序建模** | 记录历史轨迹，让智能体"知道自己走过哪里" |
| ④ | **视觉-状态融合** | ResNet18 视觉特征与 Landmark 嵌入联合训练，实现感知与推理协同 |
| ⑤ | **课程式实验设计** | min→mid→max 三级区域，从局部验证到全域泛化 |

---

*文档生成时间：2026-06-23 | 作者：Yang Kaisen*
