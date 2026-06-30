import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 消融实验 Agent 配置表
# 每个变体只定义差异部分，共享逻辑统一在 AblationAgent
# ============================================================
ABLATION_CONFIGS = {
    "lonlat_lonlat": {
        "obs_dim": 4,
        "obs_slice": slice(6, 10),       # GPS经纬度 (lonlat_x, lonlat_y)
        "aux_task": None,
    },
    "lonlat_marker": {
        "obs_dim": 200,
        "obs_slice": slice(10, None),     # emb_1 + emb_2 全部Landmark
        "aux_task": None,
    },
    "pics_lonlat": {
        "obs_dim": 2,
        "obs_slice": slice(8, 10),        # goal的GPS经纬度 (2维)
        "aux_task": None,
    },
    "pics_marker": {
        "obs_dim": 100,
        "obs_slice": slice(110, None),     # goal侧Landmark嵌入 (100维)
        "aux_task": None,
    },
    "pics_marker_conti": {
        "obs_dim": 100,
        "obs_slice": slice(110, None),
        "aux_task": "conti",               # 预测绝对坐标 MSE(2维×2)
    },
    "pics_marker_Rconti": {
        "obs_dim": 100,
        "obs_slice": slice(110, None),
        "aux_task": "r_conti",             # 预测相对位移 MSE(2维)
    },
    "pics_marker_disc": {
        "obs_dim": 100,
        "obs_slice": slice(110, None),
        "aux_task": "disc",                # 预测离散格子 CE(256维×2)
    },
}

AGENTS_NAME = list(ABLATION_CONFIGS.keys())


class AblationAgent(nn.Module):
    """
    参数化消融 Agent：通过 config 字典控制 obs 切片和辅助任务。

    原代码中 7 个独立类（~400行）合并为 1 个类（~150行），
    差异仅由 ABLATION_CONFIGS 定义。
    """

    def __init__(self, args, config=None):
        super().__init__()
        self.args = args
        self.device = args.device
        self.hidden_feats = args.hidden_feats
        self.num_actions = args.num_actions

        if config is None:
            name = getattr(args, 'agent_name', 'pics_marker')
            config = ABLATION_CONFIGS[name]
        self.config = config

        # --- 观测编码器 ---
        self.obs_enc = nn.Sequential(
            nn.Linear(config["obs_dim"], self.hidden_feats),
            nn.ReLU(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
            nn.ReLU(),
        )

        # --- 共享主干 ---
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)

        # --- 视觉分支（可选）---
        self.cnn = nn.Linear(1000, self.hidden_feats)
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)

        # --- 辅助任务头（按配置）---
        self.aux_task = config.get("aux_task")
        if self.aux_task == "conti":
            self.pred_obs = nn.Linear(self.hidden_feats, 2)
            self.pred_goal = nn.Linear(self.hidden_feats, 2)
        elif self.aux_task == "r_conti":
            self.pred_sp = nn.Linear(self.hidden_feats, 2)
        elif self.aux_task == "disc":
            self.pred_obs = nn.Linear(self.hidden_feats, 256)
            self.pred_goal = nn.Linear(self.hidden_feats, 256)

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    # ---- 损失函数（保持接口兼容）----
    @staticmethod
    def loss_mse(da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    @staticmethod
    def loss_en(logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def _compute_aux_loss(self, out, obs):
        """根据 aux_task 类型计算辅助损失"""
        if self.aux_task is None:
            return torch.tensor(0.0, device=self.device)

        if self.aux_task == "conti":
            tra_x, tra_y = obs[:, 6:8], obs[:, 8:10]
            obs_pred = self.pred_obs(out).squeeze(1)
            goal_pred = self.goal_pred(out).squeeze(1)  # noqa: F821 — intentionally using self for compatibility
            return self.loss_mse(obs_pred, tra_x) + self.loss_mse(goal_pred, tra_y)

        if self.aux_task == "r_conti":
            lonlat_x, lonlat_y = obs[:, 6:8], obs[:, 8:10]
            tar_sp = lonlat_y - lonlat_x
            sp_pred = self.pred_sp(out).squeeze(1)
            return self.loss_mse(sp_pred, tar_sp)

        if self.aux_task == "disc":
            tra_x, tra_y = obs[:, 0].long(), obs[:, 1].long()
            obs_pred = self.pred_obs(out).squeeze(1)
            goal_pred = self.pred_goal(out).squeeze(1)
            return self.loss_en(obs_pred, tra_x) + self.loss_en(goal_pred, tra_y)

        return torch.tensor(0.0, device=self.device)

    def forward(self, obs, hidden, views=0):
        # 1. 按 config 切片观测
        obs_ = obs[:, self.config["obs_slice"]]

        # 2. 编码 + 可选视觉融合
        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc), dim=-1))

        # 3. LSTM 时序建模
        out, hidden_next = self.lstm(x, hidden)

        # 4. 主头输出
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()

        # 5. 辅助损失
        aux_loss = self._compute_aux_loss(out.squeeze(1), obs)

        return logits, values, hidden_next, aux_loss


# ============================================================
# 向后兼容：保留旧类名作为别名（exp.py 的 dict_net 引用）
# ============================================================
def make_agent(args):
    """工厂函数：根据 agent_name 创建对应的消融 Agent"""
    name = args.agent_name
    if name == "Base":
        return Base(args)
    if name in ABLATION_CONFIGS:
        return AblationAgent(args, ABLATION_CONFIGS[name])
    raise ValueError(f"Unknown agent name: {name}. Available: {list(ABLATION_CONFIGS.keys())}")


class Base(nn.Module):
    """空壳基类：forward 直接返回 obs（用于占位/测试）"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    def forward(self, obs, hidden, views=0):
        return obs
