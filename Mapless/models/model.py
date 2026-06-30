import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class AgentPosi(nn.Module):
    """
    完整版导航 Agent：Landmark嵌入 + ResNet18视觉 + 3个辅助任务

    观测向量 obs (212维) 布局:
        [0:2]     labels      离散格子编号 (2维)
        [2:6]     grids       归一化网格坐标 (4维)
        [6:10]    lonlat      标准化GPS经纬度 (4维)
        [10]      sp          朝向/360 (1维)
        [11]      sp_bin      朝向离散格 (1维)
        [12:112]  emb_1       agent位置Landmark高斯核嵌入 (100维)
        [112:212] emb_2       goal位置Landmark高斯核嵌入 (100维)

    网络结构:
        obs[12:22] + obs[112:122] → MLP(20→128) ─┐
                                                ├→ cat → fusion → LSTM(128) → actor/critic + 3辅助头
        views → ResNet18 → Linear(1000→128) ────┘

    辅助任务（共享参数，无门控）:
        pred_obs:   预测agent绝对坐标  MSE(2维)
        pred_goal:  预测goal绝对坐标   MSE(2维)
        pred_sp:    预预测向离散格     CE(16类)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.state_feats = getattr(args, 'state_fests', 20)
        self.num_actions = args.num_actions
        self.hidden_feats = args.hidden_feats

        # --- 观测编码：取 agent+goal 各前10维 Landmark 嵌入 ---
        self.obs_enc = nn.Sequential(
            nn.Linear(20, self.hidden_feats),
            nn.Tanh(),
            nn.Linear(self.hidden_feats, self.hidden_feats),
        )
        self.ten = nn.Tanh()

        # --- 共享主干 ---
        self.lstm = nn.LSTM(self.hidden_feats, self.hidden_feats)

        # --- 输出头 ---
        self.actor = nn.Linear(self.hidden_feats, self.num_actions)
        self.critic = nn.Linear(self.hidden_feats, 1)

        # --- 辅助任务头（3个，全部挂在 LSTM 输出 out 上）---
        self.pred_obs = nn.Linear(self.hidden_feats, 2)    # agent 绝对坐标
        self.pred_goal = nn.Linear(self.hidden_feats, 2)   # goal 绝对坐标
        self.pred_sp = nn.Linear(self.hidden_feats, 16)    # 朝向分类

        # --- 视觉分支 ---
        self.cnn = nn.Linear(1000, self.hidden_feats)      # ResNet18 特征投影
        self.fusion = nn.Linear(self.hidden_feats * 2, self.hidden_feats)

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_feats, device=self.device)
        return (h0, c0)

    @staticmethod
    def loss_mse(da, da1):
        return torch.sum((da - da1) ** 2, dim=-1).mean()

    @staticmethod
    def loss_en(logits, labels):
        return F.cross_entropy(logits, labels, reduction='sum')

    def forward(self, obs, hidden, views=None):
        if views is None:
            views = 0

        # 1. 解析观测分量
        # bins_x, bins_y = obs[:, 0].long(), obs[:, 1].long()
        # grids_x, grids_y = obs[:, 2:4], obs[:, 4:6]
        lonlat_x, lonlat_y = obs[:, 6:8], obs[:, 8:10]
        tra_x, tra_y = lonlat_x, lonlat_y
        tar_sp = obs[:, 11].long()

        # 2. Landmark嵌入切片：agent 前10维 + goal 前10维 = 20维
        obs_1 = obs[:, 12:22]    # emb_1[:10] (agent)
        obs_2 = obs[:, 112:122]  # emb_2[:10] (goal)
        obs_ = torch.cat([obs_1, obs_2], dim=-1)

        # 3. 编码 + 可选视觉融合
        x = self.obs_enc(obs_).unsqueeze(1)
        if isinstance(views, torch.Tensor):
            view_enc = self.cnn(views).unsqueeze(1)
            x = self.fusion(torch.cat((x, view_enc), dim=-1))

        # 4. LSTM
        out, hidden_next = self.lstm(x, hidden)
        out = self.ten(out)

        # 5. 主头输出
        logits = self.actor(out).squeeze()
        values = self.critic(out).squeeze()

        # 6. 辅助损失
        sp_pred = self.pred_sp(out).squeeze(1)
        loss_sp = self.loss_en(sp_pred, tar_sp)

        obs_pred = self.pred_obs(out).squeeze(1)
        goal_pred = self.pred_goal(out).squeeze(1)
        loss_coord = self.loss_mse(obs_pred, tra_x) + self.loss_mse(goal_pred, tra_y)

        return logits, values, hidden_next, loss_coord + loss_sp
