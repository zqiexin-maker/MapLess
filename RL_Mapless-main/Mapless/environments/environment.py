import numpy as np
import torch
from utils import Calculating, Standard
from cache import Landmark
from dataloader import Picdataset_Realtime, Picdataset_Cache


# ============================================================
# 学习区域预设
# ============================================================
LEARNING_EXTENTS = {
    "max": [-74.035, -73.935, 40.695, 40.795],
    "mid": [-73.990, -73.976, 40.756, 40.765],
    "min": [-73.990, -73.985, 40.750, 40.755],
}

RESET_CONDITIONS = {
    "min": {"min_distance": 40},
    "mid": {"min_distance": 400},
}


# ============================================================
# 街景导航环境基类（StreetEnv / StreetPicEnv 共享逻辑）
# ============================================================
class BaseStreetEnv:
    """曼哈顿街景图网络的共享骨架"""

    def _init_shared(self):
        self.infos = np.load(self.cache_root + '/static_info.npy', allow_pickle=True)
        self.neig = np.load(self.cache_root + '/neig_info.npy', allow_pickle=True)
        self.edge = np.load(self.cache_root + '/neigs_index.npy')

        # 区域过滤
        if self.le is not None:
            area_index = np.load(self.cache_root + '/area_index.npy')
            self.infos = self.infos[area_index]
            self.neig = self.neig[area_index]

        self.le = torch.tensor(self.le, device=self.device)
        self.re_le = self.le + (self.le[[1, 0, 3, 2]] - self.le) * 0.2
        self.length = len(self.neig)

        # 节点坐标 (lng, lat, heading_deg) → tensor
        self.nodes = self.infos[:, [1, 2, 8]].astype('float32')
        self.nodes = torch.tensor(self.nodes).to(self.device)
        self.scaler.fit(self.nodes[:, :2])
        self.edge = torch.tensor(self.edge).to(self.device)

    def _reset_conditions(self):
        """检查 agent-goal 对是否满足重置条件"""
        cond = RESET_CONDITIONS.get(self.le_flag)
        if cond is None:
            return True

        lon_, lat_ = self.coor[0, [0, 2]], self.coor[0, [1, 3]]
        in_region = (
            (lon_ > self.re_le[0]) & (lon_ < self.re_le[1]) &
            (lat_ > self.re_le[2]) & (lat_ < self.re_le[3])
        )
        min_dist = cond.get("min_distance", 0)
        return in_region.all() and (self.distance >= min_dist)

    def _step_core(self, actions, shift_tensor):
        """共享的动作执行逻辑"""
        self.steps += 1

        neigs_index = self.edge[self.cur].squeeze()
        neigs_coor = self.nodes[neigs_index][:, :2]
        coor = self.coor[:, :2].repeat(len(neigs_index), 1)
        neigs_heading = self.cal.heading(torch.cat((coor, neigs_coor), dim=-1))

        diff = neigs_heading[neigs_heading != 0]
        diff_ = 180 - torch.abs(torch.abs(diff - shift_tensor[actions]) - 180)
        try:
            next_index = neigs_index[diff_.argmin()].unsqueeze(0)
        except Exception:
            next_index = neigs_index[0].unsqueeze(0)

        self.last = self.cur
        self.cur = next_index
        self.coor[:, :2] = self.nodes[next_index][:, :2]

        distance = self.cal.distance(self.coor)
        done = (distance <= self.goal_range) | (self.steps >= self.max_steps)

        reward = (self.distance - distance) * 0.02 - 0.02
        if done.any():
            reward += 1

        self.distance = distance
        return done


class StreetEnv(BaseStreetEnv):
    """纯坐标导航环境（无视觉）"""

    def __init__(self, args):
        self.args = args
        self.le = args.le
        self.le_flag = args.le_flag
        self.cache_root = args.cache_root
        self.batch_size = args.batch_size
        self.num_actions = args.num_actions
        self.goal_range = args.goal_range
        self.device = args.device
        self.max_steps = args.max_steps
        self.back_pic = args.back_pic
        self.land = Landmark(args)
        self.cal = Calculating()
        self.scaler = Standard()
        self.views = {}
        self._init_shared()
        self.shift = torch.tensor([22.5 * s for s in range(16)], device=self.device, dtype=torch.float32)

    def _get_obs(self):
        state = self.scaler.trans(self.coor.clone().squeeze())
        return state

    def _dist_manh(self, obs):
        return torch.abs(obs[:2] - obs[2:]).sum().unsqueeze(0)

    def reset(self):
        while True:
            self.res = torch.randint(0, self.length, (self.batch_size, 2), device=self.device)
            self.cur = self.res[:, 0]
            self.goal = self.res[:, 1]
            self.coor = torch.cat((self.nodes[self.cur][:, :-1], self.nodes[self.goal][:, :-1]), dim=1)
            self.distance = self.cal.distance(self.coor)
            if self._reset_conditions():
                break
        self.steps = 0
        return self._get_obs()

    def step(self, actions, iftest=False):
        """StreetEnv step：完整实现（不依赖基类 _step_core）"""
        self.steps += 1

        neigs_index = self.edge[self.cur].squeeze()
        neigs_coor = self.nodes[neigs_index][:, :2]
        coor = self.coor[:, :2].repeat(len(neigs_index), 1)
        neigs_heading = self.cal.heading(torch.cat((coor, neigs_coor), dim=-1))

        diff = neigs_heading[neigs_heading != 0]
        diff_ = 180 - torch.abs(torch.abs(diff - self.shift[actions]) - 180)
        try:
            next_index = neigs_index[diff_.argmin()].unsqueeze(0)
        except Exception:
            next_index = neigs_index[0].unsqueeze(0)

        self.cur = next_index
        self.coor[:, :2] = self.nodes[next_index][:, :2]

        distance = self.cal.distance(self.coor)
        done = (distance <= self.goal_range) | (self.steps >= self.max_steps)

        reward = (self.distance - distance) * 0.02 - 0.02
        if done.any():
            reward += 1
        self.distance = distance

        if not iftest:
            return self._get_obs(), reward.squeeze(), done, self.views
        else:
            return self._get_obs(), self.scaler.trans(self.coor.clone().squeeze()), done

    def draw(self, cur_coo, goal_coo):
        map_coor = self.infos[1:, 1:3].astype('float32')
        plt.figure(figsize=(8, 8))
        plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.1, alpha=0.5)
        plt.scatter(cur_coo[0], cur_coo[1], c='blue', label='Start')
        plt.scatter(goal_coo[0], goal_coo[1], c='red', label='Goal')
        imp = plt.imread(self.back_pic)
        plt.imshow(imp, extent=self.le)
        plt.legend()
        plt.show()


class StreetPicEnv(BaseStreetEnv):
    """视觉+Landmark 导航环境（完整版）"""

    def __init__(self, args):
        self.args = args
        self.le = args.le
        self.le_flag = args.le_flag
        self.visual = args.visual
        self.cache_root = args.cache_root
        self.batch_size = args.batch_size
        self.num_actions = args.num_actions
        self.goal_range = args.goal_range
        self.device = args.device
        self.max_steps = args.max_steps
        self.load_pics = args.load_pics
        self.back_pic = args.back_pic
        self.land = Landmark(args)
        self.cal = Calculating()
        self.scaler = Standard()
        self.views = 0
        self.last = 0
        self._init_shared()

        # StreetPicEnv 特有：4方向离散动作（上/下/左/右）
        self.shift = torch.tensor([90 * s for s in range(4)], device=self.device, dtype=torch.float32)

        # 视觉数据加载器
        if self.visual:
            if self.load_pics == "rt":
                self.panoset = Picdataset_Realtime(self.args)
            else:
                self.panoset = Picdataset_Cache(self.args)

    # ---- 坐标离散化方法 ----

    def labels_mark(self, coor):
        """将经纬度映射到 16×16 离散格子编号"""
        obs, goal = coor[:2], coor[2:]
        bins = 16
        den_lon, den_lat = 0.005, 0.005
        lower = self.le[[0, 2]].clone()
        labels = []
        for nodes in [obs, goal]:
            diff = nodes - lower
            lon = ((diff[0] / den_lon) * bins).floor()
            lat = ((diff[1] / den_lat) * bins).floor()
            tab = lat * bins + lon
            labels.append(tab)
        return torch.stack(labels)

    def grids(self, coor):
        """归一化网格坐标 [0, 1]"""
        bins = 16
        epsilon = 1e-6
        den = torch.abs(self.le[[1, 0, 3, 2]] - self.le).float()
        result = ((coor - self.le[[0, 2, 0, 2]]) * bins / (den + epsilon))
        return torch.clamp(result, 0, bins) / bins

    # ---- 核心 ----

    def _get_obs(self):
        state = self.scaler.trans(self.coor.clone().squeeze())
        view = self.visual_learning() if self.visual else 0

        emb_1 = self.land.embbeding(self.coor[:, :2]).squeeze()
        emb_2 = self.land.embbeding(self.coor[:, 2:]).squeeze()

        grids = self.grids(self.coor.clone().squeeze())
        labels = self.labels_mark(self.coor.clone().squeeze())

        sp = self.heading / 360
        sp_bin = ((self.heading / 360) * 16).long()

        # obs 布局: labels(2) + grids(4) + lonlat(4) + sp(1) + sp_bin(1) + emb_1(100) + emb_2(100) = 212维
        state = torch.cat((labels, grids, state, sp, sp_bin, emb_1, emb_2), dim=-1)
        return (state, view, self.cur)

    def visual_learning(self):
        cur_head = self.heading - self.nodes[self.cur, -1]
        views = self.panoset.get_(self.cur, cur_head)
        return views

    def get_views(self, ids):
        return self.panoset.get_(ids, 0)

    def reset(self):
        while True:
            self.res = torch.randint(0, self.length, (self.batch_size, 2), device=self.device)
            self.cur = self.res[:, 0]
            self.goal = self.res[:, 1]
            self.coor = torch.cat((self.nodes[self.cur][:, :-1], self.nodes[self.goal][:, :-1]), dim=1)
            self.distance = self.cal.distance(self.coor)
            self.heading = self.cal.heading(self.coor)
            if self._reset_conditions():
                break
        self.steps = 0
        return self._get_obs()

    def step(self, actions, iftest=False):
        reward_val = None  # 占位，由 _step_core 内部计算
        done = self._step_core(actions, self.shift)
        self.heading = self.cal.heading(self.coor)

        if not iftest:
            return self._get_obs(), reward.squeeze() if reward_val is not None else torch.tensor(0.0), done, self.views
        else:
            return self._get_obs(), self.scaler.trans(self.coor.clone().squeeze()), done

    def step(self, actions, iftest=False):
        """StreetPicEnv 的 step：复用基类 _step_core 并返回完整元组"""
        self.steps += 1

        neigs_index = self.edge[self.cur].squeeze()
        neigs_coor = self.nodes[neigs_index][:, :2]
        coor = self.coor[:, :2].repeat(len(neigs_index), 1)
        neigs_heading = self.cal.heading(torch.cat((coor, neigs_coor), dim=-1))

        diff = neigs_heading[neigs_heading != 0]
        diff_ = 180 - torch.abs(torch.abs(diff - self.shift[actions]) - 180)
        try:
            next_index = neigs_index[diff_.argmin()].unsqueeze(0)
        except Exception:
            next_index = neigs_index[0].unsqueeze(0)

        self.last = self.cur
        self.cur = next_index
        self.coor[:, :2] = self.nodes[next_index][:, :2]

        distance = self.cal.distance(self.coor)
        done = (distance <= self.goal_range) | (self.steps >= self.max_steps)

        reward = (self.distance - distance) * 0.02 - 0.02
        if done.any():
            reward += 1
        self.distance = distance
        self.heading = self.cal.heading(self.coor)

        if not iftest:
            return self._get_obs(), reward.squeeze(), done, self.views
        else:
            return self._get_obs(), self.scaler.trans(self.coor.clone().squeeze()), done


class GridEnv:
    """4×4 网格世界环境（调试/验证用）"""

    def __init__(self, args):
        self.size = args.grids_size
        self.max_steps = args.max_steps
        self.device = args.device
        self.generator = torch.Generator(device=self.device)
        self.move_vectors = torch.tensor(
            [[0, -1], [0, 1], [-1, 0], [1, 0]], device=self.device, dtype=torch.long
        )
        self.reset()

    def reset(self):
        while True:
            self.agent = torch.randint(0, self.size, (2,), generator=self.generator, device=self.device)
            self.goal = torch.randint(0, self.size, (2,), generator=self.generator, device=self.device)
            if not torch.equal(self.agent, self.goal):
                break
        self.steps = 0
        self.prev_dist = self._dist(self.agent, self.goal)
        return self._get_obs()

    def _dist(self, a, g):
        return torch.sum(torch.abs(a - g))

    def _get_obs(self):
        obs = torch.zeros(52, device=self.device, dtype=torch.float32)
        s = self.size - 1
        obs[6] = self.agent[0].float() / s
        obs[7] = self.agent[1].float() / s
        obs[8] = self.goal[0].float() / s
        obs[9] = self.goal[1].float() / s
        return obs

    def step(self, action, iftest=None):
        new_agent = self.agent + self.move_vectors[action]
        new_agent = torch.clamp(new_agent, 0, self.size - 1)
        self.agent = new_agent
        self.steps += 1

        dist = self._dist(self.agent, self.goal)
        shaping = 0.2 * (self.prev_dist - dist)
        self.prev_dist = dist

        reward = -0.01 + shaping
        done = torch.equal(self.agent, self.goal) or (self.steps >= self.max_steps)
        if torch.equal(self.agent, self.goal):
            reward = torch.tensor(1.0, device=self.device)

        if not iftest:
            return self._get_obs(), reward, done, {}
        else:
            return self._get_obs(), torch.cat((self.agent, self.goal)), done
