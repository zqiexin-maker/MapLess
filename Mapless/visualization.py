import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch
from core.utils import Standard, Get_circle


def _to_cpu(data):
    """统一将 torch.Tensor 转为 numpy（处理 CUDA 设备）"""
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()
        return data.numpy()
    return data


class Draw_result:
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.env_name = args.env_name
        self.back_extent = args.le
        self.back_pic = args.back_pic
        self.size = getattr(args, 'grids_size', 20)
        self.cache_root = args.cache_root
        self.scaler = Standard()

    def draw_trajs(self, paths):
        dispatch = {
            "GridEnv": self.draw_trajs_grid,
            "StreetEnv": self.draw_trajs_street_multi,
            "StreetPicEnv": self.draw_trajs_street_multi,
        }
        handler = dispatch.get(self.env_name)
        if handler:
            handler(paths)

    # ---- GridEnv 绘图 ----

    def draw_trajs_grid(self, paths):
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for e, axe in enumerate(axs.flatten()):
            path = _to_cpu(paths[e])
            axe.scatter(path[0, 0] + 0.5, path[0, 1] + 0.5, c='blue', label='Start')
            axe.scatter(path[0, -2] + 0.5, path[0, -1] + 0.5, c='red', label='Goal')
            axe.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, '-o', markersize=2, label=f'Test_{e}')
            axe.set_xlim(0, self.size)
            axe.set_ylim(0, self.size)
            axe.legend()
        fig.tight_layout()
        fig.savefig(fname=f'{self.save_dir}/test.jpg', dpi=100, format='jpg', bbox_inches='tight')
        plt.close()

    # ---- 街景绘图 ----

    def _load_map_info(self):
        """加载地图背景信息"""
        infos = np.load(f"{self.cache_root}/static_info.npy", allow_pickle=True)
        if self.args.le is not None:
            area_index = np.load(f"{self.cache_root}/area_index.npy")
            infos = infos[area_index]
        return infos

    def draw_back_pic(self, map_coor):
        pic_path = f'{self.save_dir}/back_pic.jpg'
        if not os.path.exists(pic_path):
            plt.figure(figsize=(5, 5))
            plt.scatter(map_coor[:, 0], map_coor[:, 1], c='grey', s=0.3, alpha=0.6)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(pic_path, dpi=300, pad_inches=0, bbox_inches='tight', transparent=True)
            plt.close()

    def draw_trajs_street_multi(self, paths):
        infos = self._load_map_info()
        map_coor = infos[1:, 1:3].astype('float32')
        self.draw_back_pic(map_coor)

        self.scaler.fit(map_coor)
        paths = [self.scaler.inver_trans(_to_cpu(p)) for p in paths]
        figsize_ = (14, 13) if self.args.le_flag == "min" else (16, 10)

        fig, axs = plt.subplots(3, 3, figsize=figsize_)
        for e, axe in enumerate(axs.flatten()):
            path = _to_cpu(paths[e])
            axe.plot(path[:, 0], path[:, 1], '-o', color="green", markersize=2, label=f'Test_{e}')
            axe.scatter(path[0, 0], path[0, 1], c='blue', label='Start', s=20)
            axe.scatter(path[0, 2], path[0, 3], c='red', label='Goal', s=20)
            if self.args.le_flag != "min":
                cir = Get_circle(path[0, 2:], radius=50)
                axe.plot(cir[:, 0], cir[:, 1], color="red", lw=1)

            imp = plt.imread(f"{self.save_dir}/back_pic.jpg")
            axe.imshow(imp, extent=self.back_extent)
            axe.legend()
        fig.tight_layout()
        fig.savefig(fname=f'{self.save_dir}/test.jpg', dpi=300, format='jpg', bbox_inches='tight')
        plt.close()

    # ---- 奖励曲线 ----

    def draw_rewads(self):
        episode_rewards = np.load(f"{self.save_dir}/ep_rewards.npy")
        plt.figure(figsize=(12, 8))
        plt.plot(episode_rewards, alpha=0.5, label="Episode Return")
        if len(episode_rewards) >= 50:
            smooth = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
            plt.plot(range(49, 49 + len(smooth)), smooth, c="#1f77b4", label="Moving Avg(50)")
        plt.legend()
        plt.savefig(fname=f"{self.save_dir}/rewards.jpg", dpi=300, format='jpg', bbox_inches='tight')
        plt.close()

    # ---- Mercator 投影范围 ----

    def extent_mercator(self, coor):
        coor = torch.tensor(coor)
        coor = self.scaler.mercator(coor)
        ex = [coor[:, 0].min(), coor[:, 0].max(), coor[:, 1].min(), coor[:, 1].max()]
        ex = [e.item() for e in ex]
        return ex
