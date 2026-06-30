"""
Mapless Navigation — 统一入口

用法:
    # 训练 + 测试（主实验）
    python train.py --env_name StreetPicEnv --agent_name AgentPosi --le_flag min

    # 仅测试（加载已有模型）
    python train.py --test_only --env_name StreetPicEnv --agent_name AgentPosi --le_flag min

    # 消融实验循环
    python train.py --run_ablation --env_name StreetPicEnv --le_flag min
"""
import os
from core.ppo_trainer import Exps
import argparse
import warnings
from cache import Cache_infons
from models.ablation_models import AGENTS_NAME
import shutil

warnings.filterwarnings('ignore')


def build_parser():
    parser = argparse.ArgumentParser(description='Mapless Navigation')

    # ---- 数据路径 ----
    parser.add_argument('--db_root_l', default='..', type=str)
    parser.add_argument('--db_root_h', default='./Streetview/manhattan_2021_highres', type=str)
    parser.add_argument('--back_pic', default='./white.png', type=str)
    parser.add_argument('--cache_root', default='./cache', type=str)
    parser.add_argument('--save_dir', default='./results_agentposi_streetpicenv', type=str)
    parser.add_argument('--le', default=[], type=list)
    parser.add_argument('--max_buf', default=100, type=int)

    # ---- 环境和Agent ----
    parser.add_argument('--agent_name', default="AgentPosi", type=str,
                        help="AgentPosi or one of: " + ", ".join(AGENTS_NAME))
    parser.add_argument('--env_name', default="StreetPicEnv", type=str,
                        help="StreetEnv, StreetPicEnv, GridEnv")
    parser.add_argument('--load_pics', default="cc", type=str, help="rt (实时) or cc (缓存)")
    parser.add_argument('--num_actions', default=4, type=int)

    # ---- PPO 超参数 ----
    parser.add_argument('--update_times', default=6, type=int)
    parser.add_argument("--grids_size", type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument("--log_interval", type=int, default=2000)
    parser.add_argument('--gamma', default=0.995, type=float)
    parser.add_argument('--clip_eps', default=0.2, type=float)
    parser.add_argument("--entropy_coef", type=float, default=0.004)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # ---- 设备和随机种子 ----
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda', type=str)

    # ---- 实验配置 ----
    parser.add_argument('--le_flag', default="min", type=str)  # min / mid / max
    parser.add_argument('--visual', default=True, type=bool)
    parser.add_argument('--pixel', default=84, type=int)  # 注意：原代码这里写成了 bool，修正为 int
    parser.add_argument('--landmark_num', default=100, type=int)
    parser.add_argument('--hidden_feats', default=128, type=int)
    parser.add_argument('--state_fests', default=52, type=int)
    parser.add_argument('--goal_range', default=5, type=int)

    # ---- 训练规模 ----
    parser.add_argument("--total_steps", default=5e5, type=int)
    parser.add_argument('--rollout_len', default=1500, type=int)
    parser.add_argument('--max_steps', default=150, type=int)
    parser.add_argument("--test_episodes", type=int, default=300)

    # ---- 运行模式 ----
    parser.add_argument('--test_only', action='store_true', help='仅测试，不训练')
    parser.add_argument('--run_ablation', action='store_true', help='运行全部消融实验')

    return parser


LEARNING_EXTENTS = {
    "max": [-74.035, -73.935, 40.695, 40.795],
    "mid": [-73.990, -73.976, 40.756, 40.765],
    "min": [-73.990, -73.985, 40.750, 40.755],
}


def setup_paths(args):
    """根据 le_flag 设置缓存/保存路径"""
    args.cache_root = f"{args.cache_root}_{args.le_flag}"
    args.save_dir = f"{args.save_dir}/{args.env_name}_{args.agent_name}"
    if args.env_name != "GridEnv":
        args.save_dir = f"{args.save_dir}/{args.le_flag}"
    return args


def run_single(args):
    """运行单次实验：训练 → 测试"""
    args = setup_paths(args)
    args.le = LEARNING_EXTENTS[args.le_flag]

    Cache_infons(args)
    print('Cache infos finished')

    exp = Exps(args)
    if not args.test_only:
        exp.train()
    exp.test()
    print(f"Done! Results saved to {args.save_dir}")


def run_ablation(args):
    """运行消融实验：遍历所有 Agent 变体"""
    for i, name in enumerate(AGENTS_NAME):
        args.agent_name = name
        if i > 1:
            args.visual = True  # pics_* 类需要视觉

        args = setup_paths(args)
        args.le = LEARNING_EXTENTS[args.le_flag]

        Cache_infons(args)
        print(f'[{i+1}/{len(AGENTS_NAME)}] Running {name}...')
        print('Cache infos finished')

        exp = Exps(args)
        exp.train()

        # 复制结果到 pic 子目录
        for src_name in ["ep_rewards.npy", "test.jpg", "agent.pth"]:
            src = f"{args.save_dir}/{src_name}"
            dst = f"{args.save_dir}/pic/{src_name.replace('.npy', f'_{i}.npy').replace('.jpg', f'_{i}.jpg').replace('.pth', f'_{i}.pth')}"
            if os.path.exists(src):
                os.makedirs(f"{args.save_dir}/pic", exist_ok=True)
                shutil.copy2(src, dst)

        # 测试
        exp.test()
        print(f"[{i+1}/{len(AGENTS_NAME)}] {name} finished.\n")


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.run_ablation:
        run_ablation(args)
    else:
        run_single(args)
