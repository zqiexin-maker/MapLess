"""
测试模式入口（等同于 python train.py --test_only）

用法:
    python evaluate.py --env_name StreetPicEnv --agent_name AgentPosi --le_flag min
"""
import sys
# 将 --test_only 注入参数
sys.argv.append('--test_only')
from train import build_parser, run_single, LEARNING_EXTENTS
from cache import Cache_infons
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = build_parser()
    # 测试模式的默认值覆盖
    parser.set_defaults(device='cpu', visual=False)
    args = parser.parse_args()

    from core.ppo_trainer import Exps
    import os

    args.cache_root = f"{args.cache_root}_{args.le_flag}"
    args.save_dir = f"{args.save_dir}/{args.env_name}_{args.agent_name}"
    if args.env_name != "GridEnv":
        args.save_dir = f"{args.save_dir}/{args.le_flag}"
    args.le = LEARNING_EXTENTS[args.le_flag]

    Cache_infons(args)
    print('Cache infos finished')
    exp = Exps(args)
    exp.test()
    print(f"Done! Results saved to {args.save_dir}")
