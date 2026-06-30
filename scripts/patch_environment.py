"""
补丁：让 StreetEnv 和 GridEnv 输出 52 维 observation，与 AgentPosi 等网络兼容。
在服务器上运行一次即可，会直接修改 environment.py 的备份。
"""

# 要修改的文件
filepath = "/home/wcl/zq/MapLess/RL_Mapless-main/Mapless/environment.py"

with open(filepath, "r") as f:
    content = f.read()

# 方案：在 StreetEnv 和 GridEnv 的 _get_obs 方法里，
# 把 4 维输出 padding 到 52 维（多余维度填 0）
# 这样 agent.py 里取 obs[:, 11] 就不会越界

# 更简单方案：直接改 exp.py 里 PPOAgent.act() 和 store()，
# 在传入网络之前把 obs padding 到 52 维

# ===== 最干净的方案：修改 environment.py，给 StreetEnv 添加 full_obs =====
# 在 StreetEnv 的 _get_obs 后拼接padding

old_get_obs = '''    def _get_obs(self):
        state = self.scaler.trans(self.coor.clone().squeeze())
        #diff = self.coor - self.bl
        #state = torch.round(diff /self.emb).squeeze()
        return state'''

new_get_obs = '''    def _get_obs(self):
        state = self.scaler.trans(self.coor.clone().squeeze())
        # 拼接成 52 维（与 StreetPicEnv 对齐，多余维度填 0）
        # 维度 6-9: 标准化经纬度（4维，已有）
        # 其余维度填 0，保证总维度 = 52
        full_state = torch.zeros(52, device=self.device)
        full_state[6:10] = state  # lonlat_lonlat 取 obs[:, 6:10]
        return full_state'''

if old_get_obs in content:
    content = content.replace(old_get_obs, new_get_obs)
    print("✓ 已修补 StreetEnv._get_obs")
else:
    print("✗ 未找到 StreetEnv._get_obs，请手动检查")

# 同样修补 GridEnv（如果存在）
# 先找到 GridEnv 的 _get_obs
import re

# 找 GridEnv 类中的 _get_obs
gridenv_pattern = r'class GridEnv.*?(?=\nclass |\Z)'
gridenv_match = re.search(gridenv_pattern, content, re.DOTALL)
if gridenv_match:
    gridenv_code = gridenv_match.group()
    if '_get_obs' in gridenv_code:
        # GridEnv 的 _get_obs 返回 [agent_x, agent_y, goal_x, goal_y]
        # 需要 padding 到 52 维
        old_grid_obs = '''    def _get_obs(self):
        return torch.tensor([self.agent_x, self.agent_y, self.goal_x, self.goal_y], dtype=torch.float32, device=self.device)'''
        new_grid_obs = '''    def _get_obs(self):
        obs = torch.zeros(52, dtype=torch.float32, device=self.device)
        obs[6] = self.agent_x
        obs[7] = self.agent_y
        obs[8] = self.goal_x
        obs[9] = self.goal_y
        return obs'''
        if old_grid_obs in content:
            content = content.replace(old_grid_obs, new_grid_obs)
            print("✓ 已修补 GridEnv._get_obs")
        else:
            print("? GridEnv._get_obs 格式不匹配，请手动检查")

# 写回文件（先备份）
backup_path = filepath + ".bak"
with open(backup_path, "w") as f:
    with open(filepath, "r") as orig:
        f.write(orig.read())
print(f"✓ 原文件已备份至 {backup_path}")

with open(filepath, "w") as f:
    f.write(content)
print(f"✓ 已写入 {filepath}")
print("\n现在可以重新运行：")
print("  python main_.py --env_name StreetEnv --agent_name lonlat_lonlat --le_flag min --total_steps 50000 --device cuda")
