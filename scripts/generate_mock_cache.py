"""
为 GridEnv 生成 mock 缓存文件，无需真实 LevelDB 数据。
在服务器上运行此脚本，生成 cache_min/ 目录下的必要文件。
"""
import numpy as np
import os

# GridEnv 用 4x4 网格，模拟 16 个节点
NUM_NODES = 16
cache_dir = "./cache_min"
os.makedirs(cache_dir, exist_ok=True)

# 1. static_info.npy: [id, lng, lat, ...]
# GridEnv 不需要真实经纬度，用网格坐标代替
infos_title = np.array(['id', 'coords.lng', 'coords.lat', 'snapped_coords.lng', 'snapped_coords.lat',
                        'alt', 'roll_deg', 'pitch_deg', 'heading_deg', 'pano_date', 'country_code'])
static_info = np.empty((NUM_NODES + 1, 11), dtype=object)
static_info[0, :] = infos_title

for i in range(NUM_NODES):
    row_id = f"node_{i}"
    x = i % 4  # 网格 x
    y = i // 4  # 网格 y
    # 用网格坐标模拟经纬度（范围：经度 -74.0 ~ -73.99，纬度 40.75 ~ 40.755）
    lng = -74.0 + x * 0.003
    lat = 40.750 + y * 0.00125
    static_info[i + 1, :] = [row_id, lng, lat, lng, lat, 0, 0, 0, 0, "", ""]

np.save(os.path.join(cache_dir, 'static_info.npy'), static_info)
print(f"✓ 生成 static_info.npy ({NUM_NODES} 个节点)")

# 2. neig_info.npy: 邻居关系（4x4 网格的 4 连通）
neig_title = np.array(['self', 'neig1', 'neig2', 'neig3', 'neig4', 'neig5', 'neig6'])
neig_info = np.empty((NUM_NODES + 1, 7), dtype=object)
neig_info[0, :] = neig_title

for i in range(NUM_NODES):
    neighbors = []
    # 上
    if i - 4 >= 0: neighbors.append(f"node_{i-4}")
    # 下
    if i + 4 < NUM_NODES: neighbors.append(f"node_{i+4}")
    # 左
    if i % 4 != 0: neighbors.append(f"node_{i-1}")
    # 右
    if i % 4 != 3: neighbors.append(f"node_{i+1}")
    row = [f"node_{i}"] + neighbors + [None] * (6 - len(neighbors))
    neig_info[i + 1, :] = row

np.save(os.path.join(cache_dir, 'neig_info.npy'), neig_info)
print(f"✓ 生成 neig_info.npy")

# 3. neigs_index.npy: 邻居的索引（用于快速查找）
neigs_id = neig_info[:, 1:]
padding = np.arange(0, len(neigs_id))
neigs_index = np.empty_like(neigs_id)
for i in range(neigs_index.shape[1]):
    neigs_index[:, i] = padding

for nid, nes in enumerate(neigs_id):
    nes = nes[nes != None]
    inds = np.where(np.isin(static_info[:, 0], nes))[0]
    length = len(inds)
    neigs_index[nid, :length] = inds

np.save(os.path.join(cache_dir, 'neigs_index.npy'), neigs_index.astype('int32'))
print(f"✓ 生成 neigs_index.npy")

# 4. area_index.npy: min 区域的节点 mask（全 True）
area_index = np.array([True] + [True] * NUM_NODES)  # 含标题行
np.save(os.path.join(cache_dir, 'area_index.npy'), area_index)
print(f"✓ 生成 area_index.npy")

# 5. landmark.npy: 随机生成 100 个 landmark（GridEnv 用不上，但需要文件存在）
np.random.seed(42)
lands = np.random.randn(100, 2).astype('float32')
lands[:, 0] = lands[:, 0] * 0.01 - 74.0   # 经度范围
lands[:, 1] = lands[:, 1] * 0.01 + 40.75   # 纬度范围
np.save(os.path.join(cache_dir, 'landmark.npy'), lands)
print(f"✓ 生成 landmark.npy")

print(f"\n所有缓存文件已生成在 {cache_dir}/")
print("现在可以运行：python main_.py --env_name GridEnv --agent_name AgentPosi --le_flag min --total_steps 50000 --device cuda")
