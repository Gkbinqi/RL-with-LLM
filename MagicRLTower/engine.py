# engine.py
import numpy as np
import copy
from config import GameConfig, Action, MapElem
from mechanics import resolve_interaction
from map_gen import generate_random_map

# engine 负责维护游戏内部数据: [地图, 勇士位置, 勇士状态], 游戏状态判断逻辑
# 以及, 适用 RL 的 step/reset 接口维护设计
# 具体的实体交互判断和计算, 由 mechanics.py 负责
class MagicRLTower:
    def __init__(self, map_size=25):
        self.map_size = map_size
        self.floor = 0
        self.reset()

    def reset(self):
        """Standard Gym Reset: returns (obs, info)"""
        # 1. 维护内部状态
        
        # grid 存储静态物体，勇士位置单独存储
        self.floor = 0  
        self._load_floor()        
        self.hero_stats = {
            'hp': GameConfig.INIT_HERO_HP,
            'atk': GameConfig.INIT_HERO_ATK,
            'def': GameConfig.INIT_HERO_DEF,
            'keys': GameConfig.INIT_HERO_KEYS, 
            'gold': GameConfig.INIT_HERO_GOLD, 
            'exp': GameConfig.INIT_HERO_EXP
        }

        # 2. 维护接口内容
        self.message = "勇士, 开始"
        self.done = False
        return self._get_obs(), {"msg": self.message}

    def step(self, action: Action):
        """
        独占写权限, 维护游戏变量, 处理高级别逻辑
        Standard Gym Step: returns (obs, reward, terminated, truncated, info)
        """
        
        if self.done:
            return self._get_obs(), 0, True, False, {"msg": "勇士, 结束"}

        # 1. 调用 Mechanics 库判断交互
        result = resolve_interaction(action, self.hero_pos, self.hero_stats, self.grid, self.floor)
        self.message = result.message

        # 2. 状态应用 (Engine 独占写权限)
        
        # 2.1 伤害结算
        if result.damage_taken > 0:
            self.hero_stats['hp'] -= result.damage_taken

        # 2.2 资源变更
        for k, v in result.resource_changes.items():
            if k in self.hero_stats:
                self.hero_stats[k] += v

        # 2.3 地图变更
        for r, c, new_elem in result.grid_changes:
            self.grid[r][c] = new_elem

        # 2.4 结算生死
        if self.hero_stats['hp'] <= 0:
            self.done = True

        # 2.5 处理楼层切换/移动
        if result.is_next_floor and not self.done:
            self.floor += 1
            self._load_floor()
            result.message += f" [ 第 {self.floor} 层 ]"
            # 切换楼层后，不需要更新旧地图的 hero_pos
        elif result.can_move and not self.done:
            # 只有当确实移动了，才更新坐标
            # 注意：Mechanics 里如果杀怪移动了，grid_changes 里应该已经把那个格子设为空了
            self.hero_pos = result.next_pos

        # 3. 计算 Reward (为 RL 准备)
        # 这里可以设计更复杂的 reward function，目前暂时只看经验值和楼层
        step_reward = result.resource_changes.get('exp', 0) + \
                        result.is_next_floor * (10 + self.floor) + \
                        (-100 if self.done and self.hero_stats['hp'] <= 0 else 0)

        return self._get_obs(), step_reward, self.done, False, {"msg": self.message}
    
    def _get_obs(self):
        # 需要优化
        # 1. 改为POMDP
        # 2. 为方便CNN, 需要进行 数据归一化 以及 分层(W,H,C)
        return {
            'grid': self.grid.copy(),
            'floor': self.floor,
            'hero_stats': self.hero_stats.copy(),
            'hero_pos': self.hero_pos.copy(),
        }
    
    def _load_floor(self):
        """根据层数加载地图"""
        # 规则: 0层(初始), 或每5层(5, 10...) 为安全屋
        if self.floor % 5 == 0:
            self.grid = copy.deepcopy(GameConfig.SAFE_ROOM)
            self.hero_pos = list(GameConfig.SAFE_ROOM_HERO_POS)
        else:
            self.grid, self.hero_pos = generate_random_map(self.map_size)