# mechanics.py
import math
import copy
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
from config import GameConfig, MapElem, MonsterProps, Action

# 定义交互结果的标准格式
@dataclass
class InteractionResult:
    # 基础结果
    can_move: bool
    message: str
    next_pos: List[int]

    # Optional 结果
    damage_taken: int = 0
    # 资源变化 (正数加，负数减)
    resource_changes: Dict[str, int] = field(default_factory=dict)
    # 地图变化列表: [(r, c, new_elem), ...]
    grid_changes: List[Tuple[int, int, int]] = field(default_factory=list)
    is_next_floor: bool = False

def get_dynamic_monster(elem_id: int, floor: int) -> MonsterProps:
    """
    根据层数计算怪物的动态属性
    公式: Stat = Base * (1 + Floor * Global_Rate * Monster_Scale)
    """
    base = GameConfig.MONSTERS[elem_id]

    # 动态计算
    # 使用 floor * growth_scale * CONFIG_RATE
    scale_hp = 1 + floor * GameConfig.FLOOR_SCALING_HP * base.growth_scale
    scale_atk = 1 + floor * GameConfig.FLOOR_SCALING_ATK * base.growth_scale
    scale_def = 1 + floor * GameConfig.FLOOR_SCALING_DEF * base.growth_scale
    scale_reward = 1 + floor * GameConfig.FLOOR_SCALING_REWARD * base.growth_scale

    return MonsterProps(
        name=f"Lv.{floor}[{base.name}]",
        hp=int(base.hp * scale_hp),
        atk=int(base.atk * scale_atk),
        defense=int(base.defense * scale_def),
        exp=int(base.exp * scale_reward),
        gold=int(base.gold * scale_reward),
        growth_scale=base.growth_scale
    )

def calculate_damage(hero_atk: int, hero_def: int, monster: MonsterProps) -> int:
    if hero_atk <= monster.defense:
        return 999999
    hero_dpt = hero_atk - monster.defense
    monster_dpt = max(0, monster.atk - hero_def)
    turns = math.ceil(monster.hp / hero_dpt)
    monster_hits = max(0, turns - 1)
    return monster_hits * monster_dpt

def use_bomb(hero_pos: List[int], hero_stats: Dict[str, int], grid) -> InteractionResult:
    cost = GameConfig.BOMB_COST
    if hero_stats['exp'] < cost:
        return InteractionResult(False, "勇士, 经验不足...", hero_pos)
    
    h, w = grid.shape
    r, c = hero_pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 记录要改变的格子，而不是直接改 grid
    changes = []
    killed_count = 0
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            elem = grid[nr][nc]
            if elem in GameConfig.MONSTERS:
                killed_count += 1
                changes.append((nr, nc, MapElem.EMPTY)) # 标记为变成空地

    msg = f"勇士, 炸掉 {killed_count} 个敌人" if killed_count > 0 else "勇士, 空大..."
    
    return InteractionResult(
        can_move=False,
        message=msg,
        next_pos=hero_pos,
        resource_changes={'exp': -cost}, # 统一用 resource_changes 处理增减
        grid_changes=changes
    )

def use_shop(action: Action, hero_pos: List[int], hero_stats: Dict[str, int]) -> InteractionResult:
    cost = GameConfig.SHOP_COST
    if hero_stats['gold'] < cost:
        return InteractionResult(False, "勇士, 没钱", hero_pos)
    
    resource_changes = {'gold': -cost}
    attr_name = ""
    gain = GameConfig.SHOP_GAIN[action]
    
    if action == Action.BUY_HP:
        resource_changes['hp'] = gain
        attr_name = "生命"
    elif action == Action.BUY_ATK:
        resource_changes['atk'] = gain
        attr_name = "攻击"
    elif action == Action.BUY_DEF:
        resource_changes['def'] = gain
        attr_name = "防御"
        
    return InteractionResult(
        can_move=False, # 买东西不移动位置
        message=f"勇士, 加 {attr_name} {gain} 点, cost {cost} 金币",
        next_pos=hero_pos,
        resource_changes=resource_changes
    )

def resolve_interaction(action: Action
                        , hero_pos: List[int], hero_stats: Dict[str, int]
                        , grid
                        , floor: int) -> InteractionResult:
    """
    处理勇士与环境的交互逻辑
    """
    # 0. 无效指令
    if action == Action.UNKNOW:
         return InteractionResult(False, "勇士, 无效指令...", hero_pos)
    
    # 1. 处理特殊技能
    if action == Action.BOMB:
        return use_bomb(hero_pos, hero_stats, grid)
    
    # 2. 处理商店购买
    if action in [Action.BUY_HP, Action.BUY_ATK, Action.BUY_DEF]:
        return use_shop(action, hero_pos, hero_stats)
    
    # 3. 处理移动/环境交互
    directions = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
    if action not in directions:
        return InteractionResult(False, "勇士, 勇士...", hero_pos)

    dy, dx = directions[action]
    target_pos = [hero_pos[0] + dy, hero_pos[1] + dx]
    
    # 检查是否越界
    h, w = grid.shape
    if not (0 <= target_pos[0] < h and 0 <= target_pos[1] < w):
        return InteractionResult(False, "勇士, 越界...", hero_pos)

    # 获取交互元素
    target_elem = grid[target_pos[0]][target_pos[1]]

    # A. 墙/空地/楼梯/门 (逻辑保持不变)
    if target_elem == MapElem.WALL:
        return InteractionResult(False, "勇士, 撞墙...", hero_pos)
    if target_elem == MapElem.EMPTY:
        return InteractionResult(True, "勇士, 移动...", target_pos)
    if target_elem == MapElem.STAIRS:
        return InteractionResult(True, "勇士祈祷中...", target_pos,
                                is_next_floor=True)
    if target_elem == MapElem.DOOR:
        if hero_stats['keys'] > 0:
            return InteractionResult(True, "勇士, 开门...", target_pos, 
                                    resource_changes={'keys': -1},
                                    grid_changes=[(target_pos[0], target_pos[1], MapElem.EMPTY)])
        else:
            return InteractionResult(False, "勇士, 钥匙...", hero_pos)

    # B. 物品
    if target_elem in GameConfig.ITEMS:
        item = GameConfig.ITEMS[target_elem]
        resource_changes = {}
        if item.type == 'hp': 
            resource_changes['hp'] = int(item.value * (1 + floor * GameConfig.FLOOR_SCALING_HP))
        if item.type == 'key': resource_changes['keys'] = item.value
        return InteractionResult(True, f"勇士, {item.name}...", target_pos,
            resource_changes=resource_changes,
            grid_changes=[(target_pos[0], target_pos[1], MapElem.EMPTY)]
        )

    # C. 战斗
    if target_elem in GameConfig.MONSTERS:
        monster = get_dynamic_monster(target_elem, floor)
        damage = calculate_damage(hero_stats['atk'], hero_stats['def'], monster)
        
        if hero_stats['hp'] <= damage:
            return InteractionResult(False, f"勇士, 坠机... 机长: {monster.name}", hero_pos,
                damage_taken=damage
            )

        # 胜利：怪物消失(变成空地)，获得金币经验，扣血
        return InteractionResult(True, f"勇士, 打败 {monster.name}，受伤 {damage}, 获得 {monster.gold} 金币 {monster.exp} 经验", target_pos,
            damage_taken=damage,
            resource_changes={'gold': monster.gold, 'exp': monster.exp},
            grid_changes=[(target_pos[0], target_pos[1], MapElem.EMPTY)]
        )

    return InteractionResult(False, "未知的虚空", hero_pos)