# config.py
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict

class Action(IntEnum):
    UNKNOW = -1
    # 移动指令
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # 商店指令 (全局可用，简化版)
    BUY_HP = 4
    BUY_ATK = 5
    BUY_DEF = 6
    # 道具指令
    BOMB = 7
    # 特殊指令
    END = 8
    QUIT = 9

class MapElem(IntEnum):
    EMPTY = 0
    WALL = 1
    DOOR = 2
    STAIRS = 3

    HERO = 9

    KEY = 10
    POTION = 11
    
    MONSTER_SLIME = 20
    MONSTER_BAT = 21
    MONSTER_WOLF = 22
    MONSTER_ORC = 23
    MONSTER_GUARD = 24
    MONSTER_KING = 25

@dataclass
class ItemProps:
    name: str
    type: str  # 'hp', 'key'
    value: int

@dataclass
class MonsterProps:
    name: str
    hp: int
    atk: int
    defense: int
    exp: int
    gold: int
    growth_scale: float = 1.0

class GameConfig:
    # --- 显示配置 ---
    ASCII_MAP = {
        MapElem.HERO: '@',
        MapElem.EMPTY: ' ', MapElem.WALL: '■', MapElem.DOOR: '#', MapElem.STAIRS: '※',

        MapElem.KEY: '🗝', MapElem.POTION: '♥',

        MapElem.MONSTER_SLIME: 's', 
        MapElem.MONSTER_BAT: 'b',
        MapElem.MONSTER_WOLF: 'w', 
        MapElem.MONSTER_ORC: 'o', 
        MapElem.MONSTER_GUARD: 'G',
        MapElem.MONSTER_KING: 'K'
    }

    # --- 商店配置 ---
    SHOP_COST = 20
    SHOP_GAIN = {
        Action.BUY_HP: 200,
        Action.BUY_ATK: 2,
        Action.BUY_DEF: 2
    }

    # --- 技能配置 ---
    BOMB_COST = 50

    # --- 物品数据 ---
    ITEMS: Dict[int, ItemProps] = {
        MapElem.KEY: ItemProps("钥匙", "key", 1),
        MapElem.POTION: ItemProps("血瓶", "hp", 200),
    }

    # --- 难度动态配置 ---
    # 每多一层，怪物属性提升的百分比
    FLOOR_SCALING_HP = 0.15   # 血量每层 +15%
    FLOOR_SCALING_ATK = 0.08  # 攻击每层 +8% 
    FLOOR_SCALING_DEF = 0.05  # 防御每层 +5%
    FLOOR_SCALING_REWARD = 0.2 # 金币经验每层 +20% (略高于怪物强度，给 Agent 容错空间)

    # --- 怪物数据 ---
    MONSTERS: Dict[int, MonsterProps] = {
        MapElem.MONSTER_SLIME: MonsterProps("史莱姆", 50, 15, 5, 1, 5, growth_scale=1.4),
        MapElem.MONSTER_BAT:   MonsterProps("蝙蝠", 100, 30, 0, 2, 8, growth_scale=0.9),
        MapElem.MONSTER_WOLF:  MonsterProps("狼", 250, 25, 10, 4, 12, growth_scale=1.0),
        MapElem.MONSTER_ORC:   MonsterProps("兽人", 500, 30, 10, 8, 15, growth_scale=1.1),
        MapElem.MONSTER_GUARD: MonsterProps("守卫", 800, 50, 25, 16, 30, growth_scale=1.0),
        MapElem.MONSTER_KING:  MonsterProps("魔王", 1200, 100, 30, 32, 100, growth_scale=1.3),
    }

    # --- 初始状态 ---
    INIT_HERO_HP = 2500
    INIT_HERO_ATK = 25
    INIT_HERO_DEF = 10
    INIT_HERO_KEYS = 3
    INIT_HERO_GOLD = 50
    INIT_HERO_EXP = 50

    # --- 安全屋 (11x11) ---
    SAFE_ROOM = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 1], 
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 11, 0, 1, 0, 1, 0, 11, 0, 1], 
        [1, 0, 11, 0, 2, 0, 2, 0, 11, 0, 1],
        [1, 0, 11, 0, 1, 0, 1, 0, 11, 0, 1], 
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], 
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 10, 0, 0, 0, 0, 0, 0, 0, 10, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.int32)

    SAFE_ROOM_HERO_POS = [9, 5]