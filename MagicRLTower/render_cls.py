# render_cls.py
import os
from config import GameConfig, MapElem

class ConsoleRenderer:
    def render(self, obs, info="勇士, 勇士..."):
        # Windows/Linux 兼容清屏
        os.system('cls' if os.name == 'nt' else 'clear')
        
        grid = obs['grid']
        stats = obs['hero_stats']
        hero_r, hero_c = obs['hero_pos']
        floor = obs.get('floor', 0)
        
        print(f"(===* Magic RL Tower -- FLOOR {floor} *===)")
        print(f"Sta: HP: {stats['hp']:>4} | ATK: {stats['atk']:>3} | DEF: {stats['def']:>3}")
        print(f"Res: EXP: {stats['exp']:>4} | GOLD: {stats['gold']:>3} | KEYS: {stats['keys']:>3}")
        print("-" * 25)

        # 地图渲染
        rows, cols = grid.shape
        display_grid = [[' ' for _ in range(cols)] for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                display_grid[r][c] = GameConfig.ASCII_MAP.get(grid[r][c], '?')
        # 勇士位置强制显示
        display_grid[hero_r][hero_c] = GameConfig.ASCII_MAP[MapElem.HERO]
        
        for row in display_grid:
            print(" ".join(row))

        print("-" * 25)
        print(f"> {info}")
        print("-" * 25)
        print("[WASD]移动 [ZXC]商店(血/攻/防) [Q]退出")