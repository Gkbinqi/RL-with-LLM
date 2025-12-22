# render.py
import sys
import os
from config import GameConfig, MapElem

# ANSI 控制码常量
class ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    CURSOR_HOME = "\033[H"     # 光标回原点
    CLEAR_SCREEN = "\033[2J"   # 清屏
    CLEAR_LINE = "\033[K"      # 清除光标后到行尾的内容
    HIDE_CURSOR = "\033[?25l"  # 隐藏光标
    SHOW_CURSOR = "\033[?25h"  # 显示光标

class ConsoleRenderer:
    def __init__(self):
        # Windows CMD有时需要执行一条空指令来激活 ANSI 转义序列支持
        if os.name == 'nt':
            os.system('')
        
        # 初始化：清屏一次，隐藏光标以获得更丝滑的体验
        sys.stdout.write(ANSI.CLEAR_SCREEN + ANSI.HIDE_CURSOR)
        sys.stdout.flush()

    def __del__(self):
        # 程序退出时恢复光标，防止把用户终端搞坏
        sys.stdout.write(ANSI.SHOW_CURSOR + ANSI.RESET)
        sys.stdout.flush()

    def render(self, obs, info=""):
        grid = obs['grid']
        stats = obs['hero_stats']
        hero_r, hero_c = obs['hero_pos']
        floor = obs.get('floor', 0)

        # --- 构建帧缓冲区 (Frame Buffer) ---
        # 我们将所有输出拼接成一个长字符串，一次性打印，减少IO操作
        buffer = []
        
        # 1. 移动光标到左上角 (不清屏!)
        buffer.append(ANSI.CURSOR_HOME)
        
        # 2. 绘制 UI 头部
        # CLEAR_LINE 用于确保如果数值变短，后面不会残留旧数字
        header_lines = [
            f"{ANSI.YELLOW}(===* Magic RL Tower -- FLOOR {floor} *===){ANSI.RESET}{ANSI.CLEAR_LINE}",
            f"Sta: HP: {self._color_hp(stats['hp'])} | ATK: {stats['atk']:<4} | DEF: {stats['def']:<4}{ANSI.CLEAR_LINE}",
            f"Res: EXP: {stats['exp']:<5} | GOLD: {stats['gold']:<4} | KEYS: {stats['keys']:<4}{ANSI.CLEAR_LINE}",
            f"{'-' * 30}{ANSI.CLEAR_LINE}"
        ]
        buffer.extend(header_lines)

        # 3. 绘制地图
        rows, cols = grid.shape
        for r in range(rows):
            line_chars = []
            for c in range(cols):
                # 判断当前格是什么
                char = ' '
                color = ANSI.RESET
                
                if r == hero_r and c == hero_c:
                    char = GameConfig.ASCII_MAP[MapElem.HERO]
                    color = ANSI.CYAN + ANSI.BOLD # 勇士高亮
                else:
                    elem = grid[r][c]
                    char = GameConfig.ASCII_MAP.get(elem, ' ')
                    # 简单上色 (可以根据喜好扩展)
                    if elem == MapElem.WALL: color = ANSI.WHITE
                    elif elem == MapElem.DOOR: color = ANSI.YELLOW
                    elif elem == MapElem.STAIRS: color = ANSI.MAGENTA
                    elif elem in [MapElem.KEY, MapElem.POTION]: color = ANSI.GREEN
                    elif elem >= 20: color = ANSI.RED # 怪物全是红色
                
                line_chars.append(f"{color}{char}{ANSI.RESET}")
            
            # 拼接这一行，并加上清除行尾指令
            buffer.append(" ".join(line_chars) + ANSI.CLEAR_LINE)

        # 4. 绘制底部信息


        footer_lines = [
            f"{'-' * 30}{ANSI.CLEAR_LINE}",
            f"> {ANSI.BOLD}{info}{ANSI.RESET}{ANSI.CLEAR_LINE}",
            f"{'-' * 30}{ANSI.CLEAR_LINE}",
            # [Mod] 增加炸弹提示
            f"[WASD]移动 [ZXC]商店 [E]炸弹(消耗EXP) [Q]退出{ANSI.CLEAR_LINE}",
            f"{ANSI.CLEAR_LINE}"
        ]
        buffer.extend(footer_lines)

        # --- 一次性输出 ---
        # 使用 sys.stdout.write 比 print 更快，且不会自动加换行符干扰布局
        sys.stdout.write("\n".join(buffer))
        sys.stdout.flush()

    def _color_hp(self, hp):
        """根据血量给颜色"""
        if hp > 500: return f"{ANSI.GREEN}{hp:<5}{ANSI.RESET}"
        if hp > 200: return f"{ANSI.YELLOW}{hp:<5}{ANSI.RESET}"
        return f"{ANSI.RED}{hp:<5}{ANSI.RESET}"