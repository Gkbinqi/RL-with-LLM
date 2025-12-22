# main.py
import sys
import os
import msvcrt
from config import Action
from engine import MagicRLTower
from render import ConsoleRenderer

# --- 跨平台按键检测模块 ---
class KeyboardListener:
    def __init__(self):
        self.is_windows = os.name == 'nt'

    def read_key(self) -> str:
        """读取一个字符，无需回车"""
        # msvcrt.getch() 返回的是 bytes (如 b'a')，需要解码
        # 这种方式会阻塞直到有一个键被按下
        try:
            char = msvcrt.getch().decode('utf-8').lower()
            return char
        except UnicodeDecodeError:
            return 'u' # 处理功能键可能导致的解码错误

def get_action_from_key(key: str) -> Action:
    key_map = {
        'w': Action.UP, 's': Action.DOWN, 'a': Action.LEFT, 'd': Action.RIGHT,
        'z': Action.BUY_HP, 'x': Action.BUY_ATK, 'c': Action.BUY_DEF,
        'b': Action.BOMB,
        'e': Action.END, 'q': Action.QUIT
    }
    return key_map.get(key, Action.UNKNOW)

def main():
    env = MagicRLTower()
    renderer = ConsoleRenderer()
    listener = KeyboardListener()
    
    obs, info = env.reset()
    renderer.render(obs, info.get('msg', "勇士, 沉默..."))

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = get_action_from_key(listener.read_key())

        if action == Action.QUIT:
            renderer.render(obs, "勇士, 逃跑...")
            break

        if action == Action.END:
            renderer.render(obs, f"""勇士, 结束... 结算: {obs['hero_stats']['exp'] * 0.1 +
                                obs['hero_stats']['gold'] * 0.1 +
                                obs['floor'] * 10 + 
                                obs['hero_stats']['keys'] * 3}""")
            break
        
        obs, reward, terminated, truncated, info = env.step(action) 
        renderer.render(obs, info.get('msg', "勇士, 沉默..."))

if __name__ == "__main__":
    main()