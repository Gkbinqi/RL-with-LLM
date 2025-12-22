# map_gen.py
import numpy as np
import random
from collections import deque
from config import MapElem

class Rect:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        
    def center(self):
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return (center_y, center_x)

class Leaf:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.min_size = 8  # 稍微增大一点，给CA留出空间
        self.room = None
        self.left = None
        self.right = None

    def split(self):
        if self.w < self.min_size * 2 or self.h < self.min_size * 2:
            return False
        
        split_h = random.random() > 0.5
        if self.w > self.h and self.w / self.h >= 1.25: split_h = False
        elif self.h > self.w and self.h / self.w >= 1.25: split_h = True

        if split_h:
            max_size = self.h - self.min_size
            if max_size < self.min_size: return False
            split_loc = random.randint(self.min_size, max_size)
            self.left = Leaf(self.x, self.y, self.w, split_loc)
            self.right = Leaf(self.x, self.y + split_loc, self.w, self.h - split_loc)
        else:
            max_size = self.w - self.min_size
            if max_size < self.min_size: return False
            split_loc = random.randint(self.min_size, max_size)
            self.left = Leaf(self.x, self.y, split_loc, self.h)
            self.right = Leaf(self.x + split_loc, self.y, self.w - split_loc, self.h)
            
        return True

    def create_rooms(self, grid):
        if self.left or self.right:
            if self.left: self.left.create_rooms(grid)
            if self.right: self.right.create_rooms(grid)
            if self.left and self.right:
                self.create_hall(self.left.get_room(), self.right.get_room(), grid)
        else:
            # --- 核心修改：生成房间时引入元胞自动机 ---
            room_w = random.randint(5, self.w - 2)
            room_h = random.randint(5, self.h - 2)
            room_x = self.x + random.randint(1, self.w - room_w - 1)
            room_y = self.y + random.randint(1, self.h - room_h - 1)
            
            self.room = Rect(room_x, room_y, room_w, room_h)
            
            # 1. 先挖空一个矩形
            for r in range(self.room.y1, self.room.y2):
                for c in range(self.room.x1, self.room.x2):
                    grid[r][c] = MapElem.EMPTY
            
            # 2. 在矩形内部应用元胞自动机 (Cellular Automata)
            # 仅当房间足够大时才生成地形，否则太挤了
            if room_w > 6 and room_h > 6:
                self._apply_ca_to_room(grid, self.room)

    def _apply_ca_to_room(self, grid, room):
        """
        在房间内部生成有机形状的障碍物
        """
        # 提取局部区域（注意保留1圈作为边界，不参与演化，保证连通性）
        # 我们只在内部区域 [y1+1 : y2-1] 生成噪音
        sub_h = room.y2 - room.y1 - 2
        sub_w = room.x2 - room.x1 - 2
        
        if sub_h <= 0 or sub_w <= 0: return

        # 初始化随机噪音 (概率越高，墙越多)
        noise_prob = 0.45 
        local_map = (np.random.rand(sub_h, sub_w) < noise_prob).astype(int)
        
        # CA 演化规则: 4-5 Rule
        # 如果周围墙数量 > 4 -> 变墙
        # 如果周围墙数量 < 4 -> 变空
        steps = 2
        for _ in range(steps):
            new_map = local_map.copy()
            for r in range(sub_h):
                for c in range(sub_w):
                    # 统计 3x3 邻居里的墙数量
                    walls = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            # 边界外视为墙
                            if nr < 0 or nr >= sub_h or nc < 0 or nc >= sub_w:
                                walls += 1
                            elif local_map[nr][nc] == 1:
                                walls += 1
                    if walls > 4:
                        new_map[r][c] = 1
                    elif walls < 4:
                        new_map[r][c] = 0
            local_map = new_map

        # 将演化结果回写到 grid
        for r in range(sub_h):
            for c in range(sub_w):
                if local_map[r][c] == 1:
                    grid[room.y1 + 1 + r][room.x1 + 1 + c] = MapElem.WALL

    def get_room(self):
        if self.room: return self.room
        if self.left and self.right:
            return self.left.get_room() if random.random() < 0.5 else self.right.get_room()
        if self.left: return self.left.get_room()
        if self.right: return self.right.get_room()
        return None

    def create_hall(self, room1, room2, grid):
        r1, c1 = room1.center()
        r2, c2 = room2.center()
        # 简单的直角连廊
        if random.random() < 0.5:
            self._h_tunnel(grid, c1, c2, r1)
            self._v_tunnel(grid, r1, r2, c2)
        else:
            self._v_tunnel(grid, r1, r2, c1)
            self._h_tunnel(grid, c1, c2, r2)

    def _h_tunnel(self, grid, x1, x2, y):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            grid[y][x] = MapElem.EMPTY
            # 走廊加宽一点点，减少死路概率
            if y+1 < grid.shape[0]-1 and grid[y+1][x] == MapElem.WALL and random.random() < 0.1:
                grid[y+1][x] = MapElem.EMPTY

    def _v_tunnel(self, grid, y1, y2, x):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            grid[y][x] = MapElem.EMPTY
            if x+1 < grid.shape[1]-1 and grid[y][x+1] == MapElem.WALL and random.random() < 0.1:
                grid[y][x+1] = MapElem.EMPTY

def prune_disconnected_areas(grid):
    """
    使用泛洪算法 (Flood Fill) 找出最大的连通空地区域，
    并将所有无法到达的微小空洞填死变成墙。
    解决：CA 经常会在房间角落生成封闭的小气泡，导致道具刷在里面拿不到。
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    
    # 1. 找到所有空地
    empty_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != MapElem.WALL:
                empty_cells.append((r, c))
    
    if not empty_cells: return [] # 全是墙?
    
    # 2. 寻找最大的连通分量
    max_region = []
    
    for r, c in empty_cells:
        if visited[r][c]: continue
        
        # 开始 BFS
        current_region = []
        queue = deque([(r, c)])
        visited[r][c] = True
        current_region.append((r, c))
        
        while queue:
            cur_r, cur_c = queue.popleft()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = cur_r + dr, cur_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr][nc] and grid[nr][nc] != MapElem.WALL:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        current_region.append((nr, nc))
        
        if len(current_region) > len(max_region):
            max_region = current_region
            
    # 3. 填死其它所有小区域
    max_region_set = set(max_region)
    for r, c in empty_cells:
        if (r, c) not in max_region_set:
            grid[r][c] = MapElem.WALL
            
    return max_region # 返回有效的空地列表

def generate_random_map(map_size):
    """
    由 _load_floor 调用, 生成一个随机地图和勇士初始位置
    """    
    h, w = map_size, map_size
    grid = np.full((h, w), MapElem.WALL, dtype=np.int32)
    
    # 1. BSP 分割
    root = Leaf(0, 0, w, h)
    leafs = [root]
    did_split = True
    while did_split:
        did_split = False
        for l in list(leafs):
            if l.left is None and l.right is None:
                if l.split():
                    leafs.append(l.left)
                    leafs.append(l.right)
                    did_split = True
    
    # 2. 生成房间 (带 CA 地形) 和走廊
    root.create_rooms(grid)
    
    # 3. 关键步骤：剪除死角，保证连通性
    # valid_cells 只包含主连通区域的空地
    valid_cells = prune_disconnected_areas(grid)
    
    # 如果地图生成失败（太拥挤导致没有空间），重试
    if len(valid_cells) < 10:
        return generate_random_map(map_size)

    # 4. 生成门
    # 逻辑同旧版，但基于清理后的 grid
    door_candidates = []
    for r, c in valid_cells:
         # 简单的门检测逻辑：左右墙上下空，或上下墙左右空
        is_h_door = (grid[r][c-1] == MapElem.WALL and grid[r][c+1] == MapElem.WALL and 
                     grid[r-1][c] == MapElem.EMPTY and grid[r+1][c] == MapElem.EMPTY)
        is_v_door = (grid[r-1][c] == MapElem.WALL and grid[r+1][c] == MapElem.WALL and
                     grid[r][c-1] == MapElem.EMPTY and grid[r][c+1] == MapElem.EMPTY)
        
        if (is_h_door or is_v_door) and random.random() < 0.15: # 降低门生成的概率
            door_candidates.append((r, c))

    random.shuffle(door_candidates)
    # 限制门的总数，每20格空地才允许一个门
    max_doors = len(valid_cells) // 20
    for i in range(min(len(door_candidates), max_doors)):
        dr, dc = door_candidates[i]
        grid[dr][dc] = MapElem.DOOR

    # 5. 放置实体 (需要重新 scan valid_cells 因为有些变成门了)
    final_empty_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == MapElem.EMPTY:
                final_empty_cells.append((r, c))
    random.shuffle(final_empty_cells)

    # 勇士
    hero_pos = list(final_empty_cells.pop())
    
    # 楼梯 (最远距离)
    best_stairs = final_empty_cells[0]
    max_dist = -1
    for cell in final_empty_cells[:min(len(final_empty_cells), 50)]:
        dist = abs(cell[0] - hero_pos[0]) + abs(cell[1] - hero_pos[1])
        if dist > max_dist:
            max_dist = dist
            best_stairs = cell
    
    final_empty_cells.remove(best_stairs)
    grid[best_stairs[0]][best_stairs[1]] = MapElem.STAIRS
    
    # 确保勇士脚下是空的
    grid[hero_pos[0]][hero_pos[1]] = MapElem.EMPTY

    # 物品和怪物
    area = len(final_empty_cells)
    # 动态调整生成率：地图越乱，生成的怪物密度稍微低一点，防止堵死
    num_items = int(area * 0.05) + 1
    num_monsters = int(area * 0.06) + 1

    for _ in range(num_items):
        if not final_empty_cells: break
        r, c = final_empty_cells.pop()
        item = MapElem.KEY if random.random() < 0.3 else MapElem.POTION
        grid[r][c] = item
        
    for _ in range(num_monsters):
        if not final_empty_cells: break
        r, c = final_empty_cells.pop()
        mob = random.choice([
            MapElem.MONSTER_SLIME, MapElem.MONSTER_BAT,
            MapElem.MONSTER_WOLF, MapElem.MONSTER_ORC,
            MapElem.MONSTER_GUARD
        ])
        grid[r][c] = mob

    return grid, hero_pos