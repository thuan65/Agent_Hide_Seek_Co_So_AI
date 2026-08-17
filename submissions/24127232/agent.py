import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque
import heapq # For A* priority queue
import random
import itertools

class PacmanAgent(BasePacmanAgent):
    """
    Optimus Pacman - Optimized for Partial Observability
    Uses 3-State FSM, Staleness Memory, and Heading-Aware A* Intercept.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Optimus Pacman (Staleness FSM)"
        
        # Base template memory
        self.last_known_enemy_pos = None
        
        # --- Advanced Optimizations Memory ---
        self.use_spawn_bias = True  
        self.graph_initialized = False
        self.traversable = set()
        self.junctions = []
        self.leaf_nodes = []
        
        self.staleness = None
        self.steps_since_seen = 999
        self.last_heading = None
        
        # --- Goal Hysteresis (Target Lock) ---
        self.current_target = None
        self.current_target_score = 0

    def _init_static_map(self, map_state: np.ndarray):
        h, w = map_state.shape
        self.staleness = np.zeros((h, w), dtype=float)
        
        # 1. Identify all valid path cells
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:  
                    self.traversable.add((r, c))
        
        # 2. Extract strategic map topology
        for (r, c) in self.traversable:
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in self.traversable:
                    neighbors += 1
            
            if neighbors > 2:
                self.junctions.append((r, c))
            elif neighbors == 1:
                self.leaf_nodes.append((r, c))
                
        if not self.junctions:
            self.junctions = list(self.traversable)
            
        self.graph_initialized = True

    def _update_memory(self, map_state: np.ndarray, enemy_position):
        self.staleness += 1
        
        for r, c in self.traversable:
            if map_state[r, c] == 0:
                self.staleness[r, c] = 0
                
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

    def _get_sweep_target(self, my_pos: tuple):
        if self.current_target:
            if my_pos == self.current_target or self.staleness[self.current_target] == 0:
                self.current_target = None
                self.current_target_score = 0
                
        candidates = self.junctions + self.leaf_nodes
        scored_candidates = []
        
        for node in candidates:
            dist = abs(node[0] - my_pos[0]) + abs(node[1] - my_pos[1])
            
            local_staleness = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for i in range(1, 6):
                    nr, nc = node[0] + dr*i, node[1] + dc*i
                    if (nr, nc) in self.traversable:
                        local_staleness += self.staleness[nr, nc]
                    else:
                        break 
                        
            bias = 1.5 if (self.use_spawn_bias and node[0] < 10) else 1.0
            score = (local_staleness * bias) / (dist + 1)
            scored_candidates.append((score, node))
            
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        best_score = scored_candidates[0][0]
        
        if self.current_target and self.current_target_score > 0:
            if best_score <= 1.8 * self.current_target_score:
                return self.current_target

        top_3 = scored_candidates[:3]
        scores = np.array([s for s, _ in top_3])
        
        tau = 5.0
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / tau)
        probs = exp_scores / np.sum(exp_scores)
        
        chosen_idx = np.random.choice(len(top_3), p=probs)
        self.current_target = top_3[chosen_idx][1]
        self.current_target_score = top_3[chosen_idx][0]
        
        return self.current_target

    def _heading_aware_astar(self, start: tuple, goal: tuple, is_chase: bool):
        if start == goal: 
            return []
            
        open_set = []
        tiebreaker = itertools.count() # <-- FIX: guarantees unique queue priorities
        
        # (f_score, g_score, tiebreaker, r, c, heading, path)
        heapq.heappush(open_set, (0, 0, next(tiebreaker), start[0], start[1], self.last_heading, []))
        visited = {}
        
        while open_set:
            f, g, _, r, c, h, path = heapq.heappop(open_set)
            
            if is_chase and (abs(r - goal[0]) + abs(c - goal[1]) <= 1):
                return path + [(r, c)]
            if (r, c) == goal:
                return path + [(r, c)]
                
            state_key = (r, c, h)
            if state_key in visited and visited[state_key] <= g:
                continue
            visited[state_key] = g
            
            for dr, dc, move_enum in [(-1, 0, Move.UP), (1, 0, Move.DOWN), 
                                      (0, -1, Move.LEFT), (0, 1, Move.RIGHT)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in self.traversable:
                    continue
                    
                if self.pacman_speed >= 2 and (h == move_enum or h is None):
                    n2r, n2c = r + 2*dr, c + 2*dc
                    if (n2r, n2c) in self.traversable:
                        g2 = g + 1 
                        h2 = abs(n2r - goal[0]) + abs(n2c - goal[1])
                        heapq.heappush(open_set, (g2 + h2, g2, next(tiebreaker), n2r, n2c, move_enum, path + [(r, c), (nr, nc)]))
                        
                g1 = g + 1
                h1 = abs(nr - goal[0]) + abs(nc - goal[1])
                heapq.heappush(open_set, (g1 + h1, g1, next(tiebreaker), nr, nc, move_enum, path + [(r, c)]))
                
        return []

    def _vec_to_move(self, dr, dc) -> Move:
        if dr < 0: return Move.UP
        if dr > 0: return Move.DOWN
        if dc < 0: return Move.LEFT
        if dc > 0: return Move.RIGHT
        return Move.STAY

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        if not self.graph_initialized:
            self._init_static_map(map_state)
            
        self._update_memory(map_state, enemy_position)
        
        target = None
        is_chase = False
        
        if self.steps_since_seen == 0:
            target = enemy_position
            is_chase = True
            self.current_target = None 
            
        elif self.steps_since_seen < 3 and self.last_known_enemy_pos:
            target = self.last_known_enemy_pos
            is_chase = True
            
        else:
            target = self._get_sweep_target(my_position)

        path = self._heading_aware_astar(my_position, target, is_chase)
        
        if not path or len(path) < 2:
            self.last_heading = Move.STAY
            return (Move.STAY, 1)
            
        curr = path[0]
        nxt1 = path[1]
        move1 = self._vec_to_move(nxt1[0] - curr[0], nxt1[1] - curr[1])
        steps = 1
        
        if self.pacman_speed >= 2 and len(path) >= 3:
            nxt2 = path[2]
            move2 = self._vec_to_move(nxt2[0] - nxt1[0], nxt2[1] - nxt1[1])
            if move1 == move2 and self.last_heading in (move1, None, Move.STAY):
                steps = 2
                
        self.last_heading = move1
        return (move1, steps)


class GhostAgent(BaseGhostAgent):
    """Ghost (Hider) implementation with organic hiding spot inference."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = {}            
        self.graph_computed = False
        self.pacman_obs_radius = self._detect_pacman_obs_radius()
        self.move_queue = deque()  # Stores Move enums to reach target

    def step(self, map_state, my_position, enemy_position, step_number):
        if not self.graph_computed:
            self.graph = self._build_reachable_graph(map_state, my_position)
            pacman_spawn = enemy_position if enemy_position is not None else (15, 10)
            
            # Filter 1 & 2: Pruning and Scoring
            self._prune_candidates(pacman_spawn)
            self._evaluate_safety_scores(map_state, pacman_spawn)
            
            # Filter 3: Target Selection & Path Calculation
            target_node = self._select_best_target(my_position)
            if target_node:
                self.move_queue = self._bfs_path_moves(my_position, target_node)
                
            self.graph_computed = True

        # Pop pre-computed moves until reaching hiding spot
        move = Move.STAY
        if self.move_queue:
            move = self.move_queue.popleft()

        return move

    def _select_best_target(self, my_position):
        """
        Filter 3: Selects candidate with highest safety score, breaking ties via 
        shortest BFS path, and choosing randomly among candidates with equal path length.
        """
        candidates = [node for node in self.graph.values() if node["is_candidate"]]
        if not candidates:
            return None

        max_score = max(c["safety_score"] for c in candidates)
        top_candidates = [c["pos"] for c in candidates if c["safety_score"] == max_score]

        if len(top_candidates) == 1:
            return top_candidates[0]

        # Tie-breaker 1: Calculate shortest BFS path length for each top candidate
        min_path_len = float('inf')
        shortest_path_candidates = []

        for pos in top_candidates:
            path = self._bfs_path_moves(my_position, pos)
            path_len = len(path)

            if path_len < min_path_len:
                min_path_len = path_len
                shortest_path_candidates = [pos]
            elif path_len == min_path_len:
                shortest_path_candidates.append(pos)

        # Tie-breaker 2: Random selection among candidates tied for shortest path length
        return random.choice(shortest_path_candidates)

    def _bfs_path_moves(self, start, target):
        """Calculates BFS path from start to target and returns a deque of Move enums."""
        if start == target:
            return deque()

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return deque(path)

            for neighbor in self.graph[curr]["neighbors"]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Determine directional Move enum based on coordinate delta
                    dr = neighbor[0] - curr[0]
                    dc = neighbor[1] - curr[1]
                    if dr == -1: move = Move.UP
                    elif dr == 1: move = Move.DOWN
                    elif dc == -1: move = Move.LEFT
                    elif dc == 1: move = Move.RIGHT
                    
                    queue.append((neighbor, path + [move]))

        return deque()

    def _detect_pacman_obs_radius(self) -> int:
        if '--pacman-obs-radius' in sys.argv:
            try:
                idx = sys.argv.index('--pacman-obs-radius')
                return int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                pass
        try:
            for frame_info in inspect.stack():
                local_self = frame_info.frame.f_locals.get("self")
                if local_self and local_self.__class__.__name__ == "Arena":
                    return getattr(local_self, "pacman_obs_radius", 5)
        except Exception:
            pass
        return 5

    def _build_reachable_graph(self, map_state, start_pos):
        rows, cols = map_state.shape
        graph = {}
        queue = deque([start_pos])
        visited = {start_pos}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr = queue.popleft()
            graph[curr] = {
                "pos": curr,
                "neighbors": [],
                "is_candidate": True,
                "safety_score": 0.0
            }
            for dr, dc in directions:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and map_state[nr, nc] != 1:
                    neighbor = (nr, nc)
                    graph[curr]["neighbors"].append(neighbor)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return graph

    def _prune_candidates(self, enemy_position):
        enemy_r, enemy_c = enemy_position
        for pos, node in self.graph.items():
            r, c = node["pos"]
            manhattan_dist = abs(r - enemy_r) + abs(c - enemy_c)
            degree = len(node["neighbors"])
            if manhattan_dist <= 5 or degree >= 3:
                node["is_candidate"] = False
                node["safety_score"] = -999.0

    def _evaluate_safety_scores(self, map_state, enemy_position):
        rows, cols = map_state.shape
        ghost_start_row = 9
        max_north_distance = 9.0
        max_runway_length = 12.0
        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for pos, node in self.graph.items():
            # Keep disqualified nodes heavily penalized on heatmap
            if not node["is_candidate"]:
                node["safety_score"] = -999.0
                continue

            r, c = node["pos"]

            # 1. North Bias [-1.0, 1.0]
            norm_north = (ghost_start_row - r) / max_north_distance
            norm_north = max(-1.0, min(1.0, norm_north))

            # 2. Infiltration & Direct Runway Risk
            max_infiltrate_risk = 0

            for dr, dc in cardinal_dirs:
                # Measure direct sightline in direction (dr, dc)
                direct_length = 0
                for step in range(1, self.pacman_obs_radius + 1):
                    ur, uc = r + dr * step, c + dc * step
                    if 0 <= ur < rows and 0 <= uc < cols and map_state[ur, uc] != 1:
                        direct_length += 1
                    else:
                        break
                
                # Direct exposure counts toward risk
                max_infiltrate_risk = max(max_infiltrate_risk, direct_length)

                # Check perpendicular bleed from intermediate tiles u
                for step in range(1, direct_length + 1):
                    ur, uc = r + dr * step, c + dc * step
                    
                    # If moving vertically (dr != 0), perpendicular is horizontal (dc == 0)
                    perp_dirs = [(0, -1), (0, 1)] if dr != 0 else [(-1, 0), (1, 0)]

                    cross_ray_length = 0
                    for p_dr, p_dc in perp_dirs:
                        for p_step in range(1, int(max_runway_length) + 1):
                            pr, pc = ur + p_dr * p_step, uc + p_dc * p_step
                            if 0 <= pr < rows and 0 <= pc < cols and map_state[pr, pc] != 1:
                                cross_ray_length += 1
                            else:
                                break

                    max_infiltrate_risk = max(max_infiltrate_risk, cross_ray_length)

            # Normalize Infiltration Risk [0.0, 1.0]
            norm_risk = min(1.0, max_infiltrate_risk / max_runway_length)

            # Safety Score: Risk heavily penalizes open runways
            node["safety_score"] = (1 * norm_north) - (1 * norm_risk)