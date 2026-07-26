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
    """
    Optimus Ghost - Optimized for Partial Observability Survival
    Uses Target-Locked Top Migration (Spawn Bias), Ray Evasion, and Tortuosity Pathfinding.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Optimus Ghost (Locked Top Migration)"
        
        # --- Spawn Bias Optimization Flag ---
        # If True: Locks the closest target in the highest row and takes the absolute shortest path UP
        self.use_spawn_bias = True  
        
        # State Tracking & Persistent Caches
        self.initialized = False
        self.last_known_pacman_pos = None
        self.spawn_bias_target = None
        self.top_row = 0
        
        self.height = 0
        self.width = 0
        self.traversable = set()
        self.neighbor_cache = {}
        self.dead_ends = set()
        self.cycle_nodes = set()

    def _init_caches(self, map_state: np.ndarray, start_pos: tuple):
        """Precomputes geometry, cycles, dead-ends, and locks top target on step 1."""
        self.height, self.width = map_state.shape
        
        # 1. Identify traversable path cells
        self.traversable = {
            (r, c) for r in range(self.height) for c in range(self.width) 
            if map_state[r, c] != 1 # Not a wall
        }
        
        # 2. Build adjacency graph with Move.UP prioritized first
        self.neighbor_cache = {}
        for r, c in self.traversable:
            neighbors = []
            # Ordering UP first guarantees BFS prefers upward steps during tie-breaks
            for move in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.traversable:
                    neighbors.append(((nr, nc), move))
            self.neighbor_cache[(r, c)] = neighbors

        # 3. Identify dead-ends (corridors with <= 1 escape route)
        degrees = {cell: len(neighs) for cell, neighs in self.neighbor_cache.items()}
        queue = [cell for cell, deg in degrees.items() if deg <= 1]
        
        while queue:
            curr = queue.pop(0)
            self.dead_ends.add(curr)
            for neighbor, _ in self.neighbor_cache.get(curr, []):
                if neighbor not in self.dead_ends:
                    valid_neighs = [n for n, _ in self.neighbor_cache[neighbor] if n not in self.dead_ends]
                    if len(valid_neighs) <= 1:
                        queue.append(neighbor)
                        
        self.cycle_nodes = self.traversable - self.dead_ends
        if not self.cycle_nodes:
            self.cycle_nodes = set(self.traversable)

        # 4. Lock persistent top target for Spawn Bias
        if self.use_spawn_bias:
            self.spawn_bias_target = self._select_locked_top_target(start_pos)

        self.initialized = True

    def _select_locked_top_target(self, start_pos: tuple) -> tuple:
        """Finds the absolute highest reachable row and selects the cell with the shortest path distance."""
        # 1. Determine the highest reachable row from start_pos
        queue = deque([start_pos])
        visited = {start_pos}
        min_row = start_pos[0]
        
        while queue:
            curr = queue.popleft()
            if curr[0] < min_row:
                min_row = curr[0]
            for nxt_pos, _ in self.neighbor_cache.get(curr, []):
                if nxt_pos not in visited:
                    visited.add(nxt_pos)
                    queue.append(nxt_pos)
                    
        self.top_row = min_row

        # 2. First node reached at min_row during BFS is guaranteed to have the shortest path
        queue = deque([(start_pos, 0)])
        visited = {start_pos}
        while queue:
            curr, dist = queue.popleft()
            if curr[0] == self.top_row:
                return curr # Nearest path target locked!
            for nxt_pos, _ in self.neighbor_cache.get(curr, []):
                if nxt_pos not in visited:
                    visited.add(nxt_pos)
                    queue.append((nxt_pos, dist + 1))

        return start_pos

    def _is_los_blocked(self, pos1: tuple, pos2: tuple, map_state: np.ndarray) -> bool:
        """Returns True if wall or corner blocks straight line of sight."""
        r1, c1 = pos1
        r2, c2 = pos2
        
        if r1 != r2 and c1 != c2:
            return True
            
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1, c] == 1:
                    return True
        else:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r, c1] == 1:
                    return True
                    
        return False

    def _calculate_tortuosity(self, pos: tuple) -> float:
        """Measures corridor windingness to strip Speed-2 Pacman down to Speed 1."""
        turns = 0
        neighbors = self.neighbor_cache.get(pos, [])
        if len(neighbors) == 2:
            m1 = neighbors[0][1]
            m2 = neighbors[1][1]
            if m1 != m2:
                turns += 1
        return turns * 15.0

    def _eval_evasion_move(self, move_pos: tuple, pacman_pos: tuple, map_state: np.ndarray) -> float:
        """Evaluates tactical evasion safety when Pacman is visible."""
        dist = abs(move_pos[0] - pacman_pos[0]) + abs(move_pos[1] - pacman_pos[1])
        
        if dist <= 1:
            return -9999.0
            
        score = dist * 10.0
        
        if self._is_los_blocked(move_pos, pacman_pos, map_state):
            score += 60.0
            
        if move_pos in self.dead_ends:
            score -= 150.0
            
        score += self._calculate_tortuosity(move_pos)
        
        if move_pos in self.cycle_nodes:
            score += 25.0
            
        return score

    def _detect_nearby_vision_ray(self, my_pos: tuple, map_state: np.ndarray):
        """Detects if Pacman's vision ray is actively sweeping adjacent fog."""
        r, c = my_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for i in range(1, 3):
                nr, nc = r + dr*i, c + dc*i
                if (nr, nc) in self.traversable:
                    if map_state[nr, nc] == 0:
                        return (nr, nc)
                else:
                    break
        return None

    def _bfs_next_step(self, start: tuple, target: tuple, allow_dead_ends: bool = False) -> Move:
        """Deterministic BFS pathfinder. Set allow_dead_ends=True for absolute shortest paths."""
        if start == target or target not in self.traversable:
            return Move.STAY
            
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path[0] if path else Move.STAY
                
            for nxt_pos, move in self.neighbor_cache.get(curr, []):
                if not allow_dead_ends and nxt_pos in self.dead_ends and nxt_pos != target:
                    continue
                if nxt_pos not in visited:
                    visited.add(nxt_pos)
                    queue.append((nxt_pos, path + [move]))
                    
        # Fallback if dead-end filter blocked path
        if not allow_dead_ends:
            return self._bfs_next_step(start, target, allow_dead_ends=True)
                    
        return Move.STAY

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        if not self.initialized:
            self._init_caches(map_state, my_position)
            
        # ------------------------------------------------------------------
        # STATE 1: PACMAN VISIBLE (Corner Breaking & Tortuosity Evasion)
        # ------------------------------------------------------------------
        if enemy_position is not None:
            self.last_known_pacman_pos = enemy_position
            
            best_score = -float('inf')
            best_move = Move.STAY
            
            candidates = self.neighbor_cache.get(my_position, []) + [(my_position, Move.STAY)]
            for nxt_pos, move in candidates:
                score = self._eval_evasion_move(nxt_pos, enemy_position, map_state)
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            return best_move

        # ------------------------------------------------------------------
        # SPAWN BIAS OVERRIDE: FORCE NEAREST PATH UP UNTIL TOPMOST ROW
        # ------------------------------------------------------------------
        if self.use_spawn_bias:
            # Check if we have arrived at the topmost traversable row
            if my_position[0] <= self.top_row:
                self.use_spawn_bias = False # Upward migration complete!
            else:
                # Take the absolute shortest path to the top target
                move = self._bfs_next_step(my_position, self.spawn_bias_target, allow_dead_ends=True)
                if move != Move.STAY:
                    return move

        # ------------------------------------------------------------------
        # STATE 2: PREDICTIVE RAY EVASION (Vision Ray Detected Nearby)
        # ------------------------------------------------------------------
        ray_pos = self._detect_nearby_vision_ray(my_position, map_state)
        if ray_pos is not None:
            best_score = -float('inf')
            best_move = Move.STAY
            
            candidates = self.neighbor_cache.get(my_position, []) + [(my_position, Move.STAY)]
            for nxt_pos, move in candidates:
                dist = abs(nxt_pos[0] - ray_pos[0]) + abs(nxt_pos[1] - ray_pos[1])
                score = dist * 10.0
                
                if self._is_los_blocked(nxt_pos, ray_pos, map_state):
                    score += 40.0
                if nxt_pos in self.dead_ends:
                    score -= 100.0
                    
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            return best_move

        # ------------------------------------------------------------------
        # STATE 3: STEALTH & NAVIGATION (Pacman Unseen)
        # ------------------------------------------------------------------
        if my_position in self.cycle_nodes and map_state[my_position] == -1:
            return Move.STAY
            
        safe_nodes = list(self.cycle_nodes)
        if safe_nodes:
            target = min(safe_nodes, key=lambda n: abs(n[0] - my_position[0]) + abs(n[1] - my_position[1]))
            move = self._bfs_next_step(my_position, target)
            if move != Move.STAY:
                return move

        return Move.STAY