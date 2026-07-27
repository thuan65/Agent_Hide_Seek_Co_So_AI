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
    Version 2 Ghost Agent designed for Pacman vs Ghost Hide-and-Seek.
    
    Exploits static map properties and upper-half spawn bias in random spawn mode:
    1. Evaluates distance to both hiding pockets on step 1 and selects the nearer one.
    2. Adapts BFS move preferences based on spawn row to maximize vertical distance
       from Pacman (prioritizing UP when spawned below Row 5, and delaying DOWN when spawned above).
    3. Stays permanently at the chosen pocket once reached.
    """

    TARGET_ROW = 5
    LEFT_POCKET = (5, 8)
    RIGHT_POCKET = (5, 12)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Proximity Pocket Exploiter Ghost v2"

        # Navigation state precomputed on step 1
        self.target_pocket = None
        self.planned_moves = None
        self.current_move_idx = 0

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple, 
             step_number: int) -> Move:
        """
        Executes one environment tick for the Ghost Agent.
        
        Ignores enemy_position. Performs single-pass dual-path calculation on step 1,
        follows the optimized path to the nearest pocket, and stays forever.
        """
        # Step 1: Initialize target pocket choice and precompute path
        if self.planned_moves is None:
            self.planned_moves = self._plan_route_to_best_pocket(my_position, map_state)

        # Reached hiding pocket or finished path -> stay forever
        if self.current_move_idx >= len(self.planned_moves):
            return Move.STAY

        # Follow planned path step-by-step
        next_move = self.planned_moves[self.current_move_idx]
        self.current_move_idx += 1
        return next_move

    def _plan_route_to_best_pocket(self, start_pos: tuple, map_state: np.ndarray) -> list:
        """
        Calculates shortest paths to both left and right pockets, chooses the nearer pocket,
        and breaks distance ties randomly.
        """
        path_left = self._bfs_shortest_path(start_pos, self.LEFT_POCKET, map_state)
        path_right = self._bfs_shortest_path(start_pos, self.RIGHT_POCKET, map_state)

        dist_left = len(path_left)
        dist_right = len(path_right)

        # Select the strictly nearer pocket path
        if dist_left < dist_right:
            self.target_pocket = self.LEFT_POCKET
            return path_left
        elif dist_right < dist_left:
            self.target_pocket = self.RIGHT_POCKET
            return path_right
        else:
            # Equidistant from both pockets: pick randomly to remain unpredictable
            if random.random() < 0.5:
                self.target_pocket = self.LEFT_POCKET
                return path_left
            else:
                self.target_pocket = self.RIGHT_POCKET
                return path_right

    def _bfs_shortest_path(self, start_pos: tuple, target_pos: tuple, map_state: np.ndarray) -> list:
        """
        Runs BFS to find the shortest path from start_pos to target_pos.
        
        Dynamically adjusts neighbor expansion order based on spawn row:
        - Spawned below Row 5: Prioritizes UP early to increase vertical distance from Pacman.
        - Spawned at or above Row 5: Prioritizes horizontal moves (LEFT/RIGHT) and UP before DOWN
          to delay downward moves into Pacman's threat zone.
        """
        if start_pos == target_pos:
            return []

        spawn_row = start_pos[0]

        # Determine directional search order based on spawn elevation relative to Row 5
        if spawn_row > self.TARGET_ROW:
            # Below target: Move UP early to climb away from lower half
            directions = [
                ((-1, 0), Move.UP),
                ((0, -1), Move.LEFT),
                ((0, 1), Move.RIGHT),
                ((1, 0), Move.DOWN)
            ]
        else:
            # At or above target: Move sideways first, delay DOWN as late as possible
            directions = [
                ((0, -1), Move.LEFT),
                ((0, 1), Move.RIGHT),
                ((-1, 0), Move.UP),
                ((1, 0), Move.DOWN)
            ]

        queue = deque([start_pos])
        visited = {start_pos}
        parent = {}  # Maps child_pos -> (parent_pos, Move)

        found = False
        while queue:
            curr_pos = queue.popleft()

            if curr_pos == target_pos:
                found = True
                break

            for (dr, dc), move in directions:
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)

                if self._is_valid_cell(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (curr_pos, move)
                    queue.append(next_pos)

        if not found:
            return []

        # Reconstruct sequence of moves from target back to start
        path_moves = []
        curr = target_pos
        while curr != start_pos:
            prev_pos, move = parent[curr]
            path_moves.append(move)
            curr = prev_pos

        path_moves.reverse()
        return path_moves

    def _is_valid_cell(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Checks if a position is within grid boundaries and non-wall."""
        r, c = pos
        rows, cols = map_state.shape
        if 0 <= r < rows and 0 <= c < cols:
            return map_state[r, c] != 1
        return False