"""
Template for student agent implementation.
"""

import sys
from pathlib import Path
import heapq

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    Optimized hybrid: A* pathfinding + motion tracking/interception + Alpha-Beta Minimax
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "AlphaBeta Interceptor Pacman"
        
        # State Tracking
        self.last_known_enemy_pos = None
        self.prev_enemy_pos = None
        self.enemy_velocity = (0, 0)
        self.visited_cells = set()
        
        # Performance/Lookup caches
        self.valid_cells = set()
        self.neighbor_cache = {}
        self.astar_cache = {}
        self.initialized = False

    def _init_caches(self, map_state: np.ndarray):
        """Precompute traversable map structure once."""
        height, width = map_state.shape
        self.valid_cells = {
            (r, c) for r in range(height) for c in range(width) if map_state[r, c] == 0
        }
        self.neighbor_cache = {}
        for r, c in self.valid_cells:
            neighbors = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.valid_cells:
                    neighbors.append(((nr, nc), move))
            self.neighbor_cache[(r, c)] = neighbors
        self.initialized = True

    def _astar(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """Optimized A* shortest-path search."""
        if start == goal:
            return []
        
        cache_key = (start, goal)
        if cache_key in self.astar_cache:
            return self.astar_cache[cache_key]

        if start not in self.valid_cells or goal not in self.valid_cells:
            return []

        # Manhattan heuristic
        def heuristic(p):
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        frontier = []
        heapq.heappush(frontier, (heuristic(start), 0, start, []))
        visited = {start: 0}

        while frontier:
            _, cost, current, path = heapq.heappop(frontier)

            if current == goal:
                self.astar_cache[cache_key] = path
                return path

            if cost > visited.get(current, float('inf')):
                continue

            for neighbor, move in self.neighbor_cache.get(current, []):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    new_path = path + [move]
                    heapq.heappush(frontier, (new_cost + heuristic(neighbor), new_cost, neighbor, new_path))

        self.astar_cache[cache_key] = []
        return []

    def _get_intercept_target(self) -> tuple:
        """Motion extrapolation with clamping."""
        if not self.last_known_enemy_pos:
            return None
        
        if not self.prev_enemy_pos or self.enemy_velocity == (0, 0):
            return self.last_known_enemy_pos

        # Choose prediction horizon based on Manhattan distance
        dist = abs(self.last_known_enemy_pos[0] - self.prev_enemy_pos[0]) + abs(self.last_known_enemy_pos[1] - self.prev_enemy_pos[1])
        horizon = 3 if dist > 4 else 1

        vr, vc = self.enemy_velocity
        pred_r = self.last_known_enemy_pos[0] + vr * horizon
        pred_c = self.last_known_enemy_pos[1] + vc * horizon
        pred_pos = (pred_r, pred_c)

        if pred_pos in self.valid_cells:
            return pred_pos
        return self.last_known_enemy_pos

    def _evaluate_state(self, pac_pos: tuple, ghost_pos: tuple) -> float:
        """Evaluation function reflecting distance, capture status, and topological traps."""
        if pac_pos == ghost_pos:
            return 10000.0

        path = self._astar(pac_pos, ghost_pos, None)
        dist = len(path) if path else (abs(pac_pos[0] - ghost_pos[0]) + abs(pac_pos[1] - ghost_pos[1]))

        # Calculate ghost mobility and trap structural score
        ghost_moves = len(self.neighbor_cache.get(ghost_pos, []))
        trap_score = 4 - ghost_moves

        # dead turn penalty
        dead_turn_penalty = 1.0 if dist > 6 else 0.0

        score = (
            - dist * 8.0
            - ghost_moves * 5.0
            + trap_score * 20.0
            - dead_turn_penalty * 4.0
        )
        return score

    def _alphabeta(self, pac_pos: tuple, ghost_pos: tuple, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """Shallow depth alphabeta adversarial minimax."""
        if depth == 0 or pac_pos == ghost_pos:
            return self._evaluate_state(pac_pos, ghost_pos)

        if is_max:
            max_val = -float('inf')
            # Generate primary promising actions (1-step and speed-scaled moves)
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                for steps in range(1, self.pacman_speed + 1):
                    next_pos = pac_pos
                    valid = True
                    for _ in range(steps):
                        dr, dc = move.value
                        candidate = (next_pos[0] + dr, next_pos[1] + dc)
                        if candidate in self.valid_cells:
                            next_pos = candidate
                        else:
                            valid = False
                            break
                    if not valid and next_pos == pac_pos:
                        continue
                    
                    val = self._alphabeta(next_pos, ghost_pos, depth - 1, alpha, beta, False)
                    max_val = max(max_val, val)
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
            return max_val
        else:
            min_val = float('inf')
            # Ghost action (1 step)
            for neighbor, _ in self.neighbor_cache.get(ghost_pos, []):
                val = self._alphabeta(pac_pos, neighbor, depth - 1, alpha, beta, True)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """Main step strategy wrapper."""
        if not self.initialized:
            self._init_caches(map_state)
            
        self.visited_cells.add(my_position)
        
        # Track enemy movement and velocity
        if enemy_position is not None:
            if self.last_known_enemy_pos is not None:
                self.prev_enemy_pos = self.last_known_enemy_pos
                self.enemy_velocity = (
                    enemy_position[0] - self.prev_enemy_pos[0],
                    enemy_position[1] - self.prev_enemy_pos[1]
                )
            self.last_known_enemy_pos = enemy_position
        else:
            self.enemy_velocity = (0, 0)

        target = self._get_intercept_target()

        # Fallback exploration
        if target is None:
            best_move = Move.STAY
            best_visited_score = float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (my_position[0] + dr, my_position[1] + dc)
                if next_pos in self.valid_cells:
                    score = 0
                    # Evaluate based on visited cell density (exploration)
                    for r_offset in range(-2, 3):
                        for c_offset in range(-2, 3):
                            check_pos = (next_pos[0] + r_offset, next_pos[1] + c_offset)
                            if check_pos in self.visited_cells:
                                score += 1
                    if score < best_visited_score:
                        best_visited_score = score
                        best_move = move
            return (best_move, 1)

        # Minimax Action selection
        best_action = (Move.STAY, 1)
        best_score = -float('inf')

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            for steps in range(1, self.pacman_speed + 1):
                next_pos = my_position
                valid = True
                for _ in range(steps):
                    dr, dc = move.value
                    candidate = (next_pos[0] + dr, next_pos[1] + dc)
                    if candidate in self.valid_cells:
                        next_pos = candidate
                    else:
                        valid = False
                        break
                if not valid and next_pos == my_position:
                    continue

                # Run shallow Alpha-Beta from simulated state
                score = self._alphabeta(next_pos, target, depth=3, alpha=-float('inf'), beta=float('inf'), is_max=False)
                if score > best_score:
                    best_score = score
                    best_action = (move, steps)

        return best_action


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    Optimized: Minimax + Alpha-Beta Pruning + Territory-Control Evaluation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.name = "Territory Minimax Ghost"
        
        # State Tracking & Navigation Cache
        self.last_known_enemy_pos = None
        self.valid_cells = set()
        self.neighbor_cache = {}
        self.dead_ends = set()
        self.initialized = False

    def _init_caches(self, map_state: np.ndarray):
        """Precompute navigation graph, adjacency lists, and dead-ends."""
        height, width = map_state.shape
        self.valid_cells = {
            (r, c) for r in range(height) for c in range(width) if map_state[r, c] == 0
        }
        
        self.neighbor_cache = {}
        for r, c in self.valid_cells:
            neighbors = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.valid_cells:
                    neighbors.append(((nr, nc), move))
            self.neighbor_cache[(r, c)] = neighbors

        # Iteratively identify structural dead-ends (corridors with 1 escape route)
        degrees = {cell: len(neighs) for cell, neighs in self.neighbor_cache.items()}
        dead_ends = set()
        queue = [cell for cell, deg in degrees.items() if deg <= 1]
        while queue:
            curr = queue.pop(0)
            dead_ends.add(curr)
            for neighbor, _ in self.neighbor_cache.get(curr, []):
                if neighbor not in dead_ends:
                    valid_neighs = [n for n, _ in self.neighbor_cache[neighbor] if n not in dead_ends]
                    if len(valid_neighs) <= 1:
                        queue.append(neighbor)
        self.dead_ends = dead_ends
        self.initialized = True

    def _get_distance(self, start: tuple, goal: tuple) -> int:
        """Fast BFS shortest-path distance."""
        if start == goal:
            return 0
        queue = [(start, 0)]
        visited = {start}
        while queue:
            curr, dist = queue.pop(0)
            if curr == goal:
                return dist
            for neighbor, _ in self.neighbor_cache.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999

    def _analyze_territory(self, ghost_pos: tuple, pac_pos: tuple) -> tuple:
        """Perform a custom flood-fill to get reachable territory and branching metric."""
        queue = [ghost_pos]
        visited = {ghost_pos}
        total_branching = 0
        
        while queue:
            curr = queue.pop(0)
            neighbors = self.neighbor_cache.get(curr, [])
            total_branching += len(neighbors)
            for neighbor, _ in neighbors:
                if neighbor != pac_pos and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        territory_size = len(visited)
        branching_factor = total_branching / max(1, territory_size)
        return territory_size, branching_factor

    def _evaluate_state(self, ghost_pos: tuple, pac_pos: tuple) -> float:
        """Robust multi-criteria territory escape evaluation."""
        if ghost_pos == pac_pos:
            return -10000.0

        pac_dist = self._get_distance(ghost_pos, pac_pos)
        territory_size, branching_factor = self._analyze_territory(ghost_pos, pac_pos)
        
        dead_end_penalty = 1.0 if ghost_pos in self.dead_ends else 0.0
        trap_risk = 1.0 if pac_dist <= 3 else 0.0

        # Weighted survival metric formula
        score = (
            territory_size * 6.0
            + branching_factor * 10.0
            + pac_dist * 4.0
            - dead_end_penalty * 20.0
            - trap_risk * 25.0
        )
        return score

    def _alphabeta(self, ghost_pos: tuple, pac_pos: tuple, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """Alpha-Beta Minimax search optimized for high survival evasion."""
        if depth == 0 or ghost_pos == pac_pos:
            return self._evaluate_state(ghost_pos, pac_pos)

        if is_max:
            # Ghost Turn - maximize escape paths and distance
            max_val = -float('inf')
            candidates = []
            for neighbor, move in self.neighbor_cache.get(ghost_pos, []):
                dist = abs(neighbor[0] - pac_pos[0]) + abs(neighbor[1] - pac_pos[1])
                candidates.append((dist, neighbor, move))
            # Sort candidates to check the most promising moves away from Pacman first
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates.append((abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1]), ghost_pos, Move.STAY))

            for _, next_pos, _ in candidates:
                val = self._alphabeta(next_pos, pac_pos, depth - 1, alpha, beta, False)
                max_val = max(max_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_val
        else:
            # Pacman Turn - minimize Ghost's survival options
            min_val = float('inf')
            pac_moves = []
            # Calculate all potential target positions Pacman could move to using its speed
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                for steps in range(1, self.pacman_speed + 1):
                    next_pos = pac_pos
                    valid = True
                    for _ in range(steps):
                        dr, dc = move.value
                        candidate = (next_pos[0] + dr, next_pos[1] + dc)
                        if candidate in self.valid_cells:
                            next_pos = candidate
                        else:
                            valid = False
                            break
                    if not valid and next_pos == pac_pos:
                        continue
                    dist = abs(next_pos[0] - ghost_pos[0]) + abs(next_pos[1] - ghost_pos[1])
                    pac_moves.append((dist, next_pos))
                    
            # Sort: Pacman chooses paths minimizing the distance to Ghost
            pac_moves.sort(key=lambda x: x[0])
            for _, next_pos in pac_moves[:4]:  # Prune search space branching factor
                val = self._alphabeta(ghost_pos, next_pos, depth - 1, alpha, beta, True)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """Determine optimal survival action using Minimax with dynamic depth control."""
        if not self.initialized:
            self._init_caches(map_state)
            
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos

        # Fallback Behavior: patrol/maximize space if threat is unseen
        if threat is None:
            best_move = Move.STAY
            best_score = -float('inf')
            for neighbor, move in self.neighbor_cache.get(my_position, []):
                score = len(self.neighbor_cache.get(neighbor, []))
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move

        best_move = Move.STAY
        best_score = -float('inf')

        # Generate, rank and filter Ghost candidate moves
        candidates = []
        for neighbor, move in self.neighbor_cache.get(my_position, []):
            dist = abs(neighbor[0] - threat[0]) + abs(neighbor[1] - threat[1])
            candidates.append((dist, neighbor, move))
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates.append((abs(my_position[0] - threat[0]) + abs(my_position[1] - threat[1]), my_position, Move.STAY))

        # Shallow 4-depth lookahead evaluation (2 moves for Ghost, 2 moves for Pacman)
        for _, next_pos, move in candidates:
            score = self._alphabeta(next_pos, threat, depth=4, alpha=-float('inf'), beta=float('inf'), is_max=False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
