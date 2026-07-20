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
from collections import deque as Queue # For BFS queue
import heapq # For A* priority queue
import random

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        super().__init__(**kwargs)
        self.name = "TVS Pacman"
        
         # Memory
        self.last_known_enemy_pos = None

        # Previous Pacman position
        self.previous_position = None

        # Ghost history for juke detection
        self.ghost_history = deque(maxlen=3)
        
        self.height = 0
        self.width = 0
        self.graph = {}
        self.graphInitialized = False
        
    def _init_graph(self, map_state: np.array):
        
        self.height, self.width = map_state.shape
        
        for r in range(self.height):
            for c in range(self.width):
                if map_state[r, c] == 0:
                    currentNode = (r,c)
                    self.graph[currentNode] = []
                    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        dr, dc = move.value
                        r1, c1 = r + dr, c + dc
                        
                        if 0 <= r1 < self.height and 0 <= c1 < self.width:
                            if map_state[r1, c1] == 0:
                                self.graph[currentNode].append((r1, c1))

                                r2, c2 = r + (dr * 2), c + (dc * 2)
                                if 0 <= r2 < self.height and 0 <= c2 < self.width:
                                    if map_state[r2, c2] == 0:
                                        self.graph[currentNode].append((r2, c2))
        self.graphInitialized = True                    
    
    ######################### Search Algorithm Implementation #########################
    def A_star(self, start: tuple, goal: tuple, block_previous=False):
        if start == goal:
            return []

        open_set = []
        # Priority queue stores: (f_score, step_neg_dist, g_score, current_node)
        heapq.heappush(open_set, (0 + self.manhattan_distance(start, goal), 0, 0, start))

        visited = {start: 0}
        parent = {}
        
        while open_set:
            current_f, _, current_g,current_node = heapq.heappop(open_set)

            if current_node == goal: # Path Found
                break

            if current_g > visited.get(current_node, float('inf')):
                continue
            
            for neighbor in self.graph.get(current_node, []):
                
                if (
                    block_previous
                    and self.previous_position is not None
                    # and steps == 1
                    and current_node == self.previous_position
                ):
                    break
                
                tentative_g_score = current_g + 1
                
                if neighbor not in visited or tentative_g_score < visited.get(neighbor, float('inf')):
                    step_distance = self.manhattan_distance(current_node, neighbor)
                    parent[neighbor] = current_node
                    visited[neighbor] = tentative_g_score
                    step_neg_dist = -step_distance
                    f_score = tentative_g_score + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, step_neg_dist ,tentative_g_score, neighbor))

        if goal not in parent:
            return []
        
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = parent[curr]
        path.append(start)
        path.reverse()
        return path
    ############################### Helper methods  ###################################
    def manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _choose_action_and_step_number_next_cell(self, pos1, pos2):
        move = self._get_move_direction(pos1, pos2)
        step = min(self.manhattan_distance(pos1, pos2), self.pacman_speed)
        return (move, step)
        
    def _get_move_direction(self, from_pos: tuple, to_pos: tuple) -> Move:
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]

        if row_diff < 0 and col_diff == 0:
            return Move.UP
        if row_diff > 0 and col_diff == 0:
            return Move.DOWN
        if row_diff == 0 and col_diff < 0:
            return Move.LEFT
        if row_diff == 0 and col_diff > 0:
            return Move.RIGHT

    def _is_valid_position(self, pos: tuple) -> bool:
       return pos in self.graph
   
   # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def _explore(self, start):

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)

        for move in moves:

            current = start

            steps = 0

            while steps < self.pacman_speed:

                nxt = self._apply_move(current, move)

                if nxt not in self.graph:
                    break

                current = nxt
                steps += 1

            if steps > 0:
                return (move, steps)

        return (Move.STAY, 1)

    # ------------------------------------------------------------------
    # Ghost juke detection
    # ------------------------------------------------------------------

    def _ghost_is_juking(self):

        if len(self.ghost_history) < 3:
            return False

        return self.ghost_history[0] == self.ghost_history[2]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_move(self, pos, move):

        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)
   
    ############################### H_M  ###################################
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        if self.graphInitialized == False:
            self._init_graph(map_state)
            self.graphInitialized = True
            
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.ghost_history.append(enemy_position)
            
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            action = self._explore(my_position)
        else:
            ghost_juking = self._ghost_is_juking()

            path = self.A_star(start= my_position, goal= enemy_position, block_previous=ghost_juking)
            if not path or len(path) < 2:
                return (Move.STAY, 1)
        
            action = self._choose_action_and_step_number_next_cell(path[0], path[1])
        
            if action is None:
                action = self._explore(my_position)

        self.previous_position = my_position
        
        return action


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
