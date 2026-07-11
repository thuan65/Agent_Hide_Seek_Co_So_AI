import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


Position = Tuple[int, int]

MOVE_ORDER = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


class PacmanAgent(BasePacmanAgent):
    """Pacman agent that chases the ghost with BFS/A* style planning."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Pathfinder Pacman"
        self.last_known_enemy_pos: Optional[Position] = None
        self.visited: Set[Position] = set()
        self.path_cache: List[Move] = []
        self.cached_map_signature: Optional[Tuple[int, int, bytes]] = None
        self.valid_cells: Set[Position] = set()

    def _map_signature(self, map_state: np.ndarray) -> Tuple[int, int, bytes]:
        return (map_state.shape[0], map_state.shape[1], map_state.tobytes())

    def _is_valid_position(self, pos: Position, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _neighbors(self, pos: Position, map_state: np.ndarray):
        for move in MOVE_ORDER:
            delta_row, delta_col = move.value
            next_pos = (pos[0] + delta_row, pos[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                yield next_pos, move

    def _build_valid_cells(self, map_state: np.ndarray):
        self.valid_cells = {
            (row, col)
            for row in range(map_state.shape[0])
            for col in range(map_state.shape[1])
            if map_state[row, col] == 0
        }

    def _bfs_path(self, start: Position, goal: Position, map_state: np.ndarray) -> List[Move]:
        if start == goal:
            return []
        if goal not in self.valid_cells or start not in self.valid_cells:
            return []

        queue = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Optional[Move]]] = {
            start: (None, None)
        }

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for next_pos, move in self._neighbors(current, map_state):
                if next_pos not in parent:
                    parent[next_pos] = (current, move)
                    queue.append(next_pos)

        if goal not in parent:
            return []

        path: List[Move] = []
        current = goal
        while parent[current][0] is not None:
            prev, move = parent[current]
            path.append(move)
            current = prev
        path.reverse()
        return path

    def _frontier_targets(self, map_state: np.ndarray) -> List[Position]:
        targets = []
        for row, col in self.valid_cells:
            for move in MOVE_ORDER:
                delta_row, delta_col = move.value
                next_pos = (row + delta_row, col + delta_col)
                if 0 <= next_pos[0] < map_state.shape[0] and 0 <= next_pos[1] < map_state.shape[1]:
                    if map_state[next_pos] == -1:
                        targets.append((row, col))
                        break
        return targets

    def _closest_reachable_target(self, start: Position, candidates: List[Position], map_state: np.ndarray) -> List[Move]:
        best_path: List[Move] = []
        best_length = float("inf")
        for target in candidates:
            path = self._bfs_path(start, target, map_state)
            if path and len(path) < best_length:
                best_path = path
                best_length = len(path)
        return best_path

    def _compress_path(self, path: List[Move]) -> Tuple[Move, int]:
        if not path:
            return Move.STAY, 1
        first_move = path[0]
        steps = 1
        for move in path[1:]:
            if move != first_move or steps >= self.pacman_speed:
                break
            steps += 1
        return first_move, steps

    def _fallback_greedy(self, my_position: Position, target: Position, map_state: np.ndarray) -> Tuple[Move, int]:
        best_move = Move.STAY
        best_score = float("inf")
        for move in MOVE_ORDER:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                continue
            score = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
            if next_pos in self.visited:
                score += 0.5
            if score < best_score:
                best_score = score
                best_move = move
        if best_move == Move.STAY:
            return Move.STAY, 1
        return best_move, 1

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        self.visited.add(my_position)

        current_signature = self._map_signature(map_state)
        if self.cached_map_signature != current_signature:
            self.cached_map_signature = current_signature
            self.path_cache = []
            self._build_valid_cells(map_state)

        target = enemy_position or self.last_known_enemy_pos

        if target is not None:
            path = self._bfs_path(my_position, target, map_state)
            if not path:
                path = self._fallback_greedy(my_position, target, map_state)
                return path
            return self._compress_path(path)

        frontier_targets = self._frontier_targets(map_state)
        if frontier_targets:
            path = self._closest_reachable_target(my_position, frontier_targets, map_state)
            if path:
                return self._compress_path(path)

        for move in MOVE_ORDER:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                return move, 1

        return Move.STAY, 1


class GhostAgent(BaseGhostAgent):
    """Ghost agent that maximizes distance from Pacman and avoids traps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Escape Ghost"
        self.last_known_enemy_pos: Optional[Position] = None
        self.visited: Dict[Position, int] = {}
        self.dead_ends: Set[Position] = set()
        self.cached_map_signature: Optional[Tuple[int, int, bytes]] = None
        self.valid_cells: Set[Position] = set()
        self.neighbor_cache: Dict[Position, List[Position]] = {}

    def _map_signature(self, map_state: np.ndarray) -> Tuple[int, int, bytes]:
        return (map_state.shape[0], map_state.shape[1], map_state.tobytes())

    def _is_valid_position(self, pos: Position, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _neighbors(self, pos: Position, map_state: np.ndarray):
        for move in MOVE_ORDER:
            delta_row, delta_col = move.value
            next_pos = (pos[0] + delta_row, pos[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                yield next_pos, move

    def _build_graph(self, map_state: np.ndarray):
        self.valid_cells = {
            (row, col)
            for row in range(map_state.shape[0])
            for col in range(map_state.shape[1])
            if map_state[row, col] == 0
        }
        self.neighbor_cache = {}
        degree: Dict[Position, int] = {}

        for cell in self.valid_cells:
            neighbors = [neighbor for neighbor, _ in self._neighbors(cell, map_state)]
            self.neighbor_cache[cell] = neighbors
            degree[cell] = len(neighbors)

        queue = deque([cell for cell, deg in degree.items() if deg <= 1])
        dead_ends: Set[Position] = set()
        while queue:
            cell = queue.popleft()
            if cell in dead_ends:
                continue
            dead_ends.add(cell)
            for neighbor in self.neighbor_cache.get(cell, []):
                if neighbor not in dead_ends:
                    remaining = sum(1 for x in self.neighbor_cache.get(neighbor, []) if x not in dead_ends)
                    if remaining <= 1:
                        queue.append(neighbor)
        self.dead_ends = dead_ends

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, dist = queue.popleft()
            if current == goal:
                return dist
            for neighbor in self.neighbor_cache.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999

    def _score_move(self, my_position: Position, move: Move, threat: Position, map_state: np.ndarray) -> float:
        delta_row, delta_col = move.value
        next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
        if not self._is_valid_position(next_pos, map_state):
            return -float("inf")

        distance = abs(next_pos[0] - threat[0]) + abs(next_pos[1] - threat[1])
        graph_distance = self._distance(next_pos, threat)
        visit_penalty = self.visited.get(next_pos, 0) * 1.5
        dead_end_penalty = 12.0 if next_pos in self.dead_ends else 0.0
        wall_pressure = 0
        for neighbor in self.neighbor_cache.get(next_pos, []):
            if len(self.neighbor_cache.get(neighbor, [])) <= 2:
                wall_pressure += 1

        return (
            graph_distance * 6.0
            + distance * 3.0
            + wall_pressure * 2.0
            - visit_penalty
            - dead_end_penalty
        )

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        current_signature = self._map_signature(map_state)
        if self.cached_map_signature != current_signature:
            self.cached_map_signature = current_signature
            self._build_graph(map_state)

        self.visited[my_position] = self.visited.get(my_position, 0) + 1

        threat = enemy_position or self.last_known_enemy_pos

        if threat is None:
            best_move = Move.STAY
            best_score = -float("inf")
            for move in MOVE_ORDER:
                score = self._score_move(my_position, move, my_position, map_state)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move if best_move != Move.STAY else Move.STAY

        best_move = Move.STAY
        best_score = -float("inf")
        for move in MOVE_ORDER:
            score = self._score_move(my_position, move, threat, map_state)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move != Move.STAY:
            return best_move

        for move in MOVE_ORDER:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                return move

        return Move.STAY
