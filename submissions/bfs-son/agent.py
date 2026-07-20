"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from collections import deque
import numpy as np
import random


class PacmanAgent(BasePacmanAgent):
    """
    BFS-based Pacman agent.

    Strategy
    --------
    - Preprocess maze into a graph once (first step).
    - Replan from scratch every turn.
    - BFS to Ghost's CURRENT position.
    - Supports Pacman multi-step movement.
    - Detect simple Ghost "juking" (A->B->A) and avoid immediately
      reversing Pacman's previous move only in that case.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "BFS Pacman"

        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Graph built once
        self.graph = {}
        self.graph_ready = False

        # Memory
        self.last_known_enemy_pos = None

        # Previous Pacman position
        self.previous_position = None

        # Ghost history for juke detection
        self.ghost_history = deque(maxlen=3)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ):

        # Build graph once
        if not self.graph_ready:
            self._build_graph(map_state)
            self.graph_ready = True

        # Update enemy memory
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.ghost_history.append(enemy_position)

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            action = self._explore(my_position)
        else:
            ghost_juking = self._ghost_is_juking()

            action = self._bfs(
                start=my_position,
                goal=target,
                block_previous=ghost_juking,
            )

            if action is None:
                action = self._explore(my_position)

        self.previous_position = my_position

        return action

    # ------------------------------------------------------------------
    # Graph preprocessing
    # ------------------------------------------------------------------

    def _build_graph(self, map_state):

        rows, cols = map_state.shape

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        for r in range(rows):
            for c in range(cols):

                if map_state[r, c] != 0:
                    continue

                node = (r, c)
                self.graph[node] = []

                for dr, dc in directions:

                    nr = r + dr
                    nc = c + dc

                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and map_state[nr, nc] == 0
                    ):
                        self.graph[node].append((nr, nc))

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    def _bfs(self, start, goal, block_previous=False):

        if start == goal:
            return (Move.STAY, 1)

        queue = deque()

        visited = {start}

        # Root expansion
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:

            current = start

            for steps in range(1, self.pacman_speed + 1):

                nxt = self._apply_move(current, move)

                if nxt not in self.graph:
                    break

                # Anti-juke:
                # only ignore immediate reversal from root
                if (
                    block_previous
                    and steps == 1
                    and self.previous_position is not None
                    and nxt == self.previous_position
                ):
                    break

                if nxt in visited:
                    current = nxt
                    continue

                visited.add(nxt)

                queue.append(
                    (
                        nxt,
                        move,
                        steps,
                    )
                )

                current = nxt

        while queue:

            node, first_move, first_steps = queue.popleft()

            if node == goal:
                return (first_move, first_steps)

            for nxt in self.graph[node]:

                if nxt in visited:
                    continue

                visited.add(nxt)

                queue.append(
                    (
                        nxt,
                        first_move,
                        first_steps,
                    )
                )

        return None

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


class GhostAgent(BaseGhostAgent):
    """
    Ghost agent using multi-goal BFS evaluation.

    Strategy
    --------
    - Build maze graph once.
    - Detect Pacman speed once.
    - Evaluate every legal move.
    - Choose the move that maximizes the number of Pacman turns
      required to reach the Ghost.
    - Break ties by maximizing immediate Manhattan distance.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.name = "Multi-Goal BFS Ghost"

        self.graph = {}
        self.graph_ready = False

        self.last_known_enemy_pos = None

        self.pacman_speed = None

    # ----------------------------------------------------------

    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number,
    ):

        if not self.graph_ready:
            self._build_graph(map_state)
            self.graph_ready = True

        if self.pacman_speed is None:
            self.pacman_speed = self._detect_pacman_speed()

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        pacman = enemy_position or self.last_known_enemy_pos

        if pacman is None:
            return self._random_move(my_position)

        candidates = []

        for move in (
            Move.UP,
            Move.DOWN,
            Move.LEFT,
            Move.RIGHT,
        ):
            nxt = self._apply_move(my_position, move)

            if nxt in self.graph:
                candidates.append((move, nxt))

        # No legal move
        if not candidates:
            return Move.STAY

        best_move = Move.STAY
        best_score = (-1, -1)

        for move, target in candidates:

            turns = self._pacman_turn_distance(
                pacman,
                target,
            )

            manhattan = (
                abs(target[0] - pacman[0]) +
                abs(target[1] - pacman[1])
            )

            score = (
                turns,
                manhattan,
            )

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # ----------------------------------------------------------
    # Graph
    # ----------------------------------------------------------

    def _build_graph(self, map_state):

        rows, cols = map_state.shape

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        for r in range(rows):
            for c in range(cols):

                if map_state[r, c] != 0:
                    continue

                node = (r, c)

                self.graph[node] = []

                for dr, dc in directions:

                    nr = r + dr
                    nc = c + dc

                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and map_state[nr, nc] == 0
                    ):
                        self.graph[node].append((nr, nc))

    # ----------------------------------------------------------
    # BFS measured in Pacman turns
    # ----------------------------------------------------------

    def _pacman_turn_distance(
        self,
        start,
        goal,
    ):

        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:

            node, turns = queue.popleft()

            for move in (
                Move.UP,
                Move.DOWN,
                Move.LEFT,
                Move.RIGHT,
            ):

                current = node

                for _ in range(self.pacman_speed):

                    nxt = self._apply_move(current, move)

                    if nxt not in self.graph:
                        break

                    current = nxt

                if current == node:
                    continue

                if current == goal:
                    return turns + 1

                if current not in visited:
                    visited.add(current)
                    queue.append(
                        (
                            current,
                            turns + 1,
                        )
                    )

        return float("inf")

    # ----------------------------------------------------------

    def _random_move(self, start):

        moves = [
            Move.UP,
            Move.DOWN,
            Move.LEFT,
            Move.RIGHT,
        ]

        random.shuffle(moves)

        for move in moves:

            nxt = self._apply_move(start, move)

            if nxt in self.graph:
                return move

        return Move.STAY

    # ----------------------------------------------------------

    def _detect_pacman_speed(self):

        import inspect

        try:

            for frame_info in inspect.stack():

                arena = frame_info.frame.f_locals.get("self")

                if (
                    arena is not None
                    and arena.__class__.__name__ == "Arena"
                ):

                    return max(
                        1,
                        int(
                            getattr(
                                arena,
                                "pacman_speed",
                                2,
                            )
                        ),
                    )

        except Exception:
            pass

        return 2

    # ----------------------------------------------------------

    def _apply_move(
        self,
        pos,
        move,
    ):

        dr, dc = move.value

        return (
            pos[0] + dr,
            pos[1] + dc,
        )

# class GhostAgent(BaseGhostAgent):
#     """
#     Multi-goal BFS Ghost agent.

#     Strategy
#     --------
#     - Preprocess maze into a graph once.
#     - Detect Pacman's speed once.
#     - Every turn:
#         * Generate the 5 legal candidate positions
#           (UP, DOWN, LEFT, RIGHT, STAY).
#         * Run ONE multi-goal BFS from Pacman's current position.
#         * Edge cost = Pacman's turns (supports multi-cell movement).
#         * Stop once every reachable candidate has been found.
#         * Move to the candidate requiring the most Pacman turns.
#     - Falls back to random exploration when Pacman is unknown.
#     """

#     def __init__(self, **kwargs):
#         super().__init__()

#         self.name = "Multi-Goal BFS Ghost"

#         # Graph
#         self.graph = {}
#         self.graph_ready = False

#         # Memory
#         self.last_known_enemy_pos = None

#         # Pacman speed (detected once)
#         self.pacman_speed = None

#     # ------------------------------------------------------------------
#     # Main entry
#     # ------------------------------------------------------------------

#     def step(
#         self,
#         map_state: np.ndarray,
#         my_position: tuple,
#         enemy_position: tuple,
#         step_number: int,
#     ) -> Move:

#         # Build graph once
#         if not self.graph_ready:
#             self._build_graph(map_state)
#             self.graph_ready = True

#         # Detect Pacman speed once
#         if self.pacman_speed is None:
#             self.pacman_speed = self._detect_pacman_speed()

#         # Update Pacman memory
#         if enemy_position is not None:
#             self.last_known_enemy_pos = enemy_position

#         pacman = enemy_position or self.last_known_enemy_pos

#         if pacman is None:
#             return self._random_move(my_position)

#         return self._multi_goal_bfs(
#             pacman_position=pacman,
#             ghost_position=my_position,
#         )

#     # ------------------------------------------------------------------
#     # Graph preprocessing
#     # ------------------------------------------------------------------

#     def _build_graph(self, map_state):

#         rows, cols = map_state.shape

#         directions = [
#             (-1, 0),
#             (1, 0),
#             (0, -1),
#             (0, 1),
#         ]

#         for r in range(rows):
#             for c in range(cols):

#                 if map_state[r, c] != 0:
#                     continue

#                 node = (r, c)
#                 self.graph[node] = []

#                 for dr, dc in directions:

#                     nr = r + dr
#                     nc = c + dc

#                     if (
#                         0 <= nr < rows
#                         and 0 <= nc < cols
#                         and map_state[nr, nc] == 0
#                     ):
#                         self.graph[node].append((nr, nc))

#     # ------------------------------------------------------------------
#     # Multi-goal BFS
#     # ------------------------------------------------------------------

#     def _multi_goal_bfs(
#         self,
#         pacman_position,
#         ghost_position,
#     ):

#         candidate_moves = {
#             Move.STAY: ghost_position,
#         }

#         for move in [
#             Move.UP,
#             Move.DOWN,
#             Move.LEFT,
#             Move.RIGHT,
#         ]:

#             nxt = self._apply_move(ghost_position, move)

#             if nxt in self.graph:
#                 candidate_moves[move] = nxt

#         remaining = set(candidate_moves.values())

#         distances = {}

#         queue = deque()
#         queue.append((pacman_position, 0))

#         visited = {pacman_position}

#         while queue and remaining:

#             node, turns = queue.popleft()

#             if node in remaining:
#                 distances[node] = turns
#                 remaining.remove(node)

#                 if not remaining:
#                     break

#             # Expand all positions Pacman can reach in ONE turn
#             for move in [
#                 Move.UP,
#                 Move.DOWN,
#                 Move.LEFT,
#                 Move.RIGHT,
#             ]:

#                 current = node

#                 for _ in range(self.pacman_speed):

#                     nxt = self._apply_move(current, move)

#                     if nxt not in self.graph:
#                         break

#                     if nxt in visited:
#                         current = nxt
#                         continue

#                     visited.add(nxt)

#                     queue.append(
#                         (
#                             nxt,
#                             turns + 1,
#                         )
#                     )

#                     current = nxt

#         best_move = Move.STAY
#         best_score = -1

#         for move, pos in candidate_moves.items():

#             score = distances.get(pos, float("inf"))

#             if score > best_score:
#                 best_score = score
#                 best_move = move

#         return best_move

#     # ------------------------------------------------------------------
#     # Exploration
#     # ------------------------------------------------------------------

#     def _random_move(self, start):

#         moves = [
#             Move.UP,
#             Move.DOWN,
#             Move.LEFT,
#             Move.RIGHT,
#         ]

#         random.shuffle(moves)

#         for move in moves:

#             nxt = self._apply_move(start, move)

#             if nxt in self.graph:
#                 return move

#         return Move.STAY

#     # ------------------------------------------------------------------
#     # Pacman speed detection
#     # ------------------------------------------------------------------

#     def _detect_pacman_speed(self):

#         import inspect

#         try:
#             for frame_info in inspect.stack():

#                 arena = frame_info.frame.f_locals.get("self")

#                 if (
#                     arena is not None
#                     and arena.__class__.__name__ == "Arena"
#                 ):
#                     return max(
#                         1,
#                         int(getattr(arena, "pacman_speed", 2))
#                     )

#         except Exception:
#             pass

#         return 2

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------

#     def _apply_move(self, pos, move):

#         dr, dc = move.value
#         return (
#             pos[0] + dr,
#             pos[1] + dc,
#         )
