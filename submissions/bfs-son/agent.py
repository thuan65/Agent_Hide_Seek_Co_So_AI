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
    Example Ghost agent using a simple evasive strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Simple evasive strategy: move away from Pacman.
        
        When enemy_position is None (limited observation mode),
        uses last known position or moves randomly.
        
        Students should implement better search algorithms like:
        - BFS to find furthest point
        - A* to plan escape route
        - Minimax for adversarial search
        - etc.
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            return self._random_move(my_position, map_state)
        
        # Calculate direction away from threat
        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]
        
        # List of possible moves in order of preference
        moves = []
        
        # Prioritize vertical movement away from Pacman
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement away from Pacman
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
        for move in moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            # Check if move is valid
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no preferred move is valid, try any valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no move is valid, stay
        return Move.STAY

    def _random_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """Random movement when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
