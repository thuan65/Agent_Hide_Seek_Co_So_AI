"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque as Queue # For BFS queue
import heapq # For A* priority queue

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Template Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None

        # ======= Not used ======
        # self.F_W_Cache = []
        # self.Floyd_Warshall_cache_initialized = False
        # # =======================
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """

        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        # Ghost is visible or last known
        ghost_moved_far = False

        if enemy_position is not None and self.last_known_enemy_pos is not None:
            if self.mahattan_distance(enemy_position, self.last_known_enemy_pos) >= 1:
                ghost_moved_far = True

        if len(self.path) == 0 or ghost_moved_far:
            #self.path = self.BFS(my_position, target, map_state)
            self.path = self.A_star(my_position, target, map_state)
            self.last_known_enemy_pos = target

        if self.last_known_enemy_pos is not None:
            distance = abs(my_position[0] )

        action = self._choose_action_and_step_number(self.path)

        if action:
            return action

        return (Move.STAY, 1)
    
    ######################### Search Algorithm Implementation #########################
    def BFS(self, start: tuple, goal: tuple, map_state: np.ndarray):
        BFS_queue = Queue([start])
        visited = {start: None} # Track visited nodes
        
        while BFS_queue:

            current = BFS_queue.popleft()
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return path[::-1]
            else:
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    delta_row, delta_col = move.value
                    neighbor = (current[0] + delta_row, current[1] + delta_col)
                    if self._is_valid_position(neighbor, map_state) and neighbor not in visited:
                        visited[neighbor] = current
                        BFS_queue.append(neighbor)
        return None  # No path found

    def A_star(self, start: tuple, goal: tuple, map_state: np.ndarray):
        if start == goal:
            return [start]
        
        open_set = []
        heapq.heappush(open_set, (0 + self.mahattan_distance(start, goal), 0, start))
        g_score = {start: 0}
        parent = {}

        while open_set:
            current_f, current_g, current_pos = heapq.heappop(open_set)

            if current_pos == goal:
                path = []
                curr = goal
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.append(start)
                path.reverse()
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                neighbor = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                if self._is_valid_position(neighbor, map_state):
                    tentative_g_score = g_score[current_pos] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        parent[neighbor] = current_pos
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.mahattan_distance(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
        return None  # No path found

#         ##################################################################

    #         ##############################Floyd_Warshall#####################
    # def Floyd_Warshall_init(self, map_state: np.ndarray):
    #     INF = 9999999999
        
    #     R = len(map_state)
    #     C = len(map_state[0])
    #     N = R * C
   

    #     self.F_W_Cache = [[INF] * N for _ in range(N)]

    #     for i in range(N):
    #         self.F_W_Cache[i][i] = 0

    #     for r in range (R):
    #         for c in range(C):
    #             if map_state[r][c] == 1:
    #                 continue
    #             current_pos = (r, c)
    #             row_array = r * C + c

    #             for move in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
    #                 delta_row, delta_col = move.value
    #                 neighbor = (current_pos[0] + delta_row, current_pos[1] + delta_col)
    #                 if self._is_valid_position(neighbor, map_state):
    #                     col_array = neighbor[0] * C + neighbor[1]
    #                     self.F_W_Cache[row_array][col_array] = 1
    #     for k in range(N):
    #         for i in range(N):
    #             for j in range(N):
    #                 if self.F_W_Cache[i][k] + self.F_W_Cache[k][j] < self.F_W_Cache[i][j]:
    #                     self.F_W_Cache[i][j] = self.F_W_Cache[i][k] + self.F_W_Cache[k][j]
    #     self.Floyd_Warshall_cache_initialized = True
    ###################################################################################

    # Helper methods (you can add more)

    def mahattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0] + abs(pos1[1] - pos2[1]))

    def _choose_action_and_step_number(self, path):
        if len(path) < 2:
            return (Move.STAY, 1)

        if len(path) >= 3:
            pos0 = path[0]
            pos1 = path[1]
            pos2 = path[2]

            mov1 = self._get_move_direction(pos0, pos1)
            mov2 = self._get_move_direction(pos1, pos2)

            if mov1 == mov2:
                path.pop(0)
                path.pop(0)
                return (mov1, 2)
        
        mov = self._get_move_direction(path[0], path[1])
        path.pop(0)
        return (mov, 1)


    def _get_move_direction(self, from_pos: tuple, to_pos: tuple) -> Move:
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]

        if row_diff == -1 and col_diff == 0:
            return Move.UP
        if row_diff == 1 and col_diff == 0:
            return Move.DOWN
        if row_diff == 0 and col_diff == -1:
            return Move.LEFT
        if row_diff == 0 and col_diff == 1:
            return Move.RIGHT

    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY
        
        # Example: Simple evasive approach (replace with your algorithm)
        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]
        
        # Try to move away from Pacman
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
    # Helper methods (you can add more)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
