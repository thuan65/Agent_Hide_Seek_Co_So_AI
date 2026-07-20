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
