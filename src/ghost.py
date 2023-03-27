from __future__ import annotations

import numpy as np
import warnings

from options import Movements, Blocks

class Ghost():
    """Ghost that serves as an enemy for Pacman.
    """
    def __init__(self,
                start_pos:np.array,
                map:np.array) -> None:
        """Constructor for the Ghost

        The main purpose of the ghost is to defeat Pacman by crossing its path
        and hitting it. A ghost can move up, down, right and left. In this case,
        the positions are calculated randomly. But to smooth its movement the
        ghost cannot move to the opposite direction from which the ghost came
        from.

        The ghost receives once the initialized map. Since the ghost should only
        be aware of the walls, it doesn't require the current state of a map to
        calculate a new direction. 

        Args:
            start_pos (np.array): starting position of the ghost (y, x)
            map (np.array): Starting map
        """
        self.map = map
        self.current_pos = start_pos
        self.last_direction = None

    def get_possible_directions(self) -> list[Movements]:
        """Calculate possible directions for the ghost

        In order for the ghost to move smoothly, movements where the ghost runs
        into a wall will be neglected.


        Returns:
            list[Movements]:
                list containing all possible and valid movements that the ghost
                can perform at his current possition
        """
        valid_moves = []

        for x in Movements:
            new_position = self.current_pos + x.direction
            if self.map[new_position[0], new_position[1]] != Blocks.WALL.id:
                valid_moves.append(x)

        return valid_moves

    def remove_last_opposite_direction(
                                        self,
                                        possible_directions:list[Movements]
                                        ) -> list[Movements]:
        """Remove opposite direction from previous step

        To further smoothen the ghost's movement, the opposite directions from
        the direction of the previous step, will also be ignored to refrain
        from zig-zag movements.

        Args:
            possible_directions (list[Movements]):
                list containing all possible and valid movements that the ghost
                can perform at his current possition
        Returns:
            list[Movements]:
                list containing all possible and valid movements that the ghost
                can perform at his current possition excluding the opposite
                direction of his previous direction
        """

        if self.last_direction is None:
            return possible_directions

        if self.last_direction == Movements.UP:
            possible_directions.remove(Movements.DOWN)
        elif self.last_direction == Movements.DOWN:
            possible_directions.remove(Movements.UP)
        elif self.last_direction == Movements.RIGHT:
            possible_directions.remove(Movements.LEFT)
        elif self.last_direction == Movements.LEFT:
            possible_directions.remove(Movements.RIGHT)
        else:
            warnings.warn(f'Error in MATRIX! Unknown direction:'
                          f'{self.last_direction}.')
            
        return possible_directions

    def move(self):
        """Move the ghost to its new position

        After a new valid direction is determined, the mind moves in the new
        direction and updates its position.

        Overview of different situations:
        x: possible path
        ^: last direction of player
        ---------------------------------------
        |-----| |-----| |-----| |-----| |-----|
        |  x  | |  x  | |  x  | |     | |     |
        | x^x | |  ^x | |  ^  | |  ^x | |  ^  |
        |  x  | |  x  | |  x  | |  x  | |  x  |
        |-----| |-----| |-----| |-----| |-----|
        Number of ways:
           4       3       2       2       1  
        """

        possible_directions = self.get_possible_directions()

        if len(possible_directions) > 1:
            possible_directions = self \
                .remove_last_opposite_direction(possible_directions)
        
        new_direction = np.random.choice(possible_directions)
        self.last_direction = new_direction
        self.current_pos = self.current_pos + new_direction.direction