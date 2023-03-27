import numpy as np

from options import Blocks, Movements

class Pacman():
    """Pacman is the main character and will be controlled by the agent
    """

    def __init__(self, 
                init_position: np.array,
                map: np.array) -> None:
        """Constructor for Pacman

        Pacman receives once the initialized map, so that he is aware of the
        walls. Therefore, the current state of the map would is not required.

        Args:
            init_position (np.array): starting position of the ghost (y, x)
            map (np.array): Starting map
        """
        self.current_pos = init_position
        self.map = map

    def move(self,
            new_direction: Movements) -> bool:
        """Move Pacman to his new position

        Moves the position of Pacman within its object and returns whether its
        location has been changed. If Pacman runs into a wall, his position does
        not change either, he remains on his old position.

        Args:
            new_direction (Movements): Direction for the new step

        Returns:
            bool: Returns True if Pacman has moved its position
        """

        new_position = self.current_pos + new_direction.direction

        # If the new position equals a wall, Pacman remains on his previous spot
        if self.map[new_position[0], new_position[1]] != Blocks.WALL.id:    
            self.current_pos = new_position
            return True
        return False