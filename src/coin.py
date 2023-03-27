import numpy as np

from options import Blocks

class Coin():
    """A coin object is assigned to a coin on the map
    """
    def __init__(self,
                position: np.array) -> None:
        """Constructor for the coin

        All coins are managed within a list. As soon as a coin is collected by
        Pacman, the coin is tahn removed from the list including its 
        corresponding object is deleted.  

        Args:
            position (np.array): fix position (y, x) of the coin
        """
        self.position = position