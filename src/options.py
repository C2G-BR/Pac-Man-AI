"""Enum overview

The option file contains two enums. The keys of both enums contain two
associated values which where implemented through a named tuple.
"""
import numpy as np
from collections import namedtuple
from enum import Enum


Block = namedtuple('Block', ['value', 'color'])
class Blocks(Enum):
    """Enum that contains different objects of the map

    This enum class contains all the states that a position on the map can have.
    These states include:
    
    | Object   |   id    |  Color   |
    |----------|---------|----------|
    | Empty    |   0     |  gray    |
    | Wall     |   1     |  black   |
    | Pacman   |   2     |  orange  |
    | Ghost(s) |   3     |  blue    |
    | Coin     |   4     |  yellow  |
        
    """

    @property
    def id(self) -> int:
        """Returns the id of an enum entry

        Returns:
            int: returns id of the corresponding key
        """
        return self.value.value

    @property
    def color(self) -> str:
        """Returns the color of an enum entry

        Returns:
            str: Color for the specific position
        """
        return self.value.color

    EMPTY = Block(0, 'gray')
    WALL = Block(1, 'black')
    PACMAN = Block(2, 'orange')
    GHOST = Block(3, 'blue')
    COIN = Block(4, 'yellow')


Movement = namedtuple('Movement', ['value', 'direction'])
class Movements(Enum):
    """Enum that contains different directions for moveable objects
      
    🡡 := (0, -1) 
    🡣 := (1, 0)
    🡢 := (0, 1)
    🡠 := (0, -1)

    The order of x, y was changed so that by matrix addition with the map the
    position of an object could be changed.
    """
    @property
    def id(self) -> int:
        """Returns the id of an enum entry

        Returns:
            int: returns id of the corresponding key
        """
        return self.value.value

    @property
    def direction(self) -> np.array:
        """Returns a numpy vector containing the direction for an enum entry

        Returns:
            np.array: Vector that represents the direction
        """
        return self.value.direction

    UP = Movement(0, np.array([-1, 0]))
    LEFT = Movement(1, np.array([0, -1]))
    DOWN = Movement(2, np.array([1, 0]))
    RIGHT = Movement(3, np.array([0, 1]))