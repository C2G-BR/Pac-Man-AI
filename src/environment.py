from __future__ import annotations

import io
import numpy as np
import matplotlib.colors as mat_colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from os.path import join

from ghost import Ghost
from coin import Coin
from pacman import Pacman
from options import Blocks, Movements

class Environment():
    """Environment that manages all objects and the map
    """

    def __init__(self,
                path:str='map.csv',
                max_steps:int=1000,
                percentage_coins:float=0.8,
                reward_for_coin:float=1,
                reward_for_step:float=0,
                reward_for_inactive:float=0,
                reward_for_hit:float=-1,
                reward_for_win:float=10,
                reward_for_max_steps:float=-1) -> None:

        """Constructor for the environment

        The environment manages all the coins and ghosts which is why the
        environment is considered as the center of the application, all players
        interact with each other through the environment.

        The map was developed using Excel/Google Sheets. Each cell was filled
        with a certain value. Through conditional formatting, the color of the
        cell changed afterwards, so you could always see what Excel looked like
        in real time. The Excel was then exported in CSV format. 

        Args:
            path (str, optional): Path to the map. Defaults to 'map.csv'.
            max_steps (int, optional): Max amount of steps possible until
                episode is cut off. Defaults to 1000.
            percentage_coins (float, optional): Probability that an empty field
                will contain a coin at start. The coins are generated newly each
                episode. Defaults to 1.
            reward_for_coin (float, optional): Reward for the agent when
                collecting a coin. Defaults to 1.
            reward_for_step (float, optional): Reward for the agent after
                performing a step (a step equals a time step). Defaults to 0.
            reward_for_inactive (float, optional): Reward for the agent if he
                runs into a wall and therefore won't change his position.
                Defaults to 0.
            reward_for_hit (float, optional): Reward for the agent when
                colliding with ghosts. Defaults to -1.
            reward_for_win (float, optional): Reward for collecting all coins
                and therefore winning. Defaults to 10.
            reward_for_max_steps (float, optional): Reward for reaching max
                amount of steps. Defaults to -1.
        """

        # Initializing rewards
        self.reward_for_coin = reward_for_coin
        self.reward_for_step = reward_for_step
        self.reward_for_inactive = reward_for_inactive
        self.reward_for_hit = reward_for_hit
        self.reward_for_win = reward_for_win
        self.reward_for_max_steps = reward_for_max_steps
        
        # Initializing objects
        self.init_map = self.load_map(path)
        self.map = None
        self.ghosts = []
        self.coins = []
        self.pacman = None

        # Initializing game related measures
        self.max_steps = max_steps
        self.percentage_coins = percentage_coins
        self.max_coins = 0
        self.current_steps = 0
        
        # Initializing visuals for the GUI
        colors = [x.color for x in Blocks]
        self.cmap = mat_colors.ListedColormap(colors)

        # Initializing empty output buffer to store the image later
        self.output_buffer = None

        # Creating all necessary objects
        self.reset()

    def load_map(self, path:str) -> np.array:
        """Loads the map from a CSV-file and parses it to a numy matrix

        Args:
            path (str): Path to the map

        Returns:
            np.array: Map as numpy matrix
        """
        return np.genfromtxt(join(path), dtype = int, delimiter = ',')

    def reset(self) -> np.array:
        """Resets the entire environment

        In order to simulate a new epsiode the environment must be set to
        default. The initial map is reloaded, coins are generated on the playing
        field, Pacman and the ghosts are initialized. 

        Returns:
            np.array: Current map
        """
        self.map = self.init_map
        self.current_steps = 0
        self.gen_coins()
        self.gen_pacman()
        self.gen_ghosts()
        self.update_map()
        return self.map

    def gen_coins(self) -> None:
        """Generates coins on the map

        The coins are randomly positioned around the map. On the positions of
        Pacman, the ghosts and on the walls no coins are positioned. In total,
        only 80% of the available spaces will have a coin placed on them.

        Furthermore, all coins will be appended to the coin list. The maximum
        amount of collectalbe coins corresponds to the initial length of the
        coin list. 
        """

        random_matrix = np.random.rand(*self.map.shape)
        self.map = np.where(((self.map == Blocks.EMPTY.id) & \
                            (random_matrix > (1 - self.percentage_coins))),
                            Blocks.COIN.id, self.map)
        init_coins_position = np.where(self.map == Blocks.COIN.id)
        self.coins = []
        for pos in zip(*init_coins_position):
            self.coins.append(
                Coin(position=np.array(pos))
            )
        self.max_coins = len(self.coins)

    def gen_pacman(self) -> None:
        """Generates/initilaises Pacman

        The Pacman object is created, the initial position is passed to the
        object. Since the position array cotains only one location, the for loop
        will only be iterated once.
        """

        init_pacman_position = np.where(self.map == Blocks.PACMAN.id)
        for pos in zip(*init_pacman_position): #only one iteration
            self.pacman = Pacman(pos, self.map)

    def gen_ghosts(self) -> None:
        """Generates ghosts

        First, the spawn points of the ghosts pre-defined in the map are read.
        For each of these spawn points a ghost is created. All ghosts are added
        to a list that allows the environment to easily communicate with all
        ghosts.
        """

        self.ghosts = []
        init_ghosts_position = np.where(self.map == Blocks.GHOST.id)
        for pos in zip(*init_ghosts_position):
            self.ghosts.append(
                Ghost(start_pos=np.array(pos), map=self.map)
            )

    def check_steps(self) -> bool:
        """Checks if maximum number of defined steps was exceeded 

        Returns:
            bool: true if max. number of steps was exceeded
        """
        return self.max_steps == self.current_steps

    def check_collision(self) -> bool:
        """Checks if Pacman has collided with any ghosts

        Returns:
            bool: true if Pacman hit another ghost
        """
        pacman_position = self.pacman.current_pos
        for ghost in self.ghosts:
            if (pacman_position[0] == ghost.current_pos[0]) and \
               (pacman_position[1] == ghost.current_pos[1]):
                return True
        return False

    def check_coins(self) -> bool:
        """Checks if all coins have been collected by Pacman

        Returns:
            bool: true if length of coin list is empty 🡢 no coins left
        """
        #if win
        coins = len(np.where(self.map == Blocks.COIN.id)[0])
        
        return coins == 0

    def step(self,
            direction:Movements
            ) -> tuple[float, np.array, bool]:
        """Performs a new iteration within current simulation

        During the step method, the ghosts and Pacman are set to the new
        position. In addition, a check is made to see if the simulation is
        complete. Furthermore, the map is updated afterwards. The reward is
        calculated according to the assigned values.

        Args:
            direction (Movements): Direction in that Pacman should move

        Returns:
            tuple[float, np.array, bool]:
                [0]: reward depending on the assigned values
                [1]: current state of the map
                [2]: true if simulation is done
        """
        self.current_steps += 1
        reward = self.reward_for_step

        has_moved = self.pacman.move(direction)
        reward += self.reward_for_inactive if not has_moved else 0

        # Termination criteria: lose.
        collision_by_pacman = self.check_collision()
        reward += self.reward_for_hit if collision_by_pacman else 0
        
        collected_a_coin = self.get_field_id(self.pacman.current_pos,
                                            change_state=True)
        reward += self.reward_for_coin if collected_a_coin else 0

        max_steps_reached = False
        collision_by_ghosts = False
        has_won = False

        # This creates an order of which criteria is more dominant towards the
        # termination of the episode: collision_by_pacman > has_won >
        # collision_by_ghosts > max_steps_reached.
        if not collision_by_pacman:
            # Termination criteria: win.
            has_won = self.check_coins()
            reward += self.reward_for_win if has_won else 0

            if not has_won:
                for ghost in self.ghosts:
                    ghost.move()
                
                # Termination criteria: lose. Collision reward os only added one
                # time.
                collision_by_ghosts = self.check_collision()
                reward += self.reward_for_hit if collision_by_ghosts else 0

                # Termination criteria: lose.
                if not collision_by_ghosts:
                    max_steps_reached = self.check_steps()
                    reward += self.reward_for_max_steps if max_steps_reached \
                        else 0

        done = has_won or collision_by_ghosts or collision_by_pacman or \
            max_steps_reached
        self.update_map()

        return reward, self.map, done
    
    def update_map(self) -> None:
        """Updates the map after each iteration

        This method updates all positions of all mutable objects on the map. To
        do this, the old positions are first overwritten and the fields are set
        to empty. Then all objects are redrawn on the map.
        """

        # resets all old positions to default empty fields
        # 🡢 will be overwritten
        self.map = np.where((self.map == Blocks.GHOST.id),
                            Blocks.EMPTY.id,
                            self.map)
        self.map = np.where((self.map == Blocks.PACMAN.id),
                            Blocks.EMPTY.id,
                            self.map)

        # ========Coins========
        # set new postion
        for coin in self.coins:
            current_pos = coin.position
            self.map[current_pos[0], current_pos[1]] = Blocks.COIN.id

        # ========Ghosts========
        # set new postion
        for ghost in self.ghosts:
            current_pos = ghost.current_pos
            self.map[current_pos[0], current_pos[1]] = Blocks.GHOST.id
    
        # ========Pacman=======
        # set new postion
        current_pos = self.pacman.current_pos
        self.map[current_pos[0], current_pos[1]] = Blocks.PACMAN.id

    def get_field_id(self,
                    current_pos:np.array,
                    change_state:bool = False) -> bool:
        """Returns the truth value if Pacman stepped on a coin field

        After each new move of Pacman it is checked if Pacman is on a field
        where there is a coin located. If this is the case and the flag variable
        ´change_state´ is set to true, the coin is picked up by Pacman, which
        corresponds to removing the coin from the list. In addition, True is
        returned, which indicates a positive outcome. Otherwise False is
        returned, which means that no coin was collected.

        Args:
            current_pos (np.array):
                Position to be checked
            change_state (bool, optional):
                If true the coin will be removed from the coin list, so that
                Pacman can collect the reward only once. Defaults to False.

        Returns:
            bool: True if Pacman collected a coin; otherwise False
        """

        for coin in self.coins:
            if (current_pos[0] == coin.position[0]) and \
               (current_pos[1] == coin.position[1]):
                if change_state:
                   self.coins.remove(coin)
                return True
        return False

    def get_collected_points(self) -> int:
        """Returns the number of collected points by Pacman

        The coins/points collected so far correspond to the difference between
        the initial stock of coins and the current amount. The amount can be
        read from the length of the list in which all coins are stored. 

        Returns:
            int: amount of collected points
        """
        return self.max_coins - len(self.coins)

    def get_patches(self) -> tuple[mpatches.Patch, mpatches.Patch]:
        """Returns patches that are displayed within the GUI
        
        Within the patches the current number of steps and the points so far
        achieved by Pacman from the current simulation are stored. These patches
        are displayed in the upper right corner of the GUI.

        Returns:
            tuple[mpatches.Patch, mpatches.Patch]:
                tuple containing two patches: one for current step (iteration)
                and the second one for the achieved points so far  
        """

        points_patch = mpatches.Patch(color=Blocks.PACMAN.color,
                        label=f'Pacman Coins: {self.get_collected_points()}')
        
        steps_patch = mpatches.Patch(color=Blocks.PACMAN.color,
                        label=f'Pacman Steps: {self.current_steps}')

        return points_patch, steps_patch

    def show(self) -> None:
        """Shows current frame of simulation

        The matplotlib figure updates in real time. In addition, the current
        number of steps and points achieved so far are displayed.
        """

        points_patch, steps_patch = self.get_patches()

        plt.figure(1)
        plt.clf()
        plt.imshow(self.map, cmap=self.cmap)
        plt.legend(handles=[points_patch, steps_patch])
        plt.pause(1e-5)
    
    def write_img_to_buffer(self) -> None:
        """Saves current frame to IO-Buffer

        The same frame that is displayed in the ´show´ method will be now stored
        """
        points_patch, steps_patch = self.get_patches()
        with io.BytesIO() as output:
            plt.matshow(self.map, cmap=self.cmap)
            plt.legend(handles=[points_patch, steps_patch])
            plt.savefig(output, format='png')
            plt.close()
            self.output_buffer = output.getvalue()