import numpy as np

from model import BaseDQModel
from options import Movement, Movements

class Agent():
    """Agent that serves as an interface between the model and the environment.
    """

    def __init__(self,
                model:BaseDQModel,
                init_epsilon:float=0.99,
                epsilon_decay:float=0.997,
                epsilon_min:float=0.05) -> None:
        """Constructor for the agent

        The agent is passed the model that predicts the new direction, as well
        as the initial value, the decay constant, and the minimum value of
        Epsilon.

        Epsilon is ultimately decisive for whether Pacman's new direction is
        determined by the model or by random. Over time, Epsilon decreases, 
        hence the model is used with more frequency. 

        Args:
            model (BaseDQModel):
                Model that is used for predicting the new directions
            init_epsilon (float, optional):
                Initial value for epsilon. Defaults to 0.99.
            epsilon_decay (float, optional):
                Factor by which Epsilon reduces. Defaults to 0.997.
            epsilon_min (float, optional):  
                Lower epsilon limit, from which epsilon must not change anymore.
                Defaults to 0.05.
        """
        
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = model

    def predict_new_move_eps(self, observation: np.array) -> Movements:
        """Determines Pacman's new direction during training 

        This function checks if a random number is above Epsilon. If this is the
        case, the new direction is calculated randomly. Otherwise, the model
        calculates the Pacman's new direction.

        Args:
            observation (np.array): Current state of the map

        Returns:
            Movements: New Direction for Pacman (up, down, left, right) 
        """

        next_move = None

        random = np.random.rand()
        if random < self.epsilon:
            next_move = np.random.choice(list(Movements)) # random move
        else:
            next_move, _ = self.model.predict(observation) # predict move

        return next_move

    def update_eps(self) -> None:
        """Update Epsilon with Decay

        Epsilon is decaying over time to reduce the randomness until it reaches
        a limit. This method should be executed after every episode during
        training.

        """
        if self.epsilon > self.epsilon_min: # check if limit is exceeded
            self.epsilon *= self.epsilon_decay # reduce epsilon
        else:
            self.epsilon = self.epsilon_min
    
    def predict_new_move(self, observation: np.array) -> Movements:
        """Determines Pacman's new direction

        This function works similar to ´predict_new_move_eps´. However, the
        agent permanently instructs the model to predict a new direction.

        Args:
            observation (np.array): Current state of the map

        Returns:
            Movements: New Direction for Pacman (up, down, left, right) 
        """
        next_move, _ = self.model.predict(observation) # predict move
        return next_move