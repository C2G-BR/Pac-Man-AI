from __future__ import annotations

import numpy as np
import torch

class Experience():
    """Experience as used in Experience Replay Buffer"""

    def __init__(self, observation:np.array, action:int, reward:float,
        next_observation:np.array) -> None:
        """Constructor

        Args:
            observation (np.array): Observation of the current step.
            action (int): Action choosen by the agent.
            reward (float): Reward returned by the environment.
            next_observation (np.array): Observation returned by the environment
                after taking the given action.
        """
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

class ReplayBuffer():
    """Replay Buffer containing previous Experiences"""

    def __init__(self, min_size:int, max_size:int, batch_size:int,
        device:torch.device) -> None:
        """Constructor

        Args:
            min_size (int): Smallest size of the buffer. Only when this is
                reached, a batch is generated.
            max_size (int): Maximum size of the buffer. If this is exceeded,
                the corresponding oldest entries are removed.
            batch_size (int): Size of a batch for training.
            device (torch.device): Device on which the training of the model
                takes place. This must be the same device on which the model
                itself lies.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.experiences = []

    def add_experience(self, observation:np.array, action:int, reward:float,
        next_observation:np.array) -> None:
        """Add an Experience object to the Buffer

        If the maximum size of the buffer is exceeded, the oldest entry will be
        removed.

        Args:
            observation (np.array): Observation of the current step.
            action (int): Action choosen by the agent.
            reward (float): Reward returned by the environment.
            next_observation (np.array): Observation returned by the environment
                after taking the given action.
        """
        experience = Experience(observation, action, reward, next_observation)
        if len(self.experiences) > self.max_size:
            del self.experiences[0]
        self.experiences.append(experience)

    def get_batch(self) -> tuple[torch.tensor, torch.tensor, torch.tensor,
        torch.tensor, torch.tensor]:
        """Get Batch of Experiences

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor,
                torch.tensor]:
                    [0]: Mask of bools indicating whether the experience was the
                        last of the episode.
                    [1]: Observation returned by the environment after taking
                        the given action.
                    [2]: Observation of the current step.
                    [3]: Actions taken.
                    [4]: Rewards received.
        """
        if len(self.experiences) < self.min_size:
            return None
        
        batch = np.random.choice(self.experiences, size = self.batch_size)

        non_final_mask = tuple(map(lambda s: s.next_observation is not None,
            batch))
        non_final_mask = torch.tensor(non_final_mask).to(device=self.device,
            dtype=torch.bool)

        non_final_next_states = [torch.tensor(np.array([[s.next_observation]]))\
            for s in batch if s.next_observation is not None]
        non_final_next_states = torch.cat(non_final_next_states) \
            .to(self.device, dtype=torch.float)

        state_batch = [torch.tensor(np.array([[s.observation]])) for s in batch]
        state_batch = torch.cat(state_batch).to(self.device, dtype=torch.float)

        action_batch = [torch.tensor([[s.action.id]]) for s in batch]
        action_batch = torch.cat(action_batch).to(self.device)

        reward_batch = torch.tensor([s.reward for s in batch]).to(self.device)

        return non_final_mask, non_final_next_states, state_batch, \
            action_batch, reward_batch