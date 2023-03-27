import argparse
import glob
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from itertools import count
from os.path import join
from typing import Union
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from environment import Environment
from model import DQModelWithoutCNN, DQModelWithCNNNew, DQModelWithCNN
from recorder import Recorder
from replay_buffer import ReplayBuffer

# Allow duplicate initialization within conda.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MODEL = DQModelWithCNN

def train(map:str, id:Union[None, str]=None, show:bool=True, record:bool=True,
    debug:bool=True, num_episodes:int=1000, target_update:int=25,
    save_state:int=50, batch_size:int=200, max_replay_size:int=1e4,
    gamma:float=0.99, init_epsilon:float=0.99, epsilon_decay:float=0.997,
    epsilon_min:float=0.05, lr:float=1e-4, reward_for_coin:float=1,
    reward_for_step:float=0, reward_for_inactive:float=0,
    reward_for_hit:float=-1, reward_for_win:float=10,
    reward_for_max_steps:float=-1, use_pretrained:bool=False,
    max_steps:int=1000, percentage_coins:float=1, last_episode:int=0) -> None:

    """Train a new reinforcement learning for the Pacman environment

    Args:
        map (str): The name of the file where the map is located in. This file
            must be located inside the 'maps' folder, of type '.csv' and
            comma-separated.
        id (str, optional): An unique identifier used to identify the
            corresponding files. If None, an id based on the date will be
            generated. This does not ensure uniqueness and therefore might
            overwrite files. Defaults to None.
        show (bool, optional): Whether to display the environment when the agent
            plays it. Defaults to True.
        record (bool, optional): Whether to record a video of the environment
            while the agent is playing it. Defaults to True. Defaults to True.
        debug (bool, optional): Whether to use debug mode for logging. Defaults
            to True.
        num_episodes (int, optional): The number of episodes to play. Defaults
            to 1000.
        target_update (int, optional): The number of episodes after which the
            target model will be updated. Defaults to 25.
        save_state (int, optional): The number of episodes after which a new log
            entry will be written. If record is true, this will also create a
            video. Defaults to 50.
        batch_size (int, optional): The number of experiences choosen to update
            the train model. Defaults to 200.
        max_replay_size (int, optional): The maximum number of experiences kept.
            Defaults to 1e4.
        gamma (float, optional): The discount rate for future expected rewards.
            Defaults to 0.99.
        init_epsilon (float, optional): The initial value for epsilon. Defaults
            to 0.99.
        epsilon_decay (float, optional): The initial value for the epsilon decay
            rate. Defaults to 0.997.
        epsilon_min (float, optional): The minimum value for the epsilon.
            Defaults to 0.05.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        reward_for_coin (float, optional): Reward for the agent when collecting
            a coin. Defaults to 1.
        reward_for_step (float, optional): Reward for the agent after performing
            a step (a step equals a time step). Defaults to 0.
        reward_for_inactive (float, optional): Reward for the agent if he runs
            into a wall and therefore won't change his position. Defaults to 0.
        reward_for_win (float, optional): Reward for collecting all coins and
            therefore winning. Defaults to 10.
        reward_for_max_steps (float, optional): Reward for reaching max amount
            of steps. Defaults to -1.
        reward_for_hit (float, optional): Reward for the agent when colliding
            with ghosts. Defaults to -1.
        use_pretrained (bool, optional): Continue training with a pretrained
            model. Defaults to False.
        last_episode (int, optional): Specifies the epsiode of which the
            training will continue. Defaults to 0.
        max_steps (int, optional): Max amount of steps possible until episode is
            cut off. Defaults to 1000.
        percentage_coins (float, optional): Probability that an empty field will
            contain a coin at start. The coins are generated newly each episode.
            Defaults to 1.
    """

    id = id if id is not None else get_date_as_string()

    writer = SummaryWriter(f'tensorboard/{id}')
    logger = get_logger(id, debug)
    logger.info(f'''Hyperparameters: 
                   \t num_episodes: {num_episodes}
                   \t target_update: {target_update}
                   \t batch_size: {batch_size}
                   \t max_replay_size: {max_replay_size}
                   \t gamma: {gamma}
                   \t init_epsilon: {init_epsilon}
                   \t epsilon_decay: {epsilon_decay}
                   \t epsilon_min: {epsilon_min}
                   \t learning_rate: {lr}
                   \t reward_for_coin: {reward_for_coin}
                   \t reward_for_step: {reward_for_step}
                   \t reward_for_inactive: {reward_for_inactive}
                   \t reward_for_hit: {reward_for_hit}
                   \t reward_for_max_steps: {reward_for_max_steps}
                   \t reward_for_win: {reward_for_win}
                   \t max_steps: {max_steps}
                   \t percentage_coins: {percentage_coins}
                   \t use_pretrained: {use_pretrained}
                   \t last_episode: {last_episode}
                    ''')

    env = Environment(path=join('maps', map),
        max_steps=max_steps,
        percentage_coins=percentage_coins,
        reward_for_coin=reward_for_coin,
        reward_for_step=reward_for_step,
        reward_for_inactive=reward_for_inactive,
        reward_for_hit=reward_for_hit,
        reward_for_max_steps=reward_for_max_steps,
        reward_for_win=reward_for_win
        )
    map_size = env.map.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model = MODEL(device, height=map_size[0], width=map_size[1])
    target_model = MODEL(device, height=map_size[0], width=map_size[1])

    if use_pretrained:
        train_model.load(f'{id}_{last_episode}')
    target_model.load_state_dict(train_model.state_dict())
    
    train_agent = Agent(model=train_model.to(device),
        init_epsilon=init_epsilon, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay)
    target_agent = Agent(model=target_model.to(device),
        init_epsilon=init_epsilon, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay)
    replay_buffer = ReplayBuffer(min_size=batch_size,
        max_size=max_replay_size, batch_size=batch_size, device=device)

    optimizer = optim.Adam(train_agent.model.parameters(), lr=lr)
    if record:
        recorder = Recorder()

    # Variables for tracking metrics.
    steps = []
    acc_rewards = []
    avg_losses = []

    for e in range(last_episode+1, last_episode+num_episodes+1):

        losses = []
        acc_reward = 0
        observation = env.reset()
        new_id = f'{id}_{e}'

        if record and (e%save_state == 0):
            recorder.init_new_video(id=new_id)

        for i in count():
            action = train_agent.predict_new_move_eps(
                torch.tensor(np.array([[observation]]), dtype=torch.float))
            reward, next_observation, done = env.step(action)
            acc_reward += reward
            
            if show:
                env.show()
            
            if record and (e%save_state == 0):
                env.write_img_to_buffer()
                recorder.add_image(env.output_buffer)
            
            if done:
                next_observation = None

            replay_buffer.add_experience(observation, action, reward,
                next_observation)

            observation = next_observation

            # Optimization
            batch = replay_buffer.get_batch()
            if batch is not None:
                non_final_mask, non_final_next_states, state_batch, \
                    action_batch, reward_batch = batch
                state_action_values = train_agent.model(state_batch) \
                    .gather(1, action_batch)
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = target_agent.model(
                    non_final_next_states).max(1)[0].detach()
                
                # Discount expected future rewards.
                expected_state_action_values = (next_state_values * gamma) + \
                    reward_batch

                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values,
                    expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                for param in train_agent.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                losses.append(loss.detach().cpu())

            if done:
                break
        
        train_agent.update_eps()

        mean_loss = np.mean(losses) if len(losses) > 0 else 0
        steps.append(i)
        acc_rewards.append(acc_reward)
        avg_losses.append(mean_loss)

        writer.add_scalar('acc_reward', acc_reward, global_step=e)
        writer.add_scalar('steps', i, global_step=e)
        writer.add_scalar('avg_loss', mean_loss, global_step=e)

        if (e % save_state == 0):
            logger.info((f'episode: {e:6d} | #steps: {np.mean(steps):9.5f} | '
                f'avg loss: {np.mean(avg_losses):9.5f} | acc reward: '
                f'{np.mean(acc_rewards):7.2f} | epsilon: '
                f'{train_agent.epsilon:7.5f}'))
            steps = []
            acc_rewards = []
            avg_losses = []
            train_agent.model.save(id=new_id)
            if record:
                recorder.close_recording()

        if e % target_update == 0:
            target_agent.model.load_state_dict(
                train_agent.model.state_dict().copy())

def get_logger(id:str, debug:bool = False) -> logging.Logger:
    """Retrieve logger

    The returned logger 'pacman_rl' writes to a file inside the 'logs' folder
    and streams the logs to the console.

    Args:
        id (str): An unique identifier used to identify the log file.
        debug (bool, optional): Whether to use the debug mode for logging. If
            False, only infos will be logged. Defaults to False.

    Returns:
        logging.Logger: Logger that streams and writes logs.
    """
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('pacman_rl')
    logger.setLevel(level=level)
    fh = logging.FileHandler(os.path.join('logs', f'pacman_{id}.log'), mode='a')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_latest_model_name() -> str:
    """Determines the name of the last added model

    Returns:
        str: Name of the '.model' file that was last added to the 'models'
            folder.
    """
    path = os.path.join(os.getcwd(), 'models', '*')
    list_of_files = glob.glob(path)
    latest_file = max(list_of_files, key=os.path.getctime)
    return os.path.basename(latest_file).split('.')[0]

def get_date_as_string() -> str:
    """Returns the current date as a string

    Returns:
        str: The current date.
    """
    return datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

def play(map:str, model_name:str = None, show:bool=True,
    record:bool=True, max_steps:int=1000, percentage_coins:float=1) -> None:
    """Run an episode of the Pacman environment

    Args:
        map (str): The name of the file where the map is located in. This file
            must be located inside the 'maps' folder, of type '.csv' and
            comma-separated.
        model_name (str, optional): The name of the model inside the 'models'
            folder to be used. If it is not further specified the latest model
            file inside that folder will be taken. Defaults to None.
        show (bool, optional): Whether to display the environment when the agent
            plays it. Defaults to True.
        record (bool, optional): Whether to record a video of the environment
            while the agent is playing it. Defaults to True.
        max_steps (int, optional): Max amount of steps possible until episode is
            cut off.
        percentage_coins (float, optional): Probability that an empty field will
            contain a coin at start. The coins are generated newly each episode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL(device)
    if model_name is None:
        model_name = get_latest_model_name()
        print(f'No model specified. Using latest model: {model_name}.')
    model.load(model_name)
    agent = Agent(model=model.to(device))

    env = Environment(path=join('maps', map), max_steps=max_steps,
        percentage_coins=percentage_coins)
    observation = env.reset()
    if record:
        recorder = Recorder()
        recorder.init_new_video(f'play_{model_name}_{get_date_as_string()}')

    for i in count():
        action = agent.predict_new_move(torch.tensor(np.array([[observation]]),
            dtype=torch.float))
        _, next_observation, done = env.step(action)
        observation = next_observation

        if show:
            env.show()
        if record:
            env.write_img_to_buffer()
            recorder.add_image(env.output_buffer)
        if done:
            break

    if record:
        recorder.close_recording()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Pacman for Reinforcement Learning',
        description='A Pacman environment for reinforcement learning agents.')
    parser.add_argument('-id', '--identifier', type=str, help=('Unique '
        'identifier used to identify the run.'), dest='id', required=True)
    parser.add_argument('-m', '--mode', default='play', type=str, dest='mode',
        choices=['play', 'train'], help='Which function to execute.')
    parser.add_argument('-map', '--map', default='map.csv', type=str,
        dest='map', help='Which map to use. The map must be located inside the '
        '\'maps\' folder, of type \'.csv\' and comma-separated.')
    parser.add_argument('-s', '--show', action='store_true', dest='show',
        help='Whether to show a video of the game while playing.')
    parser.add_argument('-r', '--record', action='store_true',
        help='Whether to record the games while playing.', dest='record')
    parser.add_argument('-d', '--debug', action='store_true',
        help=('Whether to use debug mode for logging. This only applies for '
        'mode \'train\'.'), dest='debug')
    parser.add_argument('-ne', '--num-episodes', default=1000, type=int,
        help=('The number of episodes to play. This only applies for mode '
        '\'train\'.'), dest='num_episodes')
    parser.add_argument('-tu', '--target-update', default=25, type=int,
        help=('The number of episodes after which the target model will be '
        'updated. This only applies for mode \'train\'.'), dest='target_update')
    parser.add_argument('-st', '--save-state', default=50, type=int,
        help=('The number of episodes after which a new log entry will be '
        'written. If record is true, this will also create a video. This only '
        'applies for mode \'train\'.'), dest='save_state')
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
        help=('The number of experiences choosen to update the train model. '
        'This only applies for mode \'train\'.'), dest='batch_size')
    parser.add_argument('-mrs', '--max-replay-size', default=1e4, type=int,
        help=('The maximum number of experiences kept. This only applies for '
        'mode \'train\'.'), dest='max_replay_size')
    parser.add_argument('-ie', '--init-epsilon', default=0.99, type=float,
        help=('The initial value for epsilon. This only applies for mode '
        '\'train\'.'), dest='init_epsilon')
    parser.add_argument('-ed', '--epsilon-decay', default=0.997, type=float,
        help=('The initial value for the epsilon decay rate. This only applies '
        'for mode \'train\'.'), dest='epsilon_decay')
    parser.add_argument('-em', '--epsilon-min', default=0.05, type=float,
        help='The minimum value for the epsilon. This only applies for mode '
        '\'train\'.', dest='epsilon_min')
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
        help='The discount rate for future expected rewards. This only applies '
        'for mode \'train\'.', dest='gamma')
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
        help='Learning rate for the optimizer. This only applies for mode '
        '\'train\'.', dest='learning_rate')
    parser.add_argument('-rfc', '--reward-for-coin', default=1, type=float,
        help='Reward for collecting a coin. This only applies for mode '
        '\'train\'.', dest='reward_for_coin')
    parser.add_argument('-rfs', '--reward-for-step', default=0, type=float,
        help='Reward for performing a step. This only applies for mode '
        '\'train\'.', dest='reward_for_step')
    parser.add_argument('-rfi', '--reward-for-inactive', default=0, type=float,
        help='Reward for no movement. This only applies for mode '
        '\'train\'.', dest='reward_for_inactive')
    parser.add_argument('-rfh', '--reward-for-hit', default=-1, type=float,
        help='Reward for hitting a ghost. This only applies for mode '
        '\'train\'.', dest='reward_for_hit')
    parser.add_argument('-rfw', '--reward-for-win', default=10, type=float,
        help='Reward for collecting all coins. This only applies for mode '
        '\'train\'.', dest='reward_for_win')
    parser.add_argument('-rfms', '--reward-for-max-steps', default=-1,
        type=float, help='Reward for reaching max amount of steps. This only '
        'applies for mode \'train\'.', dest='reward_for_max_steps')
    parser.add_argument('-ms', '--max-steps', default=1000, type=int,
        help='Max amount of steps possible.', dest='max_steps')
    parser.add_argument('-pc', '--percentage-coins', default=1,
        type=float, help='Percentage of randomly generated coins on empty '
        'fields.', dest='percentage_coins')
    parser.add_argument('-up', '--use-pretrained', action='store_true', 
        help='Continue training with pretrained model. This only applies for '
        'mode \'train\'.', dest='use_pretrained')
    parser.add_argument('-le', '--last-episode', default=0, type=int, 
        help='Continue training from last_episode. This only applies for mode'
        ' \'train\'.', dest='last_episode')
 
    args = parser.parse_args()

    if args.mode == 'play':
        play(map=args.map, model_name=args.id, show=args.show,
            record=args.record, max_steps=args.max_steps,
            percentage_coins=args.percentage_coins)
    else:
        train(map=args.map, id=args.id, show=args.show, record=args.record,
            num_episodes=args.num_episodes, target_update=args.target_update,
            batch_size=args.batch_size, max_replay_size=args.max_replay_size,
            gamma=args.gamma, init_epsilon=args.init_epsilon, debug=args.debug,
            epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
            save_state=args.save_state, lr=args.learning_rate,
            reward_for_coin=args.reward_for_coin,
            reward_for_step=args.reward_for_step,
            reward_for_inactive=args.reward_for_inactive,
            reward_for_hit=args.reward_for_hit,
            reward_for_win=args.reward_for_win,
            reward_for_max_steps=args.reward_for_max_steps,
            use_pretrained=args.use_pretrained, last_episode=args.last_episode,
            max_steps=args.max_steps, percentage_coins=args.percentage_coins)