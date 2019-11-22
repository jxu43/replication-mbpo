import argparse
import gym
import torch
import numpy as np
from itertools import count
from sac.replay_memory import ReplayMemory
from sac.sac import SAC

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env-name', default="Hopper-v2",
        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
        help='random seed (default: 123456)')


    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                    help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                    help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                    help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                    help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=20, metavar='A',
                    help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                    help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100e3, metavar='A',
                    help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')
    parser.add_argument('--rollout_length', type=int, default=1, metavar='A',
                    help='rollout length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                    help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                    help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.1, metavar='A',
                    help='ratio of env samples / model samples')

    parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: True)')
    return parser.parse_args()

def train(args, env, predict_env, agent, env_pool, model_pool):
    steps_cnt = 0
    for epoch_step in range(args.num_epoch):
        start_step = steps_cnt
        for i in count():
            cur_step = steps_cnt - start_step

            if (cur_step >= start_step + args.epoch_length and len(env_pool) > args.min_pool_size):
                break

            if cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                pass


def train_predict_model(env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((reward, delta_state), axis=-1)

    predict_env.model.train(inputs, labels)


def rollout_model(args, predict_env, agent, model_pool):
    state, action, reward, next_state, done = env_pool.sample(args.rollout_batch_size)
    for i in range(args.rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(obs, act)
        # TODO: Push a batch of samples
        model_pool.push(state, action, rewards, next_states, terminals)
        nonterm_mask = ~term.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def main():
    args = readParser()

    # Initial environment
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    env_model = Ensemble_Model(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(args.rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)



if __name__ == '__main__':
    main()
