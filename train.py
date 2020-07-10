import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

import time
import collections
from model import v_net
from tetris_agent import Agent

from tensorboardX import SummaryWriter

MEAN_REWARD_BOUND = 1000

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.1
EPSILON_FINAL = 0.01

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, exp):
        self.buffer.append(exp)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states, dtype=np.float32), np.array(actions), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), \
                np.array(next_states)
    

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_values = net(states_v)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v)
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v.unsqueeze(-1)
    
    return nn.MSELoss()(state_values,
                        expected_state_action_values)



if __name__ == "__main__":
    env = gym.make("gym_tetris:tetris-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = v_net(env.observation_space.shape).to(device)
    tgt_net = v_net(env.observation_space.shape).to(device)
    

    writer = SummaryWriter(comment="-tetris")

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.step(net, epsilon, device=device)
        if reward is not None: # just ended one episode
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), 
                           "tetris-best_%.0f.pit" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if len(total_rewards) % 1000 == 0:
                torch.save(net.state_dict(), 
                           "tetris-%d.pit" % len(total_rewards))
            if m_reward > MEAN_REWARD_BOUND:
                torch.save(net.state_dict(), 
                           "tetris-best_%.0f.pit" % m_reward)
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
