# %%
import argparse
import os
import random
import time
import sys
from distutils.util import strtobool
from dataclasses import dataclass

from typing import Optional
import numpy as np
import torch
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from typing import Any, List, Optional, Union, Tuple, Iterable
from einops import rearrange
from w4d3_chapter4_ppo.utils import ppo_parse_args, make_env
import w4d2_chapter4_dqn.solutions # register the probe environments

MAIN = __name__ == "__main__"

# %%
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        "SOLUTION"
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
def test_agent(Agent):
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test-run") for i in range(5)])
    print(envs.single_observation_space.shape)
    agent = Agent(envs)
    print(sum(p.numel() for p in agent.critic.parameters()))
    print(agent.critic[-1].weight.std())

test_agent(Agent)

# %%
@torch.no_grad()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    """Compute advantages using Generalized Advantage Estimation.
    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)
    Return: shape (t, env)
    """
    "SOLUTION"
    T, _ = values.shape
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages

# %%


@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor


def minibatch_indexes(batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.
    Each index should appear exactly once.
    """
    assert batch_size % minibatch_size == 0
    "SOLUTION"
    b_inds = np.arange(batch_size)
    np.random.shuffle(b_inds)
    return [b_inds[start : start + minibatch_size] for start in range(0, batch_size, minibatch_size)]


def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    """Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches."""
    "SOLUTION"
    b_obs = obs.reshape((-1,) + obs_shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_shape)
    b_advantages = advantages.reshape(-1)
    b_values = values.reshape(-1)
    b_returns = b_advantages + b_values
    return [
        Minibatch(b_obs[ind], b_logprobs[ind], b_actions[ind], b_advantages[ind], b_returns[ind], b_values[ind])
        for ind in minibatch_indexes(batch_size, minibatch_size)
    ]

# %%
def calc_policy_loss(
    probs: Categorical,
    mb_action: t.Tensor,
    mb_advantages: t.Tensor,
    mb_logprobs: t.Tensor,
    clip_coef: float,
    normalize: bool = True,
) -> t.Tensor:
    """Return the negative policy loss, suitable for minimization with gradient descent.
    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)
    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    """
    "SOLUTION"
    newlogprob: t.Tensor = probs.log_prob(mb_action)  # TBD: I don't fully understand this part
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()
    # CM: removed the toggle so we always normalize advantages
    if normalize:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return pg_loss


# %%
def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, v_coef: float) -> t.Tensor:
    """Compute the value function portion of the loss function.
    v_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    """
    "SOLUTION"
    newvalue = critic(mb_obs)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
    return v_coef * v_loss


# %%
def calc_entropy_loss(
    probs: Categorical,
    ent_coef: float,
):
    """Return the entropy loss term.
    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    """
    "SOLUTION"
    entropy = probs.entropy()
    loss = -ent_coef * entropy.mean()
    assert loss <= 0
    return loss


# %%
class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        """Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr."""
        "SOLUTION"
        self.n_step_calls += 1
        frac = 1.0 - (self.n_step_calls - 1.0) / self.num_updates
        lr_now = frac * self.initial_lr + (1 - frac) * self.end_lr
        self.optimizer.param_groups[0]["lr"] = lr_now


def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    """Return an appropriately configured Adam with its attached scheduler."""
    """SOLUTION"""
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
    return optimizer, scheduler


# %%

@dataclass
class PPOArgs:
    exp_name : str = os.path.basename(__file__).rstrip(".py")
    seed : int = 1
    torch_deterministic : bool = True
    cuda : bool = True
    track : bool = True
    wandb_project_name : str = "PPOCart"
    wandb_entity : str = None
    capture_video : bool = False
    env_id : str = "CartPole-v1"
    total_timesteps : int = 500000
    learning_rate : float = 2.5e-4
    num_envs : int = 4
    num_steps : int = 128
    gamma : float = 0.99
    gae_lambda : float = 0.95
    num_minibatches : int = 4
    update_epochs : int = 4
    clip_coef : float = 0.2
    ent_coef : float = 0.01
    vf_coef : float = 0.5
    max_grad_norm : float = 0.5
    batch_size : int = 512
    minibatch_size : int = 128

# %%

def train_ppo(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    optimizer, scheduler = make_optimizer(agent, num_updates, args.learning_rate, 0.0)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for _ in range(num_updates):
        for i in range(0, args.num_steps):
            if "SOLUTION":
                global_step += args.num_envs
                obs[i] = next_obs
                dones[i] = next_done
                with torch.no_grad():
                    logits = agent.actor(next_obs)
                    value = agent.critic(next_obs)
                probs = Categorical(logits=logits)
                action = probs.sample()
                values[i] = value.flatten()
                actions[i] = action
                logprobs[i] = probs.log_prob(action)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[i] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(done).to(device)
            else:
                """YOUR CODE: Rollout phase (see detail #1)"""

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )

        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:
                if "SOLUTION":
                    logits = agent.actor(mb.obs)
                    probs = Categorical(logits=logits)
                    policy_loss = calc_policy_loss(
                        probs,
                        mb.actions,
                        mb.advantages,
                        mb.logprobs,
                        args.clip_coef,
                    )
                    value_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, args.vf_coef)
                    entropy_loss = calc_entropy_loss(probs, args.ent_coef)
                    loss = policy_loss + entropy_loss + value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                else:
                    """YOUR CODE: compute loss on the minibatch and step the optimizer (not the scheduler). Do detail #11 (global gradient clipping) here using nn.utils.clip_grad_norm_."""

        scheduler.step()
        y_pred, y_true = mb.values.cpu().numpy(), mb.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean().item()
            approx_kl = ((ratio - 1) - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if global_step % 10 == 0:
            print("steps per second (SPS):", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()
    if "SKIP":
        import utils

        if args.env_id == "Probe1-v0":
            batch = t.tensor([[0.0]]).to(device)
            value = agent.critic(batch)
            print("Value: ", value)
            expected = t.tensor([[1.0]]).to(device)
            utils.allclose_atol(value, expected, 1e-4)
        elif args.env_id == "Probe2-v0":
            batch = t.tensor([[-1.0], [+1.0]]).to(device)
            value = agent.critic(batch)
            print("Value:", value)
            expected = batch
            utils.allclose_atol(value, expected, 1e-4)
        elif args.env_id == "Probe3-v0":
            batch = t.tensor([[0.0], [1.0]]).to(device)
            value = agent.critic(batch)
            print("Value: ", value)
            # TBD: is this right if we use GAE?
            # Seems to be giving .995 for each, idk if that's right
            expected = t.tensor([[args.gamma], [1.0]])
            utils.allclose_atol(value, expected, 1e-4)
        elif args.env_id == "Probe4-v0":
            # 30K steps
            batch = t.tensor([[0.0]]).to(device)
            value = agent.critic(batch)
            expected_value = t.tensor([[1.0]])
            print("Value: ", value)
            policy_probs = agent.actor(batch).softmax(dim=-1)
            expected_probs = t.tensor([[0, 1]]).to(device)
            print("Policy: ", policy_probs)
            utils.allclose_atol(policy_probs, expected_probs, 1e-2)
            utils.allclose_atol(value, expected_value, 1e-2)
        elif args.env_id == "Probe5-v0":
            batch = t.tensor([[0.0], [1.0]]).to(device)
            value = agent.critic(batch)
            expected_value = t.tensor([[1.0], [1.0]]).to(device)
            print("Value: ", value)
            policy_probs = agent.actor(batch).softmax(dim=-1)
            expected_probs = t.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device)
            print("Policy: ", policy_probs)
            utils.allclose_atol(policy_probs, expected_probs, 1e-2)
            utils.allclose_atol(value, expected_value, 1e-2)

# %%
from gym.envs.classic_control.cartpole import CartPoleEnv
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import math

class EasyCart(CartPoleEnv):

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if "SOLUTION":
            x, v, theta, omega = obs
            reward = 1 - (x/2.5)**2
            return obs, reward, done, info

gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)



class SpinCart(CartPoleEnv):

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if "SOLUTION":
            x, v, theta, omega = obs
            reward = 0.5*abs(omega) - (x/2.5)**4
            if abs(x) > self.x_threshold:
                done = True
            else:
                done = False 
            return obs, reward, done, info

gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)

# class DanceCart(CartPoleEnv):

#     def __init__(self):
#         super().__init__()

#         # Angle at which to fail the episode
#         self.theta_threshold_radians = 60 * 2 * math.pi / 360
#         self.x_threshold = 2.4

#         # Angle limit set to 2 * theta_threshold_radians so failing observation
#         # is still within bounds.
#         high = np.array(
#             [
#                 self.x_threshold * 2,
#                 np.finfo(np.float32).max,
#                 self.theta_threshold_radians * 2,
#                 np.finfo(np.float32).max,
#             ],
#             dtype=np.float32,
#         )
#         self.observation_space = spaces.Box(-high, high, dtype=np.float32)


#     def step(self, action):
#         obs, rew, done, info = super().step(action)
#         if "SOLUTION":
#             x, v, theta, omega = obs

#             if abs(x) > self.x_threshold:
#                 done = True
#             else:
#                 done = False 

#             rew = 0.1*abs(v) - max(abs(x) - 1, 0)**2
              

#             theta = (theta + math.pi) % (2 * math.pi) - math.pi #wrap angle around

#             return np.array([x, v, theta, omega]), rew, done, info

# gym.envs.registration.register(id="DanceCart-v0", entry_point=DanceCart, max_episode_steps=1000)



if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
    train_ppo(args)
