# https://www.youtube.com/watch?v=MEt6rrxH8W4

import argparse
import os
import random
import numpy as np
from distutils.util import strtobool
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym

def make_env(gym_id, seed, idx, capture_video, run_name, score=None):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0 and score is not None:
            #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 50 == 0)
            env=gym.wrappers.RecordVideo(
                env, 
                f"videos/{run_name}", 
                episode_trigger=lambda ep_idx: ep_idx == 0,  # lambda _: True,
                name_prefix=f"best_{score}")
        env.reset(seed=seed) # NOT env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self,envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Agent")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the random number generator")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), 
        default=True, nargs='?', const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), 
        default=True, nargs='?', const=True,
        help="if toggled, cuda is used")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help="if toggled, this experiment will be tracked on W&B (Weights and Biases)")
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) to use for W&B")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help="whether to capture a video of the environment (check out `videos` folder)")
    
        # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggle learning rate annealing for policy and values networks')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation. Whether to use Generalized Advantage Estimation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='The lambda for the General Advantege Estimator')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of minibatches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help='the K epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggle advantage normalization')
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggle value loss clipping. Toggle whenever or not to use a clipper loss for the value function , as per paper')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help='the entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help='the value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence')
    # parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False,
    #     help="whether to save the model")
    # parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False,
    #     help="if toggled, the experiment will be run in debug mode")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    # run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.gym_id}_{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            #monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )
    # for i in range(1, 100):
    #     writer.add_scalar("test_loss", i*2, global_step=i)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda"  if torch.cuda.is_available() and args.cuda else "cpu")
    #print(f"Using device: {device}")

# DEMO1
    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda t: t % 100 == 0)
    # observation, info = env.reset()
    # for _ in range(300):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     if done:
    #         observation, _ = env.reset()
    #         print(f"episodic return: {info['episode']['r']}")
    # env.close()


# DEMO2
    # envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    # observation, info = envs.reset()
    # for _ in range(200):
    #     action = envs.action_space.sample()
    #     observation, reward, terminated, truncated, info = envs.step(action)
    #     done = terminated or truncated
    #     if "episode" in info:
    #         print(f"episodic return: {info['episode']['r'][0]}")
            #  NOTE: there is no `observation = env.reset()` anymore, it is handled by the wrapper    

    # envs setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(
            args.gym_id, 
            args.seed + i, 
            i, 
            args.capture_video, 
            run_name) 
        for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action spaces are supported"
    # print("envs.single_observation_space.shape: ", envs.single_observation_space.shape)
    # print("envs.single_action_space.n: ", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # print(agent)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    # print(num_updates)
    # print("next_obs.shape: ", next_obs.shape)
    # print("agent.get_value(next_obs): ", agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape: ", agent.get_value(next_obs).shape)
    # print()
    # print("agent.get_action_and_value(next_obs): ", agent.get_action_and_value(next_obs))
    

    best_return = -201.0
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac*args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # ALGO LOGIC: action logic
            # sample action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
                
            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = terminated | truncated
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            if "episode" in info:
                for i in range(args.num_envs):
                    if info["_episode"][i]:
                        ep_return = info["episode"]["r"][i]
                        ep_length = info["episode"]["l"][i]

                        if ep_return > best_return:
                            best_return = ep_return
                            print(f"🎉 New best return: {best_return}, recording video!")

                            envs.close()

                            envs = gym.vector.SyncVectorEnv([
                                make_env(
                                    args.gym_id,
                                    args.seed + j,
                                    j,
                                    capture_video=True,      # 👈 enable recording
                                    run_name=run_name,
                                    score=best_return
                                )
                            for j in range(args.num_envs)
                        ])
                        #print(f"env {i} | return={info['episode']['r'][i]} | length={info['episode']['l'][i]}")
                        print(f"global_step: {global_step}, episodic_return: {ep_return}, episodic_length: {ep_length}")
                        #print()
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        break

         # bootstrap reward if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1) #flatten()
            if args.gae:
            # for idx, d in enumerate(next_done):
            #     if d:
            #         next_value[idx] = 0
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values      
            else:
                # not using gae
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done #returns[t] = rewards[t] + args.gamma * next_value * (1.0 - next_done)
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1] # returns[t] = rewards[t] + args.gamma * values[t+1] * (1.0 - dones[t+1])
                        next_return = values[t+1]
                    returns[t] = rewards[t] + args.gamma * next_return * nextnonterminal
            
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print("start and end idx: ", start, end)
                # print("mb_inds: ", mb_inds)
                # print()
                #input("Paused for inspection. Press Enter to continue...")

                _, newlogprob, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)
            
                with torch.no_grad():
                    old_approx_kl = (- logratio).mean()
                    approx_kl = (ratio - 1 - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]

                # normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_values = new_values.view(-1) #reshape(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                                                            new_values - b_values[mb_inds], 
                                                            -args.clip_coef, 
                                                            args.clip_coef
                                                        )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = (new_values - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * v_loss.mean()

                entropy_loss = entropy.mean() # more entropy -> more exploration
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_variance = np.nan
        if var_y > 0:
            explained_variance = 1 - np.var(y_true - y_pred) / var_y


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("charts/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/clipfrac", torch.mean(torch.stack(clipfracs)).item(), global_step)
        writer.add_scalar("charts/explained_variance", explained_variance, global_step)
        print("SPS: ", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    envs.close()
    writer.close()

    # resume from 17:23