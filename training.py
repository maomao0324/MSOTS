import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from collections import deque
from networks import Actor, FCCritic
from utils import device

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """计算广义优势估计(GAE)"""
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    return advantages

def train_mappo(env, num_agents, agents, num_episodes, batch_size, max_steps, 
                actor_lr=1e-4, critic_lr=3e-4):
    """训练MAPPO算法 """
    
    # 初始化Critic和共享Actor
    shared_actor = Actor(env.task_feature_dim, hidden_dim=128).to(device)
    critic_input_dim = 10
    critic = FCCritic(critic_input_dim, hidden_dim=128).to(device)

    default_agent_params = agents[0]
    shared_actor_optimizer = optim.Adam(
        shared_actor.parameters(), 
        lr=8e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    critic_optimizer = optim.Adam(
        critic.parameters(), 
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    actor_scheduler = StepLR(
        shared_actor_optimizer, 
        step_size=40,
        gamma=0.9
    )
    critic_scheduler = StepLR(
        critic_optimizer, 
        step_size=40,
        gamma=0.9
    )
    
    for agent in agents:
        agent.actor = shared_actor
        agent.actor_optimizer = shared_actor_optimizer
    
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    entropy_values = []
    completion_rates = []
    uniformity_scores = []
    timeliness_scores = []
    
    recent_actor_losses = deque(maxlen=10)
    stability_threshold = 0.5
    
    reward_improvement_threshold = 10.0
    last_check_episode = 0
    
    episode_reward_components = []
    
    all_ep_revisit_first = []
    all_ep_revisit_later = []
    all_ep_completion = []
    all_events_r_first = []
    all_events_r_later = []
    all_events_r_comp = []
    ep_totals_first = []
    ep_totals_later = []
    ep_totals_comp = []
    all_ep_waiting_prob_means = [[] for _ in range(num_agents)]

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        
        trajectory = {
            'obs': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': [],
            'global_features': []
        }
        
        episode_reward = 0
        
        for step in range(max_steps):
            actions = []
            log_probs = []
            
            with torch.no_grad():
                for agent_id in range(num_agents):
                    if obs[agent_id] and obs[agent_id]['num_valid_tasks'] is not None:
                        action, log_prob = agents[agent_id].select_action(obs[agent_id])
                    else:
                        action, log_prob = None, None
                    actions.append(action)
                    log_probs.append(log_prob)
            
            current_global_features = env.get_global_state_features()
            trajectory['global_features'].append(current_global_features)
            
            next_obs, rewards, done, info = env.step(actions)
            
            trajectory['obs'].append(obs)
            trajectory['actions'].append(actions)
            trajectory['rewards'].append(rewards)
            trajectory['dones'].append(done)
            trajectory['log_probs'].append(log_probs)
            
            obs = next_obs
            episode_reward += sum(rewards)
            
            if done:
                break
        
        total_observations = sum(task.observation_counts for task in env.tasks)
        completed_observations = sum(task.current_observation_counts for task in env.tasks)
        observation_completion_rate = completed_observations / total_observations if total_observations > 0 else 0
        completion_rates.append(observation_completion_rate)
        
        uniformity_mse = env.calculate_avg_uniformity_mse()
        timeliness_score = env.calculate_avg_timeliness_score()
        uniformity_scores.append(uniformity_mse)
        timeliness_scores.append(timeliness_score)
        
        episode_component_summary = {
            'revisit_reward': 0.0,
            'completion_reward': 0.0,
            'interval_penalty': 0.0,
            'final_reward': 0.0,
            'count': 0
        }
        
        if hasattr(env, 'episode_reward_components') and env.episode_reward_components:
            for components in env.episode_reward_components:
                for key in episode_component_summary.keys():
                    if key != 'count' and key in components:
                        episode_component_summary[key] += components[key]
                episode_component_summary['count'] += 1
        
        # 只保留观测完成率、时效性、均匀性的打印
        completion_info = f"  观测完成率: {observation_completion_rate:.2%}, 均匀性MSE: {uniformity_mse:.4f}, 时效性: {timeliness_score:.2f}"
        print(completion_info)
        
        if hasattr(env, 'episode_reward_components') and env.episode_reward_components:
            r_first, r_later, r_comp = [], [], []
            for comp in env.episode_reward_components:
                if 'revisit_reward_first' in comp:
                    r_first.append(comp.get('revisit_reward_first', 0.0))
                    r_later.append(comp.get('revisit_reward_later', 0.0))
                    r_comp.append(comp.get('completion_reward', 0.0))
            all_events_r_first.extend(r_first)
            all_events_r_later.extend(r_later)
            all_events_r_comp.extend(r_comp)
            ep_totals_first.append(float(np.sum(r_first)))
            ep_totals_later.append(float(np.sum(r_later)))
            ep_totals_comp.append(float(np.sum(r_comp)))
            if r_first:
                all_ep_revisit_first.append(float(np.mean(r_first)))
                all_ep_revisit_later.append(float(np.mean(r_later)))
                all_ep_completion.append(float(np.mean(r_comp)))
        
        for i, agent in enumerate(agents):
            if hasattr(agent, '_waiting_probs_buffer') and agent._waiting_probs_buffer:
                mean_waiting = float(np.mean(agent._waiting_probs_buffer))
                all_ep_waiting_prob_means[i].append(mean_waiting)
                agent.waiting_prob_history.append(mean_waiting)
                agent._waiting_probs_buffer = []

        update_epochs = 2
        
        with torch.no_grad():
            values = []
            for g in trajectory['global_features']:
                global_features_tensor = torch.tensor(g, dtype=torch.float32, device=device)
                timestep_value = critic(global_features_tensor.unsqueeze(0)).squeeze(0).item()
                values.append(timestep_value)
        
        next_value = 0.0
        if not trajectory['dones'][-1]:
            with torch.no_grad():
                global_features = env.get_global_state_features()
                global_features_tensor = torch.tensor(global_features, dtype=torch.float32, device=device)
                next_value = critic(global_features_tensor.unsqueeze(0)).squeeze(0).item()

        summed_rewards = [sum(r) for r in trajectory['rewards']]
        
        advantages = compute_gae(summed_rewards, values, trajectory['dones'], next_value, 
                                 default_agent_params.gamma, default_agent_params.gae_lambda)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=device)
        
        if advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = advantages - adv_mean
        
        num_timesteps = len(trajectory['obs'])
        all_agent_obs = [obs for step_obs in trajectory['obs'] for obs in step_obs]
        
        all_actions = []
        for t in range(num_timesteps):
            for agent_id in range(num_agents):
                action = trajectory['actions'][t][agent_id]
                num_valid = trajectory['obs'][t][agent_id]['num_valid_tasks']
                action_idx = num_valid if action is None else action
                all_actions.append(action_idx)

        all_log_probs = [lp for step_lp in trajectory['log_probs'] for lp in step_lp]
        
        valid_indices = [i for i, (lp, obs) in enumerate(zip(all_log_probs, all_agent_obs)) 
                        if lp is not None and obs['num_valid_tasks'] is not None]
        
        if not valid_indices:
            continue

        ep_actor_losses = []
        ep_critic_losses = []
        ep_entropy_values = []
        for _ in range(update_epochs):
            perm = np.random.permutation(valid_indices)
            
            for i in range(0, len(perm), batch_size):
                batch_indices = perm[i:i+batch_size]
                
                mb_obs = [all_agent_obs[j] for j in batch_indices]
                mb_actions = torch.tensor([all_actions[j] for j in batch_indices], dtype=torch.long, device=device)
                mb_old_log_probs = torch.tensor([all_log_probs[j] for j in batch_indices], device=device)
                
                timestep_indices = [idx // num_agents for idx in batch_indices]
                mb_advantages = advantages[timestep_indices]
                mb_returns = returns[timestep_indices]

                action_probs_list = [shared_actor(o['task_features']) for o in mb_obs]
                dists = [Categorical(probs) for probs in action_probs_list]
                new_log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, mb_actions)])
                entropy = torch.stack([dist.entropy() for dist in dists]).mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 0.2, 5.0)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - default_agent_params.clip_ratio, 1.0 + default_agent_params.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                if ratio.numel() > 1:
                    ratio_std = ratio.std()
                    if ratio_std > 3.0:
                        stability_penalty = 0.05 * ratio_std
                        actor_loss += stability_penalty
                
                batch_state_features = torch.tensor(
                    [trajectory['global_features'][t] for t in timestep_indices],
                    dtype=torch.float32, device=device
                )
                mb_recomputed_values = critic(batch_state_features).squeeze(-1)

                mb_recomputed_values = torch.clamp(mb_recomputed_values, -100, 100)
                mb_returns = torch.clamp(mb_returns, -100, 100)
                
                ppo_actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(mb_recomputed_values, mb_returns)
                critic_loss = torch.clamp(critic_loss, 0, 1000)
                entropy_loss = -default_agent_params.entropy_coeff * entropy
                
                total_loss = ppo_actor_loss + 0.5 * critic_loss + entropy_loss

                shared_actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(shared_actor.parameters(), 0.4)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                shared_actor_optimizer.step()
                critic_optimizer.step()
                
                if episode > 0 and episode % 40 == 0:
                    actor_scheduler.step()
                    critic_scheduler.step()

                ep_actor_losses.append(ppo_actor_loss.item())
                ep_critic_losses.append(critic_loss.item())
                ep_entropy_values.append(entropy.item())

        episode_rewards.append(episode_reward)
        
        if ep_actor_losses:
            avg_actor_loss = float(np.mean(ep_actor_losses))
            actor_losses.append(avg_actor_loss)
            recent_actor_losses.append(avg_actor_loss)
            
        if len(recent_actor_losses) >= 5:
            loss_std = float(np.std(recent_actor_losses))
        
        if ep_critic_losses:
            critic_losses.append(float(np.mean(ep_critic_losses)))
        if ep_entropy_values:
            entropy_values.append(float(np.mean(ep_entropy_values)))
    
    return agents, shared_actor, critic
