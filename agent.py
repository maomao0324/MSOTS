import torch
import torch.optim as optim
import random
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from networks import Actor, FCCritic
from utils import to_device

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.device = device

    def add(self, *args):
        try:
            new_args = []
            for arg in args:
                if arg is None:
                    new_args.append(None)
                    continue
                elif hasattr(arg, 'x') and hasattr(arg, 'edge_index'):
                    arg = arg.to(self.device)
                    new_args.append(arg)
                elif isinstance(arg, list):
                    processed_list = []
                    for item in arg:
                        if item is None:
                            processed_list.append(None)
                        else:
                            processed_list.append(to_device(item, self.device))
                    new_args.append(processed_list)
                else:
                    new_args.append(to_device(arg, self.device))
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(new_args)
            else:
                self.buffer[self.pos] = new_args
            self.pos = (self.pos + 1) % self.capacity
        except Exception as e:
            pass

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        try:
            indices = random.sample(range(len(self.buffer)), min(len(self.buffer), batch_size))
            batch = [self.buffer[idx] for idx in indices]
            result = []
            for i in range(len(batch[0])):
                data_col = [exp[i] for exp in batch]
                result.append(data_col)
            return result
        except Exception as e:
            return []

    def size(self):
        return len(self.buffer)

# 智能体类
class Agent:
    def __init__(self, task_feature_dim, agent_id, critic_input_dim=10,
                 gamma=0.7, gae_lambda=0.95, clip_ratio=0.15, entropy_coeff=0.03,
                 actor_lr=1e-5,  
                 critic_lr=5e-5,  
                 max_grad_norm=0.5,
                 shared_actor=None, shared_actor_optimizer=None,
                 shared_critic=None, shared_critic_optimizer=None,
                 use_adamw=True):
        self.agent_id = agent_id
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.critic_lr = critic_lr
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化Actor网络
        if shared_actor is not None:
            self.actor = shared_actor
            self.actor_optimizer = shared_actor_optimizer
        else:
            self.actor = Actor(task_feature_dim, hidden_dim=128).to(self.device)
                 
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=8e-5,  
            betas=(0.9, 0.999),
            eps=1e-8
        )
            
        # 初始化Critic网络
        if shared_critic is not None:
            self.critic = shared_critic
            self.critic_optimizer = shared_critic_optimizer
        else:
            self.critic = FCCritic(critic_input_dim, hidden_dim=128).to(self.device)
    
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), 
                lr=5e-5,  
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
        
        # Actor: 使用StepLR
        self.actor_scheduler = StepLR(
            self.actor_optimizer, 
            step_size=40,
            gamma=0.9
        )
        # Critic: 使用StepLR
        self.critic_scheduler = StepLR(
            self.critic_optimizer, 
            step_size=40,
            gamma=0.9
        )
        
        self.waiting_prob_history = []
        self._waiting_probs_buffer = []

    def select_action(self, obs):
        task_features = obs['task_features']
        num_valid_tasks = obs['num_valid_tasks']
        
        if num_valid_tasks == 0 or num_valid_tasks is None:
            return None, None
        
        # Ensure data is on the correct device
        task_features = to_device(task_features, self.device)
        
        with torch.no_grad():
            action_probs = self.actor(task_features)
            
            if action_probs.numel() > 0:
                self._waiting_probs_buffer.append(float(action_probs[-1].item()))
            
            # 确保概率分布有效
            if torch.any(torch.isnan(action_probs)) or torch.any(torch.isinf(action_probs)):
                action_probs = torch.ones_like(action_probs) / len(action_probs)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 动作解释：索引映射到具体动作
            action_idx = action.item()
            if action_idx >= num_valid_tasks:
                # 选择了"不执行"动作
                return None, log_prob.item()
            else:
                # 选择了具体任务
                return action_idx, log_prob.item()

