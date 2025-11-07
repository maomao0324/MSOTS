import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor网络
class Actor(nn.Module):
    def __init__(self, task_feature_dim, hidden_dim=128):
        super(Actor, self).__init__()
        
        self.task_feature_dim = task_feature_dim
        
        # 任务特征编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # "不执行任务"动作
        self.no_action_embedding = nn.Parameter(torch.randn(task_feature_dim))
        
        # 3层神经网络 
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, task_features):
        """
        输入：task_features
        输出：action_probs: [num_tasks + 1] 每个任务+不执行的动作概率分布
        """
        num_tasks = task_features.size(0)
        
        if num_tasks > 0:
            # 编码任务特征
            task_encoded = self.task_encoder(task_features)  # [num_tasks, hidden_dim]
            
            # 通过3层网络得到任务logits
            x = self.activation(self.fc1(task_encoded))
            x = self.dropout(x)
            x = self.activation(self.fc2(x))
            x = self.dropout(x)
            task_logits = self.fc3(x).squeeze(-1)  # [num_tasks]
            
            # 不执行任务
            no_action_encoded = self.task_encoder(self.no_action_embedding.unsqueeze(0))  # [1, hidden_dim]
            
           
            x_no = self.activation(self.fc1(no_action_encoded))
            x_no = self.dropout(x_no)
            x_no = self.activation(self.fc2(x_no))
            x_no = self.dropout(x_no)
            no_action_logit = self.fc3(x_no).squeeze(-1)  # [1]
            
            # 合并所有动作的logits
            all_logits = torch.cat([task_logits, no_action_logit])  # [num_tasks + 1]
            action_probs = F.softmax(all_logits, dim=0)
            
            return action_probs
        else:
            # 如果没有任务，返回只有"不执行"概率为1的分布
            return torch.tensor([1.0], device=task_features.device)

#Critic网络
class FCCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FCCritic, self).__init__()
        
        # 3层神经网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 3层网络
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        
        return value.squeeze(-1)

