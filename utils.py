import numpy as np
import os
import torch
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 全局设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将数据移动到指定设备的工具函数
def to_device(x, device):
    """将数据移动到指定设备（CPU/GPU）"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, (int, float)):
        return torch.tensor(x, device=device)
    elif isinstance(x, list):
        return [to_device(item, device) for item in x]
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x

# 数据合并函数
def merge_csv_files(tw_file_path, meshedtarget_file_path, output_file_path):
    """合并时间窗口和目标数据文件"""
    tw_df = pd.read_csv(tw_file_path, low_memory=False)
    meshedtarget_df = pd.read_csv(meshedtarget_file_path, low_memory=False)
    tw_df.rename(columns={'h3Id': 'H3ID'}, inplace=True)
    merged_df = pd.merge(tw_df, meshedtarget_df, on='H3ID')
    merged_df = merged_df[['H3ID', 'sat_id', 'sensor_id', 'time_start', 'time_end', 'duration',
                          'WEIGHT', 'REVISIT', 'PAYLOADTYPE']]
    merged_df.rename(columns={
        'H3ID': 'h3Id',
        'duration': 'time_window_duration',
        'WEIGHT': 'weight',
        'REVISIT': 'revisit',
        'PAYLOADTYPE': 'payload'
    }, inplace=True)
    merged_df['required_observation_time'] = 20
    global_start_time = merged_df['time_start'].min()
    merged_df['global_start_time'] = global_start_time
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    merged_df.to_csv(output_file_path, index=False)
    return global_start_time, merged_df

