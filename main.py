import os
import numpy as np
import torch
from utils import merge_csv_files, set_seed
from task import Task
from environment import Satellite, SatelliteSchedulingEnv
from agent import Agent
from training import train_mappo

def main():
    set_seed(42)
    
    tw_file_path = 'tw.csv'
    meshedtarget_file_path = 'meshedtarget.csv'
    output_file_path = 'data/merged_tasks.csv'
    
    global_start_time, merged_df = merge_csv_files(tw_file_path, meshedtarget_file_path, output_file_path)
    
    sat_ids = merged_df['sat_id'].unique()
    num_satellites_to_keep = max(1, int(len(sat_ids) * 1))
    selected_sat_ids = np.random.choice(sat_ids, size=num_satellites_to_keep, replace=False)
    
    satellites = {}
    for sat_id in selected_sat_ids:
        satellites[sat_id] = Satellite(satellite_id=sat_id)
    
    tasks = []
    tasks_dict = {}
    for index, row in merged_df.iterrows():
        h3Id = row['h3Id']
        sat_id = row['sat_id']
        
        if sat_id not in satellites:
            continue
            
        sensor_id = row['sensor_id']
        time_start = row['time_start']
        time_end = row['time_end']
        time_window = (time_start, time_end)
        time_window_duration = row['time_window_duration']
        required_observation_time = row['required_observation_time']
        weight = row['weight']
        revisit = row['revisit']
        payload = row['payload']
        
        if h3Id not in tasks_dict:
            task = Task(index, h3Id, required_observation_time, weight, revisit, payload)
            task.add_satellite_info(sat_id, sensor_id, time_window, time_window_duration)
            tasks_dict[h3Id] = task
        else:
            task = tasks_dict[h3Id]
            task.add_satellite_info(sat_id, sensor_id, time_window, time_window_duration)
            
    tasks = list(tasks_dict.values())
    for task in tasks:
        task.compute_time_info()
    
    max_time = 86400
    
    env = SatelliteSchedulingEnv(tasks, satellites, global_start_time, max_time)
    num_agents = len(satellites)
    
    num_episodes = 300
    batch_size = 128
    max_steps = 4320
    
    agents = []
    for i in range(num_agents):
        agent = Agent(
            task_feature_dim=env.task_feature_dim,
            agent_id=i,
            critic_input_dim=10
        )
        agents.append(agent)
    
    trained_agents, actor, critic = train_mappo(
        env=env,
        num_agents=num_agents,
        agents=agents,
        num_episodes=num_episodes,
        batch_size=batch_size,
        max_steps=max_steps
    )
    
    os.makedirs('models1', exist_ok=True)
    torch.save(actor.state_dict(), 'models1/mappo_actor_no_heuristic.pth')
    torch.save(critic.state_dict(), 'models1/mappo_critic_no_heuristic.pth')

if __name__ == "__main__":
    main()
