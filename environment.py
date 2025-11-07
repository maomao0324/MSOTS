import numpy as np
import torch
import math
from utils import device

class Satellite:
    def __init__(self, satellite_id):
        self.satellite_id = satellite_id
        self.is_idle = True
        self.current_task = None
        self.valid_tasks = []
        self.cooldown = 0
        
        self.decided_time_windows = set()
        self.window_decisions = {}
        self.window_extra_info = {}
        self.current_executing_window = None
        
    def can_observe(self, current_time, required_duration):
        """检查卫星是否可以执行新的观察"""
        if self.cooldown > 0:
            return False
        return True
    
    def record_observation(self, current_time, duration):
        """记录一次观察"""
        self.cooldown = 20
    
    def get_undecided_time_windows(self, current_time, valid_tasks):
        """获取当前时间点需要决策的时间窗"""
        need_decision_tasks = []
        
        for task in valid_tasks:
            if self.satellite_id in task.details:
                for tw in task.details[self.satellite_id]['time_windows']:
                    if tw[0] <= current_time < tw[1]:
                        window_key = (task.index, tw[0], tw[1])
                        if window_key not in self.decided_time_windows:
                            need_decision_tasks.append(task)
                            break
        
        return need_decision_tasks
    
    def make_time_window_decision(self, task, current_time, decision):
        """为特定任务的时间窗做决策"""
        if self.satellite_id in task.details:
            for tw in task.details[self.satellite_id]['time_windows']:
                if tw[0] <= current_time < tw[1]:
                    window_key = (task.index, tw[0], tw[1])
                    self.decided_time_windows.add(window_key)
                    self.window_decisions[window_key] = decision
                    has_future = any(tw2[0] >= tw[1] for tw2 in task.details[self.satellite_id]['time_windows'])
                    self.window_extra_info[window_key] = {"has_future": bool(has_future)}
                    break
    
    def has_committed_task_in_window(self, current_time):
        for window_key, decision in self.window_decisions.items():
            task_id, start_time, end_time = window_key
            if start_time <= current_time < end_time and decision is not None:
                return decision
        return None
        
    def reset_for_new_episode(self):
        """重置卫星状态，为新回合准备"""
        self.is_idle = True
        self.current_task = None
        self.valid_tasks = []
        self.cooldown = 0
        self.decided_time_windows = set()
        self.window_decisions = {}
        self.window_extra_info = {}
        self.current_executing_window = None

class SimpleRewardTracker:
    def __init__(self):
        self.reward_history = {
            'completion': [],
            'uniformity': [],
            'timeliness': [],
            'final': []
        }
        
    def record_reward(self, reward_type, value):
        """记录奖励值用于统计"""
        if reward_type in self.reward_history:
            self.reward_history[reward_type].append(value)
    
    def get_stats(self, reward_type):
        """获取奖励统计信息"""
        if reward_type in self.reward_history and self.reward_history[reward_type]:
            values = self.reward_history[reward_type]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        return None

class SatelliteSchedulingEnv:
    def __init__(self, tasks, satellites, global_start_time, max_time):
        self.tasks = tasks
        self.satellites = satellites
        self.global_start_time = global_start_time
        self.max_time = max_time
        self.num_satellites = len(satellites)
        self.max_observation_history = 4
        self.max_time_windows_per_day = 10
        self.satellite_specific_dim = 7
        self.task_feature_dim = 4 + self.max_observation_history + 2 * self.max_time_windows_per_day + self.satellite_specific_dim
      
        self.max_weight = max(task.weight for task in self.tasks) if self.tasks else 1
        self.time_step = 0
        
        self.satellite_task_dict = {sat_id: {} for sat_id in satellites.keys()}
        for task in self.tasks:
            for sat_id, info in task.details.items():
                for tw in info['time_windows']:
                    time_key = (tw[0], tw[1])
                    if time_key not in self.satellite_task_dict[sat_id]:
                        self.satellite_task_dict[sat_id][time_key] = []
                    self.satellite_task_dict[sat_id][time_key].append(task)
        self.satellite_time_keys = {}
        for sat_id in self.satellite_task_dict:
            self.satellite_time_keys[sat_id] = sorted(self.satellite_task_dict[sat_id].keys())
        self.time_step_length = 20
        self.time_window_size = 5 * 60 // self.time_step_length
        
        self.episode_reward_components = []
        self.last_reward_components = {}
        
        self.global_decided_task_windows = set()
        self.current_time_task_decisions = set()
        
        self.missed_window_penalty_n = 7

    def reset(self):
        """重置环境状态"""
        for sat in self.satellites.values():
            sat.reset_for_new_episode()
        
        for task in self.tasks:
            task.current_observation_time = 0
            task.completed = False
            task.locked = False
            task.current_observation_counts = 0
            task.observation_timestamps = []
            task.observation_satellite_ids = []
            
        self.time_step = 0
        
        self.episode_reward_components = []
        self.last_reward_components = {}
        
        self.global_decided_task_windows = set()
        self.current_time_task_decisions = set()
        
        return self._get_observations()

    def step(self, actions):
        """执行一步环境交互"""
        if not isinstance(actions, list):
            actions = list(actions)
        
        if len(actions) < self.num_satellites:
            actions.extend([None] * (self.num_satellites - len(actions)))
            
        for sat in self.satellites.values():
            if sat.is_idle and sat.cooldown > 0:
                sat.cooldown = max(0, sat.cooldown - self.time_step_length)

        rewards = [0.0 for _ in range(self.num_satellites)]
        done = False
        current_time = self.global_start_time + self.time_step * self.time_step_length
        tasks_completed = 0

        satellite_list = list(self.satellites.values())
        
        for idx, action in enumerate(actions):
            if idx >= len(satellite_list):
                break
                
            sat = satellite_list[idx]
            
            if sat.is_idle and sat.cooldown > 0:
                continue

            valid_tasks = self._get_valid_tasks_for_satellite(sat)
            sat.valid_tasks = valid_tasks
            
            undecided_tasks = sat.get_undecided_time_windows(current_time, valid_tasks)
            needs_decision = len(undecided_tasks) > 0
            
            if needs_decision:
                candidate_tasks = []
                
                for task in undecided_tasks:
                    current_window = None
                    for tw_start, tw_end in task.details[sat.satellite_id]['time_windows']:
                        if tw_start <= current_time < tw_end:
                            current_window = (tw_start, tw_end)
                            break
                    
                    if current_window is None:
                        continue
                    
                    window_key = (task.index, current_window)
                    if window_key not in self.global_decided_task_windows:
                        time_key = (task.index, current_time)
                        if time_key not in self.current_time_task_decisions:
                            candidate_tasks.append(task)
                        else:
                            continue
                    else:
                        continue
                
                needs_decision = len(candidate_tasks) > 0
            
            if sat.current_task is not None:
                task = sat.current_task

                if not self._is_task_executable(task, sat):
                    task.locked = False
                    task.current_observation_time = 0
                    sat.current_task = None
                    sat.is_idle = True
                    continue

                if hasattr(sat, 'current_executing_window') and sat.current_executing_window is not None:
                    exec_start, exec_end = sat.current_executing_window
                    if not (exec_start <= current_time < exec_end):
                        try:
                            window_key_cancel = (task.index, exec_start, exec_end)
                            if window_key_cancel in sat.window_decisions:
                                sat.window_decisions[window_key_cancel] = None
                        except Exception:
                            pass
                        task.current_observation_time = 0
                        sat.current_task = None
                        sat.is_idle = True
                        task.locked = False
                        sat.current_executing_window = None
                        continue
                task.current_observation_time += self.time_step_length

                if task.current_observation_time >= task.required_observation_time:
                    task.current_observation_counts += 1
                    task.observation_timestamps.append(current_time)
                    task.observation_satellite_ids.append(sat.satellite_id)

                    if task.current_observation_counts == 1:
                        if task.earliest_start_time is not None:
                            deadline = task.earliest_start_time + 3 * 3600
                            task.timeliness_satisfied = current_time <= deadline

                    obs_window = None
                    for tw_start, tw_end in task.details[sat.satellite_id]['time_windows']:
                        if tw_start <= current_time < tw_end:
                            obs_window = (tw_start, tw_end)
                            break
                    
                    if obs_window:
                        window_key = (task.index, obs_window)
                        self.global_decided_task_windows.add(window_key)
                        time_key = (task.index, current_time)
                        self.current_time_task_decisions.add(time_key)
                        try:
                            window_key_local = (task.index, obs_window[0], obs_window[1])
                            if window_key_local in sat.window_decisions:
                                sat.window_decisions[window_key_local] = None
                        except Exception:
                            pass

                    task.current_observation_time = 0
                    sat.current_task = None
                    sat.is_idle = True
                    task.locked = False
                    sat.current_executing_window = None
                    sat.record_observation(current_time, task.required_observation_time)

                    if task.current_observation_counts >= task.observation_counts:
                        task.completed = True
                        tasks_completed += 1

                        reward = self._calculate_reward(
                            task=task,
                            task_completed=True,
                            current_time=current_time,
                            completed_tasks=tasks_completed,
                            satellite=sat,
                            is_intermediate=False
                        )
                        rewards[idx] += reward

            else:
                if needs_decision:
                    if action is None or not isinstance(action, int) or action < 0 or action >= len(candidate_tasks):
                        for task in candidate_tasks:
                            sat.make_time_window_decision(task, current_time, None)
                            current_window = None
                            for tw_start, tw_end in task.details[sat.satellite_id]['time_windows']:
                                if tw_start <= current_time < tw_end:
                                    current_window = (tw_start, tw_end)
                                    break
                            if current_window:
                                window_key = (task.index, current_window)
                                self.global_decided_task_windows.add(window_key)
                                time_key = (task.index, current_time)
                                self.current_time_task_decisions.add(time_key)
                    else:
                        for i, task in enumerate(candidate_tasks):
                            current_window = None
                            for tw_start, tw_end in task.details[sat.satellite_id]['time_windows']:
                                if tw_start <= current_time < tw_end:
                                    current_window = (tw_start, tw_end)
                                    break
                            
                            if i == action:
                                sat.make_time_window_decision(task, current_time, task.index)
                                if current_window:
                                    window_key = (task.index, current_window)
                                    self.global_decided_task_windows.add(window_key)
                                    time_key = (task.index, current_time)
                                    self.current_time_task_decisions.add(time_key)
                                if self._is_task_executable(task, sat):
                                    if current_window:
                                        window_key = (task.index, current_window)
                                        if window_key in self.global_decided_task_windows:
                                            continue
                                        sat.current_executing_window = current_window
                                    
                                    sat.current_task = task
                                    sat.is_idle = False
                                    task.locked = True
                            else:
                                sat.make_time_window_decision(task, current_time, None)
                                if current_window:
                                    window_key = (task.index, current_window)
                                    self.global_decided_task_windows.add(window_key)
                                    time_key = (task.index, current_time)
                                    self.current_time_task_decisions.add(time_key)
                else:
                    committed_task_idx = sat.has_committed_task_in_window(current_time)
                    if committed_task_idx is not None and sat.is_idle:
                        for task in valid_tasks:
                            if task.index == committed_task_idx:
                                if self._is_task_executable(task, sat):
                                    sat.current_task = task
                                    sat.is_idle = False
                                    task.locked = True
                                break

        if self.time_step >= self.max_time / self.time_step_length:
            done = True

        all_tasks_completed = all(task.completed for task in self.tasks)
        if all_tasks_completed:
            done = True

        completion_ratio = sum(1 for task in self.tasks if task.completed) / len(self.tasks)

        self.time_step += 1
        observations = self._get_observations()

        info = {
            'completed_tasks': tasks_completed,
            'total_tasks': len(self.tasks),
            'completion_ratio': completion_ratio,
            'timeliness_satisfied': sum(1 for task in self.tasks if hasattr(task, 'timeliness_satisfied') and task.timeliness_satisfied),
            'timeliness_eligible': sum(1 for task in self.tasks if task.earliest_start_time is not None)
        }
        return observations, rewards, done, info

    def _get_valid_tasks_for_satellite(self, satellite):
        """获取卫星的有效任务列表"""
        current_time = self.global_start_time + self.time_step * self.time_step_length
        
        if satellite.is_idle and satellite.cooldown > 0:
            return []
            
        valid_tasks = []
        sat_id = satellite.satellite_id
        time_window_end = current_time + self.time_window_size * self.time_step_length
        time_keys = self.satellite_time_keys.get(sat_id, [])
            
        for time_key in time_keys:
            if time_key[1] < current_time:
                continue
                
            if time_key[0] > time_window_end:
                break
                
            for task in self.satellite_task_dict[sat_id][time_key]:
                if task.completed or (task.locked and satellite.current_task != task):
                    continue
                    
                for tw in task.details[sat_id]['time_windows']:
                    time_start, time_end = tw
                    if time_start <= current_time < time_end:
                        required_duration = task.required_observation_time - task.current_observation_time
                        remaining_window_time = time_end - current_time
                        if remaining_window_time >= required_duration:
                            valid_tasks.append(task)
                            break
        
        return valid_tasks

    def _is_task_executable(self, task, satellite):
        """检查任务是否可由卫星执行"""
        current_time = self.global_start_time + self.time_step * self.time_step_length
        
        if (task.completed or 
            (task.locked and satellite.current_task != task) or 
            satellite.satellite_id not in task.details):
            return False
            
        required_duration = task.required_observation_time - task.current_observation_time
        if not satellite.can_observe(current_time, required_duration):
            return False
            
        for tw in task.details[satellite.satellite_id]['time_windows']:
            time_start, time_end = tw
            if (time_start <= current_time < time_end and 
                (time_end - current_time) >= required_duration):
                return True
                
        return False

    def _get_observations(self):
        """观测获取方法"""
        observations = []
        current_time = self.global_start_time + self.time_step * self.time_step_length
        
        for sat in self.satellites.values():
            obs = {}
            valid_tasks = self._get_valid_tasks_for_satellite(sat)
            
            undecided_tasks = sat.get_undecided_time_windows(current_time, valid_tasks)
            needs_decision = len(undecided_tasks) > 0
            if needs_decision:
                candidate_tasks = []
                for task in undecided_tasks:
                    current_window = None
                    for tw_start, tw_end in task.details[sat.satellite_id]['time_windows']:
                        if tw_start <= current_time < tw_end:
                            current_window = (tw_start, tw_end)
                            break
                    
                    if current_window is not None:
                        window_key = (task.index, current_window)
                        if window_key not in self.global_decided_task_windows:
                            time_key = (task.index, current_time)
                            if time_key not in self.current_time_task_decisions:
                                candidate_tasks.append(task)
                
                needs_decision = len(candidate_tasks) > 0
            if not needs_decision:
                obs['task_features'] = torch.zeros((0, self.task_feature_dim), dtype=torch.float32, device=device)
                obs['num_valid_tasks'] = None
                observations.append(obs)
                continue
            
            if not needs_decision:
                obs['task_features'] = torch.zeros((0, self.task_feature_dim), dtype=torch.float32, device=device)
                obs['num_valid_tasks'] = None
                observations.append(obs)
                continue
                
            decision_tasks = candidate_tasks
            
            task_features = []
            for task in decision_tasks:
                task_weight_normalized = task.weight / self.max_weight
                
                remaining_obs = task.observation_counts - task.current_observation_counts
                remaining_obs_normalized = remaining_obs / task.observation_counts
                
                current_timestep = self.time_step
                if task.earliest_start_time is not None:
                    earliest_timestep = (task.earliest_start_time - self.global_start_time) // self.time_step_length
                    time_diff_steps = current_timestep - earliest_timestep
                    time_diff_normalized = min(1.0, time_diff_steps / 540.0)
                else:
                    time_diff_normalized = 0.0
                
                first_obs_completed = 1.0 if task.current_observation_counts > 0 else 0.0
                
                obs_history = []
                if task.observation_timestamps:
                    obs_timesteps = []
                    for timestamp in task.observation_timestamps:
                        timestep = (timestamp - self.global_start_time) // self.time_step_length
                        timestep_normalized = timestep / (self.max_time // self.time_step_length)
                        obs_timesteps.append(timestep_normalized)
                    
                    recent_obs = obs_timesteps[-self.max_observation_history:]
                    obs_history = recent_obs + [0.0] * (self.max_observation_history - len(recent_obs))
                else:
                    obs_history = [0.0] * self.max_observation_history
                
                window_visit_vector = [0.0] * self.max_time_windows_per_day
                window_history_vector = [0.0] * self.max_time_windows_per_day
                sat_id = sat.satellite_id
                if sat_id in task.details:
                    all_windows = sorted(task.details[sat_id]['time_windows'], key=lambda w: w[0])
                    trimmed = all_windows[:self.max_time_windows_per_day]
                    for idx, (tw_start, tw_end) in enumerate(trimmed):
                        if tw_end <= current_time:
                            window_history_vector[idx] = 1.0
                        if tw_end > current_time:
                            window_visit_vector[idx] = 1.0
                
                Ws = 0
                Wall = 0
                if hasattr(task, 'details'):
                    for sid, info in task.details.items():
                        for tw in info['time_windows']:
                            if tw[1] > current_time:
                                if sid == sat.satellite_id:
                                    Ws += 1
                                Wall += 1
                Ws_ratio = min(1.0, Ws / 10.0)
                Wall_ratio = min(1.0, Wall / 50.0)
                sigma = 1.0 - (Ws / max(1, Wall))
                R = max(0, task.observation_counts - task.current_observation_counts)
                phi = Wall / max(1, R)
                phi_norm = min(1.0, phi / 10.0)
                cooldown_norm = min(1.0, max(0, sat.cooldown) / 100.0)
                busy_flag = 0.0 if sat.is_idle else 1.0
                workload_norm = 1.0 - float(sat.is_idle)

                satellite_specific = [Ws_ratio, Wall_ratio, sigma, phi_norm, cooldown_norm, busy_flag, workload_norm]

                task_feature = [
                    task_weight_normalized,
                    remaining_obs_normalized,
                    time_diff_normalized,
                    first_obs_completed
                ] + obs_history + window_visit_vector + window_history_vector + satellite_specific

                task_features.append(task_feature)
            
            if not task_features:
                task_features = np.zeros((0, self.task_feature_dim), dtype=np.float32)
            else:
                task_features = np.array(task_features, dtype=np.float32)
            
            obs['task_features'] = torch.as_tensor(task_features, dtype=torch.float32, device=device)
            obs['num_valid_tasks'] = len(decision_tasks)
            
            observations.append(obs)
            
        return observations

    def get_global_state_features(self):
        """提取全局状态特征作为Critic输入"""
        current_time = self.global_start_time + self.time_step * self.time_step_length
        
        completed_tasks = sum(1 for task in self.tasks if task.completed)
        total_tasks = len(self.tasks)
        completion_rate = completed_tasks / max(1, total_tasks)
        
        uncompleted_tasks = [task for task in self.tasks if not task.completed]
        timeliness_satisfied_count = 0
        timeliness_eligible_count = 0
        
        for task in uncompleted_tasks:
            if task.earliest_start_time is not None:
                timeliness_eligible_count += 1
                if hasattr(task, 'timeliness_satisfied') and task.timeliness_satisfied:
                    timeliness_satisfied_count += 1
        
        timeliness_satisfaction_rate = timeliness_satisfied_count / max(1, timeliness_eligible_count)
        
        multi_obs_tasks = [task for task in self.tasks if task.observation_counts >= 2 and not task.completed]
        uniformity_satisfied_count = 0
        
        for task in multi_obs_tasks:
            if len(task.observation_timestamps) >= 2:
                sorted_timestamps = sorted(task.observation_timestamps)
                intervals = []
                for i in range(len(sorted_timestamps) - 1):
                    intervals.append(sorted_timestamps[i+1] - sorted_timestamps[i])
                
                if intervals:
                    first_obs_time = sorted_timestamps[0]
                    first_obs_relative = first_obs_time - self.global_start_time
                    remaining_time = self.max_time - first_obs_relative
                    remaining_observations = task.observation_counts - 1
                    
                    if remaining_observations > 0:
                        ideal_interval = remaining_time / remaining_observations
                    else:
                        ideal_interval = self.max_time / task.observation_counts
                    
                    if ideal_interval > 0:
                        variance = sum((interval - ideal_interval) ** 2 for interval in intervals) / len(intervals)
                        if variance < (0.2 * ideal_interval) ** 2:
                            uniformity_satisfied_count += 1
        
        uniformity_satisfaction_rate = uniformity_satisfied_count / max(1, len(multi_obs_tasks))
        
        idle_satellites = sum(1 for sat in self.satellites.values() if sat.is_idle and sat.cooldown == 0)
        idle_rate = idle_satellites / max(1, len(self.satellites))
        
        working_satellites = sum(1 for sat in self.satellites.values() if not sat.is_idle)
        working_rate = working_satellites / max(1, len(self.satellites))
        
        cooling_satellites = sum(1 for sat in self.satellites.values() if sat.cooldown > 0)
        cooling_rate = cooling_satellites / max(1, len(self.satellites))
        
        satellite_workloads = {}
        for sat_id in self.satellites.keys():
            satellite_workloads[sat_id] = 0
        
        for task in self.tasks:
            for timestamp in task.observation_timestamps:
                for sat_id in task.details.keys():
                    satellite_workloads[sat_id] += 1 / len(task.details)
                    break
        
        workloads = list(satellite_workloads.values())
        if len(workloads) > 1:
            mean_workload = sum(workloads) / len(workloads)
            workload_variance = sum((w - mean_workload) ** 2 for w in workloads) / len(workloads)
            max_possible_variance = mean_workload ** 2
            load_balance = 1.0 - min(workload_variance / max(max_possible_variance, 1e-6), 1.0)
        else:
            load_balance = 1.0
        
        total_remaining_windows = 0
        total_possible_windows = 0
        
        for task in self.tasks:
            if not task.completed:
                for sat_id, info in task.details.items():
                    total_possible_windows += len(info['time_windows'])
                    for tw in info['time_windows']:
                        if tw[1] > current_time:
                            total_remaining_windows += 1
        
        window_availability = total_remaining_windows / max(1, total_possible_windows)
        
        multi_obs_uncompleted = [task for task in self.tasks 
                                if task.observation_counts >= 2 and not task.completed]
        
        if multi_obs_uncompleted:
            remaining_windows_times = []
            for task in multi_obs_uncompleted:
                for sat_id, info in task.details.items():
                    for tw in info['time_windows']:
                        if tw[1] > current_time:
                            window_start_relative = (tw[0] - self.global_start_time) % 86400
                            remaining_windows_times.append(window_start_relative)
            
            if remaining_windows_times:
                hour_coverage = set()
                for window_time in remaining_windows_times:
                    hour = int(window_time // 3600) % 24
                    hour_coverage.add(hour)
                time_coverage = len(hour_coverage) / 24.0
            else:
                time_coverage = 0.0
        else:
            time_coverage = 1.0
        
        remaining_time = self.max_time - (current_time - self.global_start_time)
        if remaining_time > 0:
            window_density = total_remaining_windows / (remaining_time / 3600.0)
            window_density_normalized = min(window_density / 10.0, 1.0)
        else:
            window_density_normalized = 0.0
        
        global_features = np.array([
            completion_rate,
            timeliness_satisfaction_rate,
            uniformity_satisfaction_rate,
            idle_rate,
            working_rate,
            cooling_rate,
            load_balance,
            window_availability,
            time_coverage,
            window_density_normalized
        ], dtype=np.float32)
        
        return global_features

    def _calculate_reward(self, task, task_completed, current_time, completed_tasks, satellite=None, is_intermediate=False):
        """奖励函数"""
        
        revisit_reward = 0.0
        revisit_reward_first = 0.0
        revisit_reward_later = 0.0
        
        if task.current_observation_counts == 1:
            if task.earliest_start_time is not None:
                tau = 3 * 3600
                dt = max(0.0, current_time - task.earliest_start_time)
                x = min(1.0, dt / tau)
                base = (1.0 - x) ** 2
                revisit_reward_first = 6.0 * base
            else:
                revisit_reward_first = 2.5
                
        elif task.current_observation_counts >= 2:
            if len(task.observation_timestamps) >= 2:
                sorted_timestamps = sorted(task.observation_timestamps)
                first_obs_time = sorted_timestamps[0]
                first_obs_relative = first_obs_time - self.global_start_time
                remaining_time_of_day = self.max_time - first_obs_relative
                remaining_observations = max(task.observation_counts - 1, 1)
                ideal_interval = remaining_time_of_day / remaining_observations if remaining_observations > 0 else 0.0

                last_interval = task.observation_timestamps[-1] - task.observation_timestamps[-2]
                if ideal_interval > 0:
                    delta = abs(last_interval - ideal_interval) / ideal_interval
                    tau = 0.2
                    align = math.exp(-delta / max(tau, 1e-6))
                    revisit_reward_later = 6.0 * align
                else:
                    revisit_reward_later = 1.0
            else:
                revisit_reward_later = 1.0
        else:
            revisit_reward_first = 0.0
            revisit_reward_later = 0.0

        revisit_reward = revisit_reward_first + revisit_reward_later
        
        completion_reward = 0.0
        if task_completed:
            base_completion = 7.0
            weight_bonus = 3.0 * (task.weight / self.max_weight)
            completion_reward = base_completion + weight_bonus
        
        interval_penalty = 0.0
        try:
            if task.current_observation_counts >= 2:
                sorted_timestamps = sorted(task.observation_timestamps)
                last_interval_seconds = sorted_timestamps[-1] - sorted_timestamps[-2]
                delta_ts_hours = last_interval_seconds / 3600.0
                
                first_obs_time = sorted_timestamps[0]
                first_obs_rel = first_obs_time - self.global_start_time
                remaining_time = self.max_time - first_obs_rel
                remaining_obs = max(task.observation_counts - 1, 1)
                delta_tc_hours = (remaining_time / remaining_obs) / 3600.0
                
                if delta_tc_hours > 0:
                    deviation_ratio = abs(delta_ts_hours - delta_tc_hours) / delta_tc_hours
                    interval_penalty = -math.exp(deviation_ratio) * float(self.missed_window_penalty_n)
        except Exception:
            interval_penalty = 0.0

        raw_final_reward = revisit_reward + completion_reward + interval_penalty
        final_reward = raw_final_reward
        
        reward_components = {
            'revisit_reward': revisit_reward,
            'revisit_reward_first': revisit_reward_first,
            'revisit_reward_later': revisit_reward_later,
            'completion_reward': completion_reward,
            'interval_penalty': interval_penalty,
            'final_reward': raw_final_reward,
            'scaled_final_reward': final_reward,
            'is_intermediate': is_intermediate,
            'observation_count': task.current_observation_counts,
            'task_completed': task_completed
        }
        
        self.episode_reward_components.append(reward_components.copy())
        self.last_reward_components = reward_components
        
        return final_reward

    def calculate_avg_timeliness_score(self):
        """计算平均时效性得分"""
        timeliness_satisfied_count = sum(1 for task in self.tasks
                                       if hasattr(task, 'timeliness_satisfied') and task.timeliness_satisfied)
        timeliness_eligible_count = sum(1 for task in self.tasks
                                      if task.earliest_start_time is not None)
        return timeliness_satisfied_count / max(1, timeliness_eligible_count)

    def calculate_avg_uniformity_mse(self):
        """计算平均均匀性均方误差"""
        mse_values = []
        
        multi_observation_tasks = [task for task in self.tasks if task.observation_counts >= 2]
        
        if not multi_observation_tasks:
            return 0.0
        
        for task in multi_observation_tasks:
            if len(task.observation_timestamps) >= 2:
                sorted_timestamps = sorted(task.observation_timestamps)
                first_obs_time = sorted_timestamps[0]
                
                first_obs_relative = first_obs_time - self.global_start_time
                remaining_time_of_day = self.max_time - first_obs_relative
                
                remaining_observations = task.observation_counts - 1
                if remaining_observations > 0:
                    ideal_interval = remaining_time_of_day / remaining_observations
                else:
                    ideal_interval = self.max_time / task.observation_counts
                
                actual_intervals = []
                
                for i in range(len(sorted_timestamps) - 1):
                    interval = sorted_timestamps[i+1] - sorted_timestamps[i]
                    actual_intervals.append(interval)
                
                if actual_intervals and ideal_interval > 0:
                    mse = sum((interval - ideal_interval) ** 2 for interval in actual_intervals) / len(actual_intervals)
                    normalized_mse = mse / (ideal_interval ** 2)
                    mse_values.append(normalized_mse)
                else:
                    mse_values.append(1.0)
            else:
                mse_values.append(1.0)
        
        return sum(mse_values) / len(mse_values) if mse_values else 0.0

