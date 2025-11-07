# 任务类
class Task:
    def __init__(self, index, h3Id, required_observation_time, weight, revisit, payload):
        self.index = index  # 任务索引
        self.h3Id = h3Id  # 任务ID
        self.required_observation_time = required_observation_time  # 所需观察时间
        self.weight = weight  # 任务权重
        self.observation_counts = revisit  # 需要观察的次数
        self.payload = payload  # 有效载荷类型
        self.details = {}  # 卫星详细信息
        self.locked = False 
        self.completed = False  # 是否完成
        self.current_observation_time = 0  # 当前观察时间
        self.current_observation_counts = 0  # 当前观察次数
        self.earliest_start_time = None  # 最早开始时间
        self.deadline = None  # 截止时间
        self.observation_timestamps = []  # 观察时间戳列表
        self.observation_satellite_ids = []  # 记录完成观测的卫星ID
        self.timeliness_satisfied = False  # 时效性满足状态

    def add_satellite_info(self, sat_id, sensor_id, time_window, time_window_duration):
        """添加卫星信息"""
        if sat_id not in self.details:
            self.details[sat_id] = {'sensor_ids': [], 'time_windows': [], 'time_window_durations': []}
        self.details[sat_id]['sensor_ids'].append(sensor_id)
        self.details[sat_id]['time_windows'].append(time_window)
        self.details[sat_id]['time_window_durations'].append(time_window_duration)

    def compute_time_info(self):
        """计算时间信息"""
        all_time_windows = []
        for info in self.details.values():
            all_time_windows.extend(info['time_windows'])
        if all_time_windows:
            self.earliest_start_time = min(tw[0] for tw in all_time_windows)
            self.deadline = self.earliest_start_time + 3 * 3600  # 3小时后截止
        else:
            self.earliest_start_time = None
            self.deadline = None

