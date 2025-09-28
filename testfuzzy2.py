import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class EnhancedFuzzyWeightAdapter:
    def __init__(self):
        # 创建更多的输入变量和输出变量
        self.error = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'error')
        self.rate = ctrl.Antecedent(np.arange(0, 10, 0.1), 'rate')
        self.acceleration = ctrl.Antecedent(np.arange(0, 50, 0.5), 'acceleration')
        self.integral = ctrl.Antecedent(np.arange(0, 5, 0.05), 'integral')
        self.std = ctrl.Antecedent(np.arange(0, 1, 0.01), 'std')

        self.weight_factor = ctrl.Consequent(np.arange(0.5, 2.0, 0.01), 'weight_factor')

        # 定义隶属度函数
        self.error.automf(3, names=['small', 'medium', 'large'])
        self.rate.automf(3, names=['slow', 'medium', 'fast'])
        self.acceleration.automf(3, names=['small', 'medium', 'large'])
        self.integral.automf(3, names=['small', 'medium', 'large'])
        self.std.automf(3, names=['low', 'medium', 'high'])
        self.weight_factor.automf(5, names=['very_low', 'low', 'medium', 'high', 'very_high'])

        # 定义更复杂的规则
        rules = [
            # 基础规则：偏差和变化率
            ctrl.Rule(self.error['small'] & self.rate['slow'], self.weight_factor['low']),
            ctrl.Rule(self.error['small'] & self.rate['fast'], self.weight_factor['medium']),
            ctrl.Rule(self.error['medium'] & self.rate['slow'], self.weight_factor['medium']),
            ctrl.Rule(self.error['medium'] & self.rate['fast'], self.weight_factor['high']),
            ctrl.Rule(self.error['large'] & self.rate['slow'], self.weight_factor['high']),
            ctrl.Rule(self.error['large'] & self.rate['fast'], self.weight_factor['very_high']),

            # 增强规则：考虑加速度
            ctrl.Rule(self.acceleration['large'], self.weight_factor['very_high']),
            ctrl.Rule(self.acceleration['medium'], self.weight_factor['high']),

            # 增强规则：考虑累积偏差
            ctrl.Rule(self.integral['large'], self.weight_factor['high']),

            # 增强规则：考虑波动性
            ctrl.Rule(self.std['high'], self.weight_factor['high'])
        ]

        self.control_system = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)

    def calculate_weight_factor(self, error, rate, acceleration, integral, std):
        self.simulator.input['error'] = error
        self.simulator.input['rate'] = rate
        self.simulator.input['acceleration'] = acceleration
        self.simulator.input['integral'] = integral
        self.simulator.input['std'] = std

        try:
            self.simulator.compute()
            return self.simulator.output['weight_factor']
        except:
            # 如果计算失败，返回默认值
            return 1.0


def calculate_enhanced_fuzzy_weights(state_input_seq, prev_state_input_seq,dt, base_weights):
    """
    使用增强的模糊逻辑计算动态权重
    state: 包含最近四个时间段的状态向量
    dt: 时间步长
    base_weights: 基础权重向量
    """
    # 提取当前值和历史值
    dynamic_weights_list = []
    for i in range(0,len(state_input_seq)):
        state = state_input_seq[i]
        prev_state = prev_state_input_seq[i]
        Δf_gen_t2, Δf_wind_t2, Δf_ess_t2, ΔP_tie_t2 = state[0:4]
        Δf_gen_t1, Δf_wind_t1, Δf_ess_t1, ΔP_tie_t1 = state[4:8]
        Δf_gen_t, Δf_wind_t, Δf_ess_t, ΔP_tie_t = state[8:12]
        Δf_gen_t3, Δf_wind_t3, Δf_ess_t3, ΔP_tie_t3 = prev_state[0:4]

        # 使用绝对值的偏差值
        abs_Δf_gen_t = abs(Δf_gen_t)
        abs_Δf_wind_t = abs(Δf_wind_t)
        abs_Δf_ess_t = abs(Δf_ess_t)
        abs_ΔP_tie_t = abs(ΔP_tie_t)

        # 计算变化率（使用加权平均法）并取绝对值
        rate_gen = abs((0.5 * (Δf_gen_t - Δf_gen_t1) + 0.3 * (Δf_gen_t1 - Δf_gen_t2) + 0.2 * (Δf_gen_t2 - Δf_gen_t3)) / dt)
        rate_wind = abs(
            (0.5 * (Δf_wind_t - Δf_wind_t1) + 0.3 * (Δf_wind_t1 - Δf_wind_t2) + 0.2 * (Δf_wind_t2 - Δf_wind_t3)) / dt)
        rate_ess = abs((0.5 * (Δf_ess_t - Δf_ess_t1) + 0.3 * (Δf_ess_t1 - Δf_ess_t2) + 0.2 * (Δf_ess_t2 - Δf_ess_t3)) / dt)
        rate_tie = abs((0.5 * (ΔP_tie_t - ΔP_tie_t1) + 0.3 * (ΔP_tie_t1 - ΔP_tie_t2) + 0.2 * (ΔP_tie_t2 - ΔP_tie_t3)) / dt)

        # 计算加速度（变化率的变化率）并取绝对值
        prev_rate_gen = abs((Δf_gen_t1 - Δf_gen_t2) / dt)
        prev_rate_wind = abs((Δf_wind_t1 - Δf_wind_t2) / dt)
        prev_rate_ess = abs((Δf_ess_t1 - Δf_ess_t2) / dt)
        prev_rate_tie = abs((ΔP_tie_t1 - ΔP_tie_t2) / dt)

        acc_gen = abs((rate_gen - prev_rate_gen) / dt) if dt > 0 else 0
        acc_wind = abs((rate_wind - prev_rate_wind) / dt) if dt > 0 else 0
        acc_ess = abs((rate_ess - prev_rate_ess) / dt) if dt > 0 else 0
        acc_tie = abs((rate_tie - prev_rate_tie) / dt) if dt > 0 else 0

        # 计算绝对值的累积偏差
        integral_gen = (abs_Δf_gen_t + abs(Δf_gen_t1) + abs(Δf_gen_t2)) * dt
        integral_wind = (abs_Δf_wind_t + abs(Δf_wind_t1) + abs(Δf_wind_t2)) * dt
        integral_ess = (abs_Δf_ess_t + abs(Δf_ess_t1) + abs(Δf_ess_t2)) * dt
        integral_tie = (abs_ΔP_tie_t + abs(ΔP_tie_t1) + abs(ΔP_tie_t2)) * dt

        # 计算绝对值的标准差
        std_gen = np.std([abs_Δf_gen_t, abs(Δf_gen_t1), abs(Δf_gen_t2)])
        std_wind = np.std([abs_Δf_wind_t, abs(Δf_wind_t1), abs(Δf_wind_t2)])
        std_ess = np.std([abs_Δf_ess_t, abs(Δf_ess_t1), abs(Δf_ess_t2)])
        std_tie = np.std([abs_ΔP_tie_t, abs(ΔP_tie_t1), abs(ΔP_tie_t2)])

        # 初始化增强的模糊适配器（使用绝对值版本）
        adapter = EnhancedFuzzyWeightAdapter()

        # 计算各分量的权重因子，使用绝对值输入
        factor_gen = adapter.calculate_weight_factor(abs_Δf_gen_t, rate_gen, acc_gen, integral_gen, std_gen)
        factor_wind = adapter.calculate_weight_factor(abs_Δf_wind_t, rate_wind, acc_wind, integral_wind, std_wind)
        factor_ess = adapter.calculate_weight_factor(abs_Δf_ess_t, rate_ess, acc_ess, integral_ess, std_ess)
        factor_tie = adapter.calculate_weight_factor(abs_ΔP_tie_t, rate_tie, acc_tie, integral_tie, std_tie)

        factors = np.array([factor_gen, factor_wind, factor_ess, factor_tie])

        # 应用权重因子
        dynamic_weights = base_weights * factors

        # 归一化
        total_base = np.sum(base_weights)
        dynamic_weights = dynamic_weights * total_base / np.sum(dynamic_weights)
        dynamic_weights_list.append(dynamic_weights)
    return dynamic_weights_list

# ==========================
# 测试部分
# ==========================
if __name__ == "__main__":

    # 初始状态和前一状态
    state = [[0, 0, 0, 0, 0, 0, 0, 0, 0.2, -0.12, 0.2, 0.1],[0, 0, 0, 0, 0, 0, 0, 0, 0.2, -0.12, 0.2, 0.1]]  # Δf_gen, Δf_wind, Δf_ess, ΔP_tie
    prev_state = [[0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0],[0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0]]  # 上一时刻状态
    dt = 1
    base_weights = np.array([1,1,1,1])

    dyn_w = calculate_enhanced_fuzzy_weights(state, prev_state, dt, base_weights)
    print("Base weights   =", base_weights)
    print("Dynamic weights=", dyn_w)

