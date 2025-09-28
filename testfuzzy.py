import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyWeightAdapter:
    def __init__(self):
        self.sim = self.create_fuzzy_system()

    def create_fuzzy_system(self):
        # 输入变量
        delta_error = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'delta_error')
        error_rate = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'error_rate')
        # 输出变量
        weight_factor = ctrl.Consequent(np.arange(0.5, 2.0, 0.01), 'weight_factor')

        # 隶属度函数
        delta_error['small'] = fuzz.trimf(delta_error.universe, [-1, -1, 0])
        delta_error['medium'] = fuzz.trimf(delta_error.universe, [-1, 0, 1])
        delta_error['large'] = fuzz.trimf(delta_error.universe, [0, 1, 1])

        error_rate['slow'] = fuzz.trimf(error_rate.universe, [-10, -10, 0])
        error_rate['medium'] = fuzz.trimf(error_rate.universe, [-10, 0, 10])
        error_rate['fast'] = fuzz.trimf(error_rate.universe, [0, 10, 10])

        weight_factor['low'] = fuzz.trimf(weight_factor.universe, [0.5, 0.5, 1.0])
        weight_factor['medium'] = fuzz.trimf(weight_factor.universe, [0.8, 1.2, 1.6])
        weight_factor['high'] = fuzz.trimf(weight_factor.universe, [1.2, 2.0, 2.0])

        # 模糊规则
        rules = [
            ctrl.Rule(delta_error['small'] & error_rate['slow'], weight_factor['low']),
            ctrl.Rule(delta_error['small'] & error_rate['fast'], weight_factor['medium']),
            ctrl.Rule(delta_error['large'] & error_rate['slow'], weight_factor['medium']),
            ctrl.Rule(delta_error['large'] & error_rate['fast'], weight_factor['high'])
        ]

        fis = ctrl.ControlSystem(rules) # 对这些规则的结果取最大值
        return ctrl.ControlSystemSimulation(fis)

    def calculate_weight_factor(self, error, error_rate):
        self.sim.input['delta_error'] = error
        self.sim.input['error_rate'] = error_rate
        self.sim.compute() # 去模糊化
        return self.sim.output['weight_factor']


def calculate_fuzzy_weights(state, prev_state, dt, base_weights, adapter):
    Δf_gen, Δf_wind, Δf_ess, ΔP_tie = state
    prev_Δf_gen, prev_Δf_wind, prev_Δf_ess, prev_ΔP_tie = prev_state

    # 变化率
    rate_gen = (Δf_gen - prev_Δf_gen) / dt
    rate_wind = (Δf_wind - prev_Δf_wind) / dt
    rate_ess = (Δf_ess - prev_Δf_ess) / dt
    rate_tie = (ΔP_tie - prev_ΔP_tie) / dt

    # 计算各分量因子
    factor_gen = adapter.calculate_weight_factor(Δf_gen, rate_gen)
    factor_wind = adapter.calculate_weight_factor(Δf_wind, rate_wind)
    factor_ess = adapter.calculate_weight_factor(Δf_ess, rate_ess)
    factor_tie = adapter.calculate_weight_factor(ΔP_tie, rate_tie)

    factors = np.array([factor_gen, factor_wind, factor_ess, factor_tie])

    # 应用权重因子
    dynamic_weights = base_weights * factors

    # 归一化保持和 base_weights 一致
    total_base = np.sum(base_weights)
    dynamic_weights = dynamic_weights * total_base / np.sum(dynamic_weights)

    return dynamic_weights


# ==========================
# 测试部分
# ==========================
if __name__ == "__main__":
    adapter = FuzzyWeightAdapter()

    # 初始状态和前一状态
    state = [0.2, -0.12, 0.2, 0.1]  # Δf_gen, Δf_wind, Δf_ess, ΔP_tie
    prev_state = [0, 0, 0, 0]  # 上一时刻状态
    dt = 0.1
    base_weights = np.array([0.4, 0.2, 0.3, 0.1])

    dyn_w = calculate_fuzzy_weights(state, prev_state, dt, base_weights, adapter)
    print("Base weights   =", base_weights)
    print("Dynamic weights=", dyn_w)
