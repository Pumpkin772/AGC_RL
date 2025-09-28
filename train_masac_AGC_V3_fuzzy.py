# !/usr/bin/env python 3 torch
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Python-Matlab -> train_ddqn
@IDE    ：PyCharm
@Author ：Ymumu
@Date   ：2021/12/3 20:37
@ Des   ：Train MASAC with MTDC
=================================================='''
import matlab.engine
import math, random
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random
import numpy as np
import time
import datetime
from tensorboardX import SummaryWriter
from copy import deepcopy
import os
import datetime
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from algo.sac import SAC
from algo.masac import MASAC
from algo.utils import setup_seed
from algo.utils_env import obtain_action, pm_converter1, state_reward, state_reward3_fuzzy

rootdir = 'D:\桌面\simulink实验'
filename = os.path.basename(__file__).split(".")[0]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = rootdir+ '/logs/' + filename + '_'+current_time # logs for tensorboard


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


def train(args):
    setup_seed(20)
    # define the type of env, flag1 and flag2
    train_num = 100
    test_num = 50
    train_env_set =  np.random.rand(train_num, 3)
    test_env_set = np.random.rand(test_num, 3)

    #model_name = 'sac_idp_v2_10seq_100beta' # states为包含10个时间步的序列, beta=100
    model_name = 'sac_idp_v3_3seq_5a_500b_Kunder_5dis_V3_fuzzy' # states为包含3个时间步的序列, beta=5
    save_reward = -2000
    model_path_best = rootdir + '/models/' + model_name + '/Best/'  # to save the actor by reinfoecement learning
    model_path_final= rootdir + '/models/' + model_name + '/Final/'  # to save the actor by reinfoecement learning
    if not os.path.exists(model_path_best):
        os.makedirs(model_path_best)
    if not os.path.exists(model_path_final):
        os.makedirs(model_path_final)
    if args.tensorboard:
        writer = SummaryWriter(logdir)
    # set the control agents
    agent_num = 2 # 两个区域
    state_dim = 12 # delta f , Ptie
    action_dim = 1 # delta P
    action_bound = 1 # action (-1,1)
    action_scale = 7
    # set the environment
    eng = matlab.engine.start_matlab()
    env_name = 'M4B11_modified_AGC_RL'
    eng.load_system(env_name)
    # set the agents
    agents =MASAC(agent_num, state_dim, action_dim, action_scale)
    if args.load_model:
        # load the model
        load_model_name =  'sac_idp_v1'
        load_ep = 200
        load_model_path_best = rootdir + '/models/' + load_model_name + '/Best/'  # to save the actor by reinfoecement learning
        load_model_path_final = rootdir + '/models/' + load_model_name + '/Final/'  # to save the actor by reinfoecement learning
        load_model_path_epoch = rootdir + '/models/' + load_model_name + '/epoch/' + str(load_ep) + '/'
        agents.load_model(load_model_path_final)
        print('policy net work is loaded!')
        print('===============Model is loaded=================')
    # set the train para
    num_training = 0
    batch_size = 256
    auto_entropy = True
    max_episodes = 1000 # 训练次数
    counter_env = 0
    counter_eps = 0
    loss_train = []
    # begin training
    for ep in range(max_episodes):
        action_step_last = np.zeros(2,)
        counter_step = 0
        t1 = time.time()
        # obtain the initial parameters for environment
        env_index = train_env_set[counter_env]
        counter_env += 1
        if counter_env == train_num:
            counter_env = 0
        flag_type, flag_value, flag_com = env_index[0], env_index[1],  env_index[2] # random seed for env
        eng.M4B11Env_init(float(flag_type), float(flag_value), float(flag_com), nargout=0)  # 三个参数为随机种子
        PauseTime = 1
        StopTime = 100.0
        #First_PauseTime = 10
        eng.set_param(env_name+'/Network1/pause_time', 'value', str(PauseTime), nargout=0)
        eng.set_param(env_name, 'StopTime', str(100.0), nargout=0)
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)

        states_list, rewards_list,actions_list, dones_list = [],[],[],[]
        state_input_seq_list = []
        actions_history = []
        rewardsA_list, rewardsF_list = [],[]
        done = 0.0

        actions_history.append(np.zeros(8))
        step_max= int(round(100.0/PauseTime))
        prev_state_input_seq = [[0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0],[0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0]]
        dyn_w = [[1,1,1,1],[1,1,1,1]]
        while True:
            ModelStatus = eng.get_param(env_name, 'SimulationStatus') # while true轮询，但仿真状态变为paused时暂停并进入if逻辑
            if ModelStatus == 'paused':
                counter_step += 1
                counter_eps += 1
                time_env = np.array(eng.eval('tout')).reshape(-1)
                data = eng.eval('logout.get(\'Sim2Python\').Values.Data')
                # 得到当前时刻的状态和上一时刻的奖励
                states, rewards, rewardsA, rewardsF = state_reward3_fuzzy(data, agent_num, counter_step,step_max,args,
                                                                            env_index,actions_history,dyn_w)
                states_list.append(states)
                rewards_list.append(rewards)
                rewardsA_list.append(rewardsA)
                rewardsF_list.append(rewardsF)
                if args.tensorboard:
                    writer.add_scalar('Reward_EP/rd', np.sum(rewards[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_1', np.sum(rewardsA[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_2', np.sum(rewardsF[0:step_max][:]), global_step=counter_eps)
                # obtain the action
                actions_step = []
                actions_save = []

                # 构造历史序列状态作为 agent 输入
                seq_length = 3
                if len(states_list) < seq_length:
                    # 如果长度不够，使用零填充或复制开头
                    padded = [states_list[0]] * (seq_length - len(states_list)) + states_list
                    state_seq = padded[-seq_length:]  # 保证是最近的3步
                else:
                    state_seq = states_list[-seq_length:]
                # 构造每个 agent 的历史状态序列
                state_input_seq = []

                for agent_id in range(agent_num):
                    # 提取该 agent 的每一步的 state，形成 [seq_length, state_dim]
                    agent_seq = [s[agent_id] for s in state_seq]
                    agent_seq_flat = np.concatenate(agent_seq)  # 展平成 4*3 = 12维
                    state_input_seq.append(agent_seq_flat)
                state_input_seq_list.append(state_input_seq)

                for agent_id in range(agent_num):
                    obs = state_input_seq[agent_id]
                    if args.uncontrol == False:
                        action = agents.model[agent_id].policy_net.get_action(obs, deterministic=False)

                    else:
                        action = np.ones(2)
                    actions_save.append(action)
                    # act_real = action*math.exp(-counter_step)
                    actions_step.extend(action)

                actions_list.append(actions_save)
                dones_list.append(done)
                dt = 1
                base_weights = np.array([1,1,1,1])
                dyn_w = calculate_enhanced_fuzzy_weights(state_input_seq, prev_state_input_seq, dt, base_weights)
                prev_state_input_seq = state_input_seq
                # update the state
                # actions_step_num = np.array(actions_step, dtype=float)
                # action_step_last = action_step_last * 0 + actions_step_num
                # for agent_id in range(agent_num):
                #     action_step_last[2*agent_id] = 200*action_step_last[2*agent_id] + 100
                #     action_step_last[2 * agent_id+1] = 400*action_step_last[2 * agent_id+1] + 200
                # for agent_id in range(agent_num):
                #     action_step_last[2*agent_id] = action_step_last[2*agent_id] \
                #         if action_step_last[2*agent_id] >= -0.49 else -0.49
                #     action_step_last[2 * agent_id + 1] = 0.0
                    # action_step_last[2*agent_id+1] = action_step_last[2*agent_id+1] \
                    #     if action_step_last[2*agent_id+1] >= -0.99 else -0.99

                actions_history.append(np.array(actions_step))
                actions_step_str = pm_converter1(actions_step)
                eng.set_param(env_name+'/Python2Sim1', 'value', actions_step_str, nargout=0)
                PauseTime += 1
                # print(
                #     'EP: {}/{} | Env time : {:.3f} | Running Time: {:.4f}'.format(counter_step, ep, time_env[-1],
                #                                                                   time.time() - t1))
                if (PauseTime + 1) > StopTime:
                    dones_list[-1] = 1.0
                    eng.set_param(env_name, 'SimulationCommand', 'Stop', nargout=0)
                    # update the reply buffer with state, action, reward done
                    if args.uncontrol == False:
                        len1 = len(states_list)

                        for i1 in range(step_max-2):
                            for agent_id in range(agent_num):
                                state = state_input_seq_list[i1][agent_id]
                                action = actions_list[i1][agent_id]
                                reward = rewards_list[i1+1][agent_id]
                                next_state = state_input_seq_list[i1+1][agent_id]
                                done = dones_list[i1+1]
                                # # 修改reward list 有关惯性系数的值
                                # interval = 2 # 3 10
                                # if i1>=step_max - 10 and i1<=step_max: # 对于最后的10个惯性系数
                                #     reward = reward + args.wH*np.sum(rewardsH_list_num[i1+2:i1+2+interval,agent_id])
                                #     + args.wD*np.sum(rewardsD_list_num[i1+2:i1+2+interval,agent_id])

                                agents.model[agent_id].replay_buffer.push(state, action, reward, next_state, done)# 10 去掉

                        buffer_length =  [len(agents.model[agent_id].replay_buffer) for agent_id in range(agent_num)]
                        if np.min(buffer_length) > batch_size:
                            for _ in range(50):
                                alpha_loss, q_value_loss1, q_value_loss2, policy_loss = agents.train(batch_size,
                                                                                                     reward_scale=1,
                                                                                                     auto_entropy=auto_entropy,
                                                                                                     target_entropy=-1. * action_dim)
                                if args.tensorboard:
                                    writer.add_scalar('Loss/Alpha_loss', alpha_loss, global_step=num_training)
                                    writer.add_scalar('Loss/Q1_loss', q_value_loss1, global_step=num_training)
                                    writer.add_scalar('Loss/Q2_loss', q_value_loss2, global_step=num_training)
                                    writer.add_scalar('Loss/pi_loss', policy_loss, global_step=num_training)
                                    num_training += 1

                    ep_rd = np.sum(rewards_list[0:step_max][:])
                    ep_rdA = np.sum(rewardsA_list[0:step_max][:])
                    ep_rdF = np.sum(rewardsF_list[0:step_max][:])
                    rewards_arr = np.array(rewards_list[:step_max])  # [step_max, 2]
                    ep_rd1 = np.sum(rewards_arr[:, 0])  # 累加 agent 1 的 reward
                    ep_rd2 = np.sum(rewards_arr[:, 1])  # 累加 agent 2 的 reward
                    rewardsA_arr = np.array(rewardsA_list[:step_max])  # [step_max, 2]
                    ep_rdA1 = np.sum(rewardsA_arr[:, 0])  # 累加 agent 1 的 reward
                    ep_rdA2 = np.sum(rewardsA_arr[:, 1])  # 累加 agent 2 的 reward
                    rewardsF_arr = np.array(rewardsF_list[:step_max])  # [step_max, 2]
                    ep_rdF1 = np.sum(rewardsF_arr[:, 0])  # 累加 agent 1 的 reward
                    ep_rdF2 = np.sum(rewardsF_arr[:, 1])  # 累加 agent 2 的 reward

                    time_now = datetime.datetime.now().strftime("%H%M%S.%f")
                    loss_train.append([float(time_now), ep, ep_rd1, ep_rdA1, ep_rdF1, ep_rd2, ep_rdA2, ep_rdF2])
                    print(
                        'Episode: {}/{} | Episode Reward: {:.4f} ({:.4f}, {:4f}) ({:.4f}, {:4f}) ({:.4f}, {:4f})| Running Time: {:.4f}'
                            .format(ep, max_episodes, ep_rd, ep_rdA, ep_rdF,ep_rdA1, ep_rdF1,ep_rdA2, ep_rdF2,time.time() - t1))
                    reward_ep = np.sum(rewards_list[0:step_max][:])
                    if reward_ep > save_reward and args.save_model:
                        agents.save_model(model_path_best)
                        save_reward = reward_ep
                        print('EP {}:Model saved with Test reward {:4f}'.format(ep, save_reward))


                    if args.tensorboard:
                        writer.add_scalar('Reward/train_rd1', ep_rd1, global_step=ep)
                        writer.add_scalar('Reward/train_rdA1', ep_rdA1, global_step=ep)
                        writer.add_scalar('Reward/train_rdF1', ep_rdF1, global_step=ep)
                        writer.add_scalar('Reward/train_rd2', ep_rd2, global_step=ep)
                        writer.add_scalar('Reward/train_rdA2', ep_rdA2, global_step=ep)
                        writer.add_scalar('Reward/train_rdF2', ep_rdF2, global_step=ep)

                    break # the simulink is stopped at 5s

                eng.set_param(env_name+'/Network1/pause_time', 'value', str(PauseTime), nargout=0)
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

            elif ModelStatus == 'stopped':  # 仿真有可能会不收敛
                print('仿真异常停止')
                break

        if ep%100==0 and ep!= 0 and args.save_model:
            model_path_pip_epoch = rootdir + '/models/' + model_name + '/epoch/' + str(ep) + '/'
            if not os.path.exists(model_path_pip_epoch):
                os.makedirs(model_path_pip_epoch)
            agents.save_model(model_path_pip_epoch)
            print('=============The model is saved at epoch {}============='.format(ep))

    if args.save_model:
        agents.save_model(model_path_final)
        print('=============The final model is saved!==========')

    print('===============Save loss in the data.csv=================')
    if args.save_data:
        loss_train = np.array(loss_train)
        pd_rl_loss_train = pd.DataFrame(loss_train, columns=['time', 'step','ep_rd1','ep_rdA1','ep_rdF1','ep_rd2','ep_rdA2','ep_rdF2'])
        data_save_path = rootdir + '/train/loss/'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name = data_save_path + model_name + '_train'  + current_time + '.csv'
        pd_rl_loss_train.to_csv(save_name, sep=',', header=True, index=False)

    eng.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--save_model', default=True, action="store_true")
    parser.add_argument('--save_data', default=True, action="store_true")
    parser.add_argument('--uncontrol', default=False, action="store_true")
    parser.add_argument('--load_model', default=False, action="store_true")
    parser.add_argument('--wF', default=10.0)  # 100
    parser.add_argument('--wH', default=0.1)  # 1
    parser.add_argument('--wD', default=0.1)  # 1
    args = parser.parse_args()
    train(args)
