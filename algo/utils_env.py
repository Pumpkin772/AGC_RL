# !/usr/bin/env python 3 torch
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Python_Matlab_SAC -> utils_env
@IDE    ：PyCharm
@Author ：Ymumu
@Date   ：2021/12/5 10:20
@ Des   ：utilis for env converter
=================================================='''
import numpy as np
import math

# for the maddqn obtain action

def obtain_action(num, dim):
    # act is 0,1,2...51*51
    stp = round(200 / dim)
    droop = np.array(range(-99, 99, stp)) / 100
    inertia = np.array(range(-99, 99, stp)) / 100
    # obtain the index for droop
    droop_index = int(num % dim)
    inertia_index = int((num - droop_index) / dim)
    act1 = droop[droop_index]
    act2 = inertia[inertia_index]
    act = np.hstack((act1, act2))
    return act

# the python action to matlab parameters
def pm_converter1(actions_step_num):
    actions_step_num = np.hstack((actions_step_num[0],actions_step_num[1])) #  将actions_step_num数组中每隔2个元素取一个，并按顺序排列
    act_str = '[' #  定义一个空字符串，用于存储转换后的字符串
    for ele in actions_step_num:
        str_i1 = format(ele, '.4f') #  将每个元素转换为小数点后4位的字符串
        act_str = act_str + str_i1 + ' ' #  将转换后的字符串添加到act_str中
    act_str = act_str[:-1] #  去掉act_str最后一个空格
    act_str = act_str + ']' #  在act_str末尾添加']'
    return act_str #  返回转换后的字符串

def pm_converter2(actions_step_num):
    actions_step_num = np.hstack((actions_step_num[0],actions_step_num[1],actions_step_num[2])) #  将actions_step_num数组中每隔2个元素取一个，并按顺序排列
    act_str = '[' #  定义一个空字符串，用于存储转换后的字符串
    for ele in actions_step_num:
        str_i1 = format(ele, '.4f') #  将每个元素转换为小数点后4位的字符串
        act_str = act_str + str_i1 + ' ' #  将转换后的字符串添加到act_str中
    act_str = act_str[:-1] #  去掉act_str最后一个空格
    act_str = act_str + ']' #  在act_str末尾添加']'
    return act_str #  返回转换后的字符串

# calculate the local reward H、D观测器
def state_reward_DHD(data, agent_no, step, step_max, args, env_index, actions_list):
    disconnect = math.floor(env_index[2] * 17)
    wF, wH, wD = args.wF, args.wH, args.wD # the weights for reward
    time_left = np.max([1-float(step/step_max),0])
    variables = np.array(data)[-1, :]  # dim = 36
    inertia = variables[56:64]
    droop = variables[64:]
    power = variables[0:8]
    freq = [variables[10], variables[13], variables[16], variables[19], variables[22], variables[25], variables[28], variables[31]]
    end_flag_F = 1 if step <= step_max else 0.0
    end_flag_D = 1 if step <= step_max else 0.0
    end_flag_H = 1 if step <= step_max else 0.0      # the last 10 H parameters will be punished
    # end_flag_F = step/step_max if step <= step_max else 0.0
    # end_flag_D = 1.0 if step>=step_max - 10 and step <= step_max else 0.0
    # end_flag_H = 1.0 if step>=step_max - 10 and step <= step_max else 0.0 # the last 10 H parameters will be punished

    states, rewards = [],[]
    rewardsF, rewardsH, rewardsD = [],[],[]
    for agent_id in range(agent_no):
        power = variables[agent_id]  # 0
        deltaF = variables[agent_id * 3 + 8: agent_id * 3 + 11]  # 8,9,10
        deltaFT = variables[agent_id * 3 + 32: agent_id * 3 + 35]  # 32,33,34
        state = np.hstack((power, deltaF, deltaFT))
        states.append(state)
        # if disconnect != 16:
        #     disconnect_T = disconnect % 8 + 1
        #     disconnect_R = math.floor(disconnect/8) + 1
        #     disconnect_node = (disconnect_T + 2*disconnect_R-3)%8
        #     disconnect_node = 8 if disconnect_node == 0.0 else disconnect_node
        #     if agent_id + 1 == disconnect_node:
        #         if disconnect_R == 1:
        #             deltaF = [deltaF[1], deltaF[2]]
        #             deltaFT = [deltaFT[1], deltaFT[2]]
        #         else:
        #             deltaF = [deltaF[0], deltaF[2]]
        #             deltaFT = [deltaFT[0], deltaFT[2]]

        # calculate the reward for each agent
        # reward_F = -1*(freq[agent_id]-np.mean(freq))**2 * end_flag_F  # 0810-1034 a negtive value
        reward_F = -1 * (freq[agent_id] - np.mean(freq)) ** 2 * end_flag_F
        # reward_H = -1*inertia[agent_id]**2 * end_flag_H
        # reward_D = -1*droop[agent_id]**2 * end_flag_D
        reward_H = -1*actions_list[2*agent_id]/800*inertia[agent_id] * end_flag_H
        reward_D = -1*actions_list[2*agent_id+1]/800*droop[agent_id] * end_flag_D
        reward = wF*reward_F + wH*reward_H + wD*reward_D
        rewards.append(reward)
        rewardsF.append(reward_F)
        rewardsH.append(reward_H)
        rewardsD.append(reward_D)

    return states, rewards, rewardsF, rewardsH, rewardsD

# calculate the local reward 集中的H、D参数
def state_reward_CHD(data, agent_no, step, step_max, args, env_index,actions_list):
    disconnect = math.floor(env_index[2] * 4) + 1
    wF, wH, wD = args.wF, args.wH, args.wD # the weights for reward
    time_left = np.max([1-float(step/step_max),0])
    variables = np.array(data)[-1, :]  # dim = 36
    inertia = variables[28:32]
    droop = variables[32:]
    power = variables[0:4]
    freq = [variables[6], variables[9], variables[12], variables[15]]
    end_flag_F = 1 if step <= step_max else 0.0
    end_flag_D = 1 if step <= step_max else 0.0
    end_flag_H = 1 if step <= step_max else 0.0 # the last 10 H parameters will be punished
    # end_flag_F = step/step_max if step <= step_max else 0.0
    # end_flag_D = 1.0 if step>=step_max - 10 and step <= step_max else 0.0
    # end_flag_H = 1.0 if step>=step_max - 10 and step <= step_max else 0.0 # the last 10 H parameters will be punished

    states, rewards = [],[]
    rewardsF, rewardsH, rewardsD = [],[],[]
    for agent_id in range(agent_no):
        power = variables[agent_id]  # 0
        deltaF = variables[agent_id * 3 + 4: agent_id * 3 + 7]  # 4,5,6
        deltaFT = variables[agent_id * 3 + 16: agent_id * 3 + 19]  # 16,17,18
        state = np.hstack((power, deltaF, deltaFT))
        states.append(state)
        if agent_id + 1 == disconnect:
            deltaF = [deltaF[1], deltaF[2]]
            deltaFT = [deltaFT[1], deltaFT[2]]
        elif agent_id == disconnect % 4:
            deltaF = [deltaF[0], deltaF[2]]
            deltaFT = [deltaFT[0], deltaFT[2]]

        # calculate the reward for each agent
        reward_F = -1*np.sum((deltaF-np.mean(deltaF))**2) * end_flag_F  # a negtive value
        reward_H = -1*np.sum(actions_list[-1][0:8:2]/400)**2 * end_flag_H
        reward_D = -1*np.sum(actions_list[-1][1:8:2]/400)**2 * end_flag_D
        reward = wF*reward_F + wH*reward_H + wD*reward_D
        rewards.append(reward)
        rewardsF.append(reward_F)
        rewardsH.append(reward_H)
        rewardsD.append(reward_D)

    return states, rewards, rewardsF, rewardsH, rewardsD

# calculate the local reward 集中控制
def state_reward_CC(data, agent_no, step, step_max, args, env_index,actions_list):
    disconnect = math.floor(env_index[2] * 17)
    wF, wH, wD = args.wF, args.wH, args.wD # the weights for reward
    time_left = np.max([1-float(step/step_max),0])
    variables = np.array(data)[-1, :]  # dim = 36
    inertia = variables[56:64]
    droop = variables[64:]
    power = variables[0:8]
    freq =  [variables[10], variables[13], variables[16], variables[19], variables[22], variables[25], variables[28], variables[31]]
    freqT = [variables[34], variables[37], variables[40], variables[43], variables[46], variables[49], variables[52], variables[55]]
    end_flag_F = 1 if step <= step_max else 0.0
    end_flag_D = 1 if step <= step_max else 0.0
    end_flag_H = 1 if step <= step_max else 0.0  # the last 10 H parameters will be punished
    # end_flag_F = step/step_max if step <= step_max else 0.0
    # end_flag_D = 1.0 if step>=step_max - 10 and step <= step_max else 0.0
    # end_flag_H = 1.0 if step>=step_max - 10 and step <= step_max else 0.0 # the last 10 H parameters will be punished

    states, rewards = [], []
    rewardsF, rewardsH, rewardsD = [], [], []
    state = np.hstack((power, freq, freqT))
    states.append(state)
    reward_F = -1 * np.sum((freq - np.mean(freq)) ** 2) * end_flag_F  # a negtive value
    reward_H = -1*(np.sum(actions_list[0:16:2]/800))**2 * end_flag_H
    reward_D = -1*(np.sum(actions_list[1:16:2]/800))**2 * end_flag_D
    reward = wF*reward_F + wH*reward_H + wD*reward_D
    rewards.append(reward)
    rewardsF.append(reward_F)
    rewardsH.append(reward_H)
    rewardsD.append(reward_D)

    return states, rewards, rewardsF, rewardsH, rewardsD

def output_constraint(action_step_last):
    action_h_sum = np.sum(action_step_last[0:8:2])
    action_d_sum = np.sum(action_step_last[1:8:2])
    action_step_last[0:8:2] = action_step_last[0:8:2] - action_h_sum / 4.0 * np.ones(4)
    action_step_last[1:8:2] = action_step_last[1:8:2] - action_d_sum / 4.0 * np.ones(4)
    if np.max(action_step_last[0:8:2]) > 298:
        scale = np.max(action_step_last[0:8:2]) / 298.0
        action_step_last[0:8:2] = action_step_last[0:8:2] / scale
    if np.min(action_step_last[0:8:2]) < -98:
        scale = np.min(action_step_last[0:8:2]) / (-98.0)
        action_step_last[0:8:2] = action_step_last[0:8:2] / scale
    if np.max(action_step_last[1:8:2]) > 596.0:
        scale = np.max(action_step_last[1:8:2]) / 596.0
        action_step_last[1:8:2] = action_step_last[1:8:2] / scale
    if np.min(action_step_last[1:8:2]) < -196.0:
        scale = np.min(action_step_last[1:8:2]) / (-196.0)
        action_step_last[1:8:2] = action_step_last[1:8:2] / scale
    return action_step_last

def state_reward(data, agent_no, step, step_max, args, env_index,actions_list):
    variables = np.array(data)[-1, :]  
    deltafarea1 = variables[0]
    deltafarea2 = variables[1]
    Ptie = variables[2]
    wb = 50
    beta = -21
    ace1 = deltafarea1/wb*beta + Ptie
    ace2 = deltafarea2/wb*beta + Ptie
    state1 = np.hstack((deltafarea1, Ptie, ace1))
    state2 = np.hstack((deltafarea2, Ptie, ace2))
    rewarda1 = -ace1**2
    rewarda2 = -ace2**2
    rewardf1 = -deltafarea1**2
    rewardf2 = -deltafarea2**2
    reward1 = rewarda1 + rewardf1
    reward2 = rewarda2 + rewardf2
    states, rewards, rewardsA, rewardsF = [],[],[],[]
    states.append(state1)
    states.append(state2)
    rewards.append(reward1)
    rewards.append(reward2)
    rewardsA.append(rewarda1)
    rewardsA.append(rewarda2)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    return states, rewards, rewardsA, rewardsF
    

def state_reward2(data, agent_no, step, step_max, args, env_index,actions_list):
    variables = np.array(data)[-1, :]  
    deltafarea1 = variables[0]
    deltafarea2 = variables[1]
    Ptie = variables[2]
    wb = 50
    alpha = 1
    beta = -10
    ace1 = deltafarea1/wb*beta + Ptie
    ace2 = deltafarea2/wb*beta + Ptie
    state1 = np.hstack((deltafarea1, Ptie))
    state2 = np.hstack((deltafarea2, Ptie))
    rewarda1 = -(Ptie*alpha)**2
    rewarda2 = -(Ptie*alpha)**2
    rewardf1 = -(deltafarea1/wb*beta)**2
    rewardf2 = -(deltafarea2/wb*beta)**2
    reward1 = rewarda1 + rewardf1
    reward2 = rewarda2 + rewardf2
    states, rewards, rewardsA, rewardsF = [],[],[],[]
    states.append(state1)
    states.append(state2)
    rewards.append(reward1)
    rewards.append(reward2)
    rewardsA.append(rewarda1)
    rewardsA.append(rewarda2)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    return states, rewards, rewardsA, rewardsF

def state_reward3(data, agent_no, step, step_max, args, env_index,actions_list):
    # M4B11
    variables = np.array(data)[-1, :]  
    deltaf1 = variables[0]
    deltafw1 = variables[1]
    deltafes1 = variables[2]
    deltaf2 = variables[3]
    deltafw2 = variables[4]
    deltafes2 = variables[5]
    Ptie = variables[6]
    wb = 50
    alpha = 5
    beta = -500
    # ace1 = deltafarea1/wb*beta + Ptie
    # ace2 = deltafarea2/wb*beta + Ptie
    state1 = np.hstack((deltaf1,deltafw1,deltafes1, Ptie))
    state2 = np.hstack((deltaf2,deltafw2,deltafes2, Ptie))
    rewardp1 = -(Ptie*alpha)**2
    rewardp2 = -(Ptie*alpha)**2
    rewardf1 = -((deltaf1+deltafw1+deltafes1)/wb*beta)**2
    rewardf2 = -((deltaf2+deltafw2+deltafes2)/wb*beta)**2
    reward1 = rewardp1 + rewardf1
    reward2 = rewardp2 + rewardf2
    states, rewards, rewardsP, rewardsF = [],[],[],[]
    states.append(state1)
    states.append(state2)
    rewards.append(reward1)
    rewards.append(reward2)
    rewardsP.append(rewardp1)
    rewardsP.append(rewardp2)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    return states, rewards, rewardsP, rewardsF

def state_reward3_fuzzy(data, agent_no, step, step_max, args, env_index,actions_list,dynamic_weights_list):
    # M4B11
    variables = np.array(data)[-1, :]
    deltaf1 = variables[0]
    deltafw1 = variables[1]
    deltafes1 = variables[2]
    deltaf2 = variables[3]
    deltafw2 = variables[4]
    deltafes2 = variables[5]
    Ptie = variables[6]
    wb = 50
    alpha = 5
    beta = -500
    # ace1 = deltafarea1/wb*beta + Ptie
    # ace2 = deltafarea2/wb*beta + Ptie
    dynamic_weights1 = dynamic_weights_list[0]
    dynamic_weights2 = dynamic_weights_list[1]
    state1 = np.hstack((deltaf1,deltafw1,deltafes1, Ptie))
    state2 = np.hstack((deltaf2,deltafw2,deltafes2, Ptie))
    rewardp1 = -(Ptie*alpha*dynamic_weights1[3])**2
    rewardp2 = -(Ptie*alpha)**2
    rewardf1 = -((deltaf1*dynamic_weights1[0]+deltafw1*dynamic_weights1[1]+deltafes1*dynamic_weights1[2])/wb*beta)**2
    rewardf2 = -((deltaf2*dynamic_weights2[0]+deltafw2*dynamic_weights2[1]+deltafes2*dynamic_weights2[2])/wb*beta)**2
    reward1 = rewardp1 + rewardf1
    reward2 = rewardp2 + rewardf2
    states, rewards, rewardsP, rewardsF = [],[],[],[]
    states.append(state1)
    states.append(state2)
    rewards.append(reward1)
    rewards.append(reward2)
    rewardsP.append(rewardp1)
    rewardsP.append(rewardp2)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    return states, rewards, rewardsP, rewardsF

def state_reward_test(data, agent_no, step, step_max, args, env_index,actions_list):
    variables = np.array(data)[-1, :]  
    deltaf1 = variables[0]
    deltafw1 = variables[1]
    deltafes1 = variables[2]
    deltaf2 = variables[3]
    deltafw2 = variables[4]
    deltafes2 = variables[5]
    Ptie = variables[6]
    wb = 50
    alpha = 5
    beta = -500
    # ace1 = deltafarea1/wb*beta + Ptie
    # ace2 = deltafarea2/wb*beta + Ptie
    state1 = np.hstack((deltaf1,deltafw1,deltafes1, Ptie))
    state2 = np.hstack((deltaf2,deltafw2,deltafes2, Ptie))
    rewardp1 = -(Ptie*alpha)**2
    rewardp2 = -(Ptie*alpha)**2
    rewardf1 = -((deltaf1+deltafw1+deltafes1)/wb*beta)**2
    rewardf2 = -((deltaf2+deltafw2+deltafes2)/wb*beta)**2
    reward1 = rewardp1 + rewardf1
    reward2 = rewardp2 + rewardf2
    states, rewards, rewardsP, rewardsF, P_list = [],[],[],[],[]
    states.append(state1)
    states.append(state2)
    rewards.append(reward1)
    rewards.append(reward2)
    rewardsP.append(rewardp1)
    rewardsP.append(rewardp2)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    for i in range(7,16):
        P_list.append(variables[i])
    return states, rewards, rewardsP, rewardsF, P_list

def state_reward_AGC(data, agent_no, step, step_max, args, env_index,actions_list):
    # M10B39
    variables = np.array(data)[-1, :]  
    deltaf1 = variables[0]
    deltafes1 = variables[1]
    deltaf2 = variables[2]
    deltafes2 = variables[3]
    deltafw2 = variables[4]
    deltaf3 = variables[5]
    deltafes3 = variables[6]
    deltafw3 = variables[7]
    Ptie12 = variables[8]
    Ptie23 = variables[9]
    Ptie31 = variables[10]
    wb = 50
    alpha = 1
    beta = 5
    # ace1 = deltafarea1/wb*beta + Ptie
    # ace2 = deltafarea2/wb*beta + Ptie
    state1 = np.hstack((deltaf1,0,deltafes1, Ptie12, Ptie31))
    state2 = np.hstack((deltaf2,deltafw2,deltafes2, Ptie12, Ptie23))
    state3 = np.hstack((deltaf3,deltafw3,deltafes3, Ptie31, Ptie23))
    rewardp1 = -((Ptie12*alpha)**2+(Ptie31*alpha)**2)
    rewardp2 = -((Ptie12*alpha)**2+(Ptie23*alpha)**2)
    rewardp3 = -((Ptie31*alpha)**2+(Ptie23*alpha)**2)
    rewardf1 = -(deltaf1/wb*beta)**2-(deltafes1/wb*beta)**2
    rewardf2 = -(deltaf2/wb*beta)**2-(deltafes2/wb*beta)**2-(deltafw2/wb*beta)**2
    rewardf3 = -(deltaf3/wb*beta)**2-(deltafes3/wb*beta)**2-(deltafw3/wb*beta)**2
    reward1 = rewardp1 + rewardf1
    reward2 = rewardp2 + rewardf2
    reward3 = rewardp3 + rewardf3
    states, rewards, rewardsP, rewardsF = [],[],[],[]
    states.append(state1)
    states.append(state2)
    states.append(state3)
    rewards.append(reward1)
    rewards.append(reward2)
    rewards.append(reward3)
    rewardsP.append(rewardp1)
    rewardsP.append(rewardp2)
    rewardsP.append(rewardp3)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    rewardsF.append(rewardf3)
    return states, rewards, rewardsP, rewardsF

def state_reward_AGC_V2(data, agent_no, step, step_max, args, env_index,actions_list):
    # M10B39_V2
    variables = np.array(data)[-1, :]  
    deltaf1 = variables[0]
    deltafes1 = variables[1]
    deltafw1 = variables[2]
    deltaf2 = variables[3]
    deltafes2 = variables[4]
    deltafw2 = variables[5]
    deltafes3 = variables[6]
    deltafw3 = variables[7]
    Ptie12 = variables[8]
    Ptie23 = variables[9]
    Ptie31 = variables[10]
    clock = variables[11]
    wb = 50
    alpha = 1
    beta = 200
    # ace1 = deltafarea1/wb*beta + Ptie
    # ace2 = deltafarea2/wb*beta + Ptie
    state1 = np.hstack((deltaf1,deltafw1,deltafes1, Ptie12, Ptie31))
    state2 = np.hstack((deltaf2,deltafw2,deltafes2, Ptie12, Ptie23))
    state3 = np.hstack((0,deltafw3,deltafes3, Ptie31, Ptie23))
    rewardp1 = -((Ptie12*alpha)**2+(Ptie31*alpha)**2)
    rewardp2 = -((Ptie12*alpha)**2+(Ptie23*alpha)**2)
    rewardp3 = -((Ptie31*alpha)**2+(Ptie23*alpha)**2)
    rewardf1 = -(deltaf1/wb*beta)**2-(deltafes1/wb*beta)**2
    rewardf2 = -(deltaf2/wb*beta)**2-(deltafes2/wb*beta)**2-(deltafw2/wb*beta)**2
    rewardf3 = -(deltafes3/wb*beta)**2-(deltafw3/wb*beta)**2
    reward1 = rewardp1 + rewardf1
    reward2 = rewardp2 + rewardf2
    reward3 = rewardp3 + rewardf3
    states, rewards, rewardsP, rewardsF = [],[],[],[]
    states.append(state1)
    states.append(state2)
    states.append(state3)
    rewards.append(reward1)
    rewards.append(reward2)
    rewards.append(reward3)
    rewardsP.append(rewardp1)
    rewardsP.append(rewardp2)
    rewardsP.append(rewardp3)
    rewardsF.append(rewardf1)
    rewardsF.append(rewardf2)
    rewardsF.append(rewardf3)
    return states, rewards, rewardsP, rewardsF, clock