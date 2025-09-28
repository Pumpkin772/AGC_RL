import matlab.engine
import math, random
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
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
from algo.sac import SAC
from algo.masac import MASAC
from algo.utils import setup_seed
from algo.utils_env import obtain_action, pm_converter, state_reward_CHD

rootdir = 'D:/PycharmProjects/Python_Matlab_SAC_Kundur_wind_4VSG_Test_Communication'
filename=os.path.basename(__file__).split(".")[0]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = rootdir+ '/logs/' + filename + '_'+current_time # logs for tensorboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--withdelay', default=False, action="store_true")
    parser.add_argument('--with_all_communication', default=False, action="store_true")
    args = parser.parse_args()

    # model_name = 'uncontrol_v2'
    model_name = 'sac_idp_v2'
    if args.withdelay:
        model_name = model_name + '_with_delay'
    model_name = model_name + 'ep1500'
    if args.with_all_communication:
        model_name = model_name + 'all_communication'
    save_name_data = rootdir + '/test/loss/' + model_name + '_data_test' + '.csv'
    data_pd = pd.read_csv(save_name_data)

    save_name_loss = rootdir + '/test/loss/' + model_name + '_loss_test' + '.csv'
    loss_pd_all = pd.read_csv(save_name_loss)

    record_id = [1,4, 16,20,24,40,46]
    data_plot = pd.DataFrame(columns=data_pd.columns)
    loss_print = pd.DataFrame(columns=loss_pd_all.columns)
    counter = 0
    for num in record_id:

        left_ep = num
        right_ep = num+1
        left_time = left_ep * 11  # s
        right_time = right_ep * 11  #
        data_tmp = data_pd[(data_pd.iloc[:, 0] < right_time) & (data_pd.iloc[:, 0] > left_time)]
        data_tmp = data_tmp.copy()
        data_tmp.iloc[:,0] = data_tmp.iloc[:,0] - left_ep*11 +counter*11
        counter += 1
        data_plot = pd.concat([data_plot,data_tmp], axis=0)

        loss_tmp = loss_pd_all.iloc[left_ep:right_ep, :]
        loss_print = pd.concat([loss_print, loss_tmp], axis=0)


    data_plot = data_plot.copy()
    loss_print = loss_print.copy()


    time_plot = data_plot.iloc[:, 0]
    power_plot = data_plot.iloc[:, 1:5]
    freq_plot = data_plot.iloc[:, [7, 10, 13, 16]]
    deltaHavg_plot = data_plot.iloc[:, 29:33]
    deltaDavg_plot = data_plot.iloc[:, 33:37]

    loss_pd = loss_print.copy()
    print(loss_pd)
    plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Control reward ')
    plt.subplot(3, 2, 1)
    plt.plot(loss_pd.iloc[:,1],loss_pd.iloc[:,2], label='Total reward')
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(loss_pd.iloc[:, 1], loss_pd.iloc[:, 3], label='Reward F')
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(loss_pd.iloc[:, 1], loss_pd.iloc[:, 4], label='Reward H')
    plt.legend()
    plt.subplot(3, 2, 4)
    plt.plot(loss_pd.iloc[:, 1], loss_pd.iloc[:, 5], label='Reward D')
    plt.legend()
    plt.subplot(3, 2, 5)
    plt.plot(loss_pd.iloc[:, 1], loss_pd.iloc[:, 6], label='Reward H1')
    plt.legend()
    plt.subplot(3, 2, 6)
    plt.plot(loss_pd.iloc[:, 1], loss_pd.iloc[:, 7], label='Reward D1')
    plt.legend()
    plt.tight_layout()


    plt.figure(figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Real time control performance')
    plt.subplot(4, 1, 1)
    plt.plot(time_plot, power_plot)
    plt.legend(['p1', 'p2', 'p3', 'p4'])
    plt.subplot(4, 1, 2)
    plt.plot(time_plot, freq_plot)
    plt.legend(['f1', 'f2', 'f3', 'f4'])
    plt.subplot(4, 1, 3)
    plt.plot(time_plot, deltaHavg_plot)
    plt.legend(['deltaHavg1', 'deltaHavg2', 'deltaHavg3', 'deltaHavg4'])
    plt.subplot(4, 1, 4)
    plt.plot(time_plot, deltaDavg_plot)
    plt.legend(['deltaDavg1', 'deltaDavg2', 'deltaDavg3', 'deltaDavg4'])
    plt.tight_layout()
    plt.show()
