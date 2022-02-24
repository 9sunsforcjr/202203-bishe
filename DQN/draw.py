import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

if __name__ == "__main__":
    # python parse.py -d ./Logs/expsample_DQN_agentLogs.txt
    # parser = argparse.ArgumentParser(description='Parse file and create plots')
    #
    # parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    # args = parser.parse_args()
    #
    # aux = args.d[0].split(".")

    # print(type(args.d[0]))
    dir='NSFNET-result.txt'

    differentiation_str = 'expsample_DQN_agentLogs2'

    list_score_test = []
    epsilon_decay = []
    list_losses = []

    # with open(dir) as fp:
    #     for line in fp:
    #         arrayLine = line.split(",")
    #         if arrayLine[0]==">":
    #             list_score_test.append(float(arrayLine[1]))
    #         elif arrayLine[0]=="-":
    #             epsilon_decay.append(float(arrayLine[1]))
    #         elif arrayLine[0]==".":
    #             list_losses.append(float(arrayLine[1]))
    #
    # model_id = -1
    # reward = 0
    # with open(dir) as fp:
    #     for line in reversed(list(fp)):
    #         arrayLine = line.split(":")
    #         if arrayLine[0]=='MAX REWD':
    #             model_id = arrayLine[2].split(",")[0]
    #             reward = arrayLine[1].split(" ")[1]
    #             break

    with open(dir, 'r') as f:
        while True:
            lines = f.readline()  # 整行读取数据
            if not lines:
                break

            p_tmp, E_tmp = [float(i) for i in lines.split(',')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。

            list_score_test.append(E_tmp)


    # print("Best model_id: "+model_id+" with Average Score Test of "+reward)
    # print(np.size(list_score_test))

    list_SP=13.0978375*np.ones(np.size(list_score_test))
    list_KSP=13.824662499999999*np.ones(np.size(list_score_test))


    plt.plot(list_SP, label="SP-FF",linestyle='--')
    plt.plot(list_KSP, label="KSP-FF",linestyle='-.')
    plt.plot(list_score_test, label="DQN")
    plt.xlabel("Episodes")
    plt.title("GNN+DQN Testing score")
    plt.ylabel("Average Score Test")
    plt.legend(loc="lower right")
    # plt.axhline(y=13.0978375, color='r', linestyle='-')
    plt.savefig("AvgTestScore_" + differentiation_str)
    plt.close()
    #
    # # Plot epsilon evolution
    # plt.plot(epsilon_decay)
    # plt.xlabel("Episodes")
    # plt.ylabel("Epsilon value")
    # plt.savefig("./Images/Epsilon_" + differentiation_str)
    # plt.close()
    #
    # # Plot Loss evolution
    # ysmoothed = savgol_filter(list_losses, 51, 3)
    # plt.plot(list_losses, color='lightblue')
    # plt.plot(ysmoothed)
    # plt.xlabel("Batch")
    # plt.title("Average loss per batch")
    # plt.ylabel("Loss")
    # plt.yscale("log")
    # plt.savefig("./Images/AvgLosses_" + differentiation_str)
    # plt.close()