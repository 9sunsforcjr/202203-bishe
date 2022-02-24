from __future__ import division
from collections import defaultdict
import struct
import types
import string
import numpy as np
import math
import copy
import random
import datetime

# 搜索 修改
# in v1_4, we start to use continuous time simulation (still with a time granularity, but instead of the number of requests at each following the Poisson distribution,
# the inter-arrival time between requests follows the negative exponential distribution)
# therefore, we can remove the use flag_map from the input
# consider only 16 slots

# for load3, we set lambda_req = 8, for load 2, 5, for load 1, 3

# change topology, here below
# Topology: NSFNet

NODE_NUM = 14
# generate source and destination pairs
# for each src-dst pair, we calculate its cumlative propability based on the traffic distribution
Src_Dest_Pair = []
prob_arr = []
for ii in range(NODE_NUM):
    for jj in range(NODE_NUM):
        if ii != jj:
            temp = []
            temp.append(ii + 1)
            temp.append(jj + 1)
            Src_Dest_Pair.append(temp)
num_src_dest_pair = len(Src_Dest_Pair)

if __name__ == "__main__":
    REQ = [8, 32, 64]
    for num in range(0,10):
        i = 0
        f = open('NSFNET-demand'+str(num)+'.txt', 'a')
        while i < 200:
            i = i + 1
            current_bandwidth = REQ[np.random.randint(0, 3)]
            temp = Src_Dest_Pair[np.random.randint(0, num_src_dest_pair)]
            print(temp)
            f.write(str(temp[0]))
            f.write(str(','))
            f.write(str(temp[1]))
            f.write(str(','))
            f.write(str(current_bandwidth))
            f.write('\n')
        f.close()