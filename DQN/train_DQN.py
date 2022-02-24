# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import numpy as np
import gym
import gc
import os
import sys
import gym_environments
import random
import mpnn as gnn
import tensorflow as tf
from collections import deque
import multiprocessing
import time as tt
import glob
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不用GPU，用CPU

ENV_NAME = 'GraphEnv-v1'
graph_topology = 0  # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
ITERATIONS = 10000
TRAINING_EPISODES = 40  # 每个迭代进行多少次request和reward
EVALUATION_EPISODES = 50
FIRST_WORK_TRAIN_EPISODE = 70

MULTI_FACTOR_BATCH = 6  # Number of batches used in training
TAU = 0.08  # Only used in soft weights copy

differentiation_str = "sample_DQN_agent"
checkpoint_dir = "./models" + differentiation_str
store_loss = 3  # Store the loss every store_loss batches

os.environ['PYTHONHASHSEED'] = str(SEED)  # 设置为0，禁止哈希随机化，结果可复现
np.random.seed(SEED)
random.seed(SEED)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

tf.random.set_seed(1)

train_dir = "./TensorBoard/" + differentiation_str
# summary_writer = tf.summary.create_file_writer(train_dir)
listofDemands = [8, 32, 64]
copy_weights_interval = 50
evaluation_interval = 20
epsilon_start_decay = 70

hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'T': 4,
    'num_demands': len(listofDemands)
}

MAX_QUEUE_SIZE = 4000


def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes


class DQNAgent:
    def __init__(self, batch_size):
        self.memory = deque(maxlen=MAX_QUEUE_SIZE)  # 4000
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.writer = None
        self.K = 4  # K-paths 选出4条候选路径
        self.listQValues = None
        self.numbersamples = batch_size
        self.action = None
        self.capacity_feature = None
        # 分配的带宽特征：以环境的边的数量为行，需求长度？为列数==3
        self.bw_allocated_feature = np.zeros((env_training.numEdges, len(env_training.listofDemands)))

        self.global_step = 0
        self.primary_network = gnn.myModel(hparams)  # mpnn.py文件
        self.primary_network.build()
        self.target_network = gnn.myModel(hparams)
        self.target_network.build()

        # SGD：随机梯度下降优化器
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], momentum=0.9, nesterov=True)

    def act(self, env, state, demand, source, destination, flagEvaluation):
        """
        Given a demand stored in the environment it allocates the K=4 shortest paths on the current 'state'
        and predicts the q_values of the K=4 different new graph states by using the GNN model.
        Picks the state according to epsilon-greedy approach. The flag=TRUE indicates that we are testing
        the model and thus, it won't activate the drop layers.
        """

        # Set to True if we need to compute K=4 q-values and take the maxium
        takeMax_epsilon = False
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # Initialize action
        action = 0

        # We get the K-paths between source-destination
        pathList = env.allPaths[str(source) + ':' + str(destination)]
        # print('//////////////')
        # print(pathList)
        path = 0

        # 1. 使用epsilon-greedy来选择分配方式
        # If flagEvaluation==TRUE 评估时选择Q值最高的action
        # 训练时使用正常的epsilon-greedy算法
        if flagEvaluation:
            # 评估时计算K=4条路的Q值并取最大值
            takeMax_epsilon = True
        else:
            # 训练时计算 epsilon-greedy
            z = np.random.random()
            if z > self.epsilon:
                # 计算K=4条路的Q值并取最大值
                # 有多条路的Q值都最大时默认返回第一条
                takeMax_epsilon = True
            else:
                # Pick a random path and compute only one q-value 从k=4里随便选一条
                path = np.random.randint(0, len(pathList))
                action = path

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path] # 正常情况对pathList里4条路径遍历
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate demand to bw_allocated
            while (j < len(currentPath)):
                state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute their q-values using cummax
            listGraphs.append(state_copy)
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)

            if not takeMax_epsilon:
                # If we don't need to compute the K=4 q-values we exit
                break

            path = path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        }
        )

        # Predict qvalues for all graphs within tensors
        self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                                                tensors['second'], tensors['num_edges'], training=False).numpy()

        if takeMax_epsilon:
            # If we computed the K=4 q-values, we take the max
            action = np.argmax(self.listQValues)
        else:
            action = 0

        # 返回改变的状态下标和改变过的state
        return action, list_k_features[action]

    def get_graph_features(self, env, copyGraph):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.bw_allocated_feature.fill(0.0)


        # Normalize capacity feature 扩容，由200ODU0改为500ODU0
        self.capacity_feature = (copyGraph[:, 0] - 100.00000001) / 200.0
        # self.capacity_feature = (copyGraph[:, 0] - 100.00000001) / 500.0

        iter = 0
        for i in copyGraph[:, 1]:
            if i == 8:
                self.bw_allocated_feature[iter][0] = 1
            elif i == 32:
                self.bw_allocated_feature[iter][1] = 1
            elif i == 64:
                self.bw_allocated_feature[iter][2] = 1
            iter = iter + 1

        sample = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2 - hparams['num_demands']]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def _write_tf_summary(self, gradients, loss):
        with summary_writer.as_default():
            tf.summary.scalar(name="loss", data=loss[0], step=self.global_step)
            tf.summary.histogram(name='gradients_5', data=gradients[5], step=self.global_step)
            tf.summary.histogram(name='gradients_7', data=gradients[7], step=self.global_step)
            tf.summary.histogram(name='gradients_9', data=gradients[9], step=self.global_step)
            tf.summary.histogram(name='FirstLayer/kernel:0', data=self.primary_network.variables[0],
                                 step=self.global_step)
            tf.summary.histogram(name='FirstLayer/bias:0', data=self.primary_network.variables[1],
                                 step=self.global_step)
            tf.summary.histogram(name='kernel:0', data=self.primary_network.variables[2], step=self.global_step)
            tf.summary.histogram(name='recurrent_kernel:0', data=self.primary_network.variables[3],
                                 step=self.global_step)
            tf.summary.histogram(name='bias:0', data=self.primary_network.variables[4], step=self.global_step)
            tf.summary.histogram(name='Readout1/kernel:0', data=self.primary_network.variables[5],
                                 step=self.global_step)
            tf.summary.histogram(name='Readout1/bias:0', data=self.primary_network.variables[6], step=self.global_step)
            tf.summary.histogram(name='Readout2/kernel:0', data=self.primary_network.variables[7],
                                 step=self.global_step)
            tf.summary.histogram(name='Readout2/bias:0', data=self.primary_network.variables[8], step=self.global_step)
            tf.summary.histogram(name='Readout3/kernel:0', data=self.primary_network.variables[9],
                                 step=self.global_step)
            tf.summary.histogram(name='Readout3/bias:0', data=self.primary_network.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1

    @tf.function
    def _forward_pass(self, x):
        prediction_state = self.primary_network(x[0], x[1], x[2], x[3], x[4], training=True)
        preds_next_target = tf.stop_gradient(self.target_network(x[6], x[7], x[9], x[10], x[11], training=True))
        return prediction_state, preds_next_target

    def _train_step(self, batch):
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            preds_state = []
            target = []
            for x in batch:
                prediction_state, preds_next_target = self._forward_pass(x)
                # Take q-value of the action performed
                preds_state.append(prediction_state[0])
                # We multiple by 0 if done==TRUE to cancel the second term
                target.append(
                    tf.stop_gradient([x[5] + self.gamma * tf.math.reduce_max(preds_next_target) * (1 - x[8])]))

            loss = tf.keras.losses.MSE(tf.stack(target, axis=1), tf.stack(preds_state, axis=1))
            # Loss function using L2 Regularization
            regularization_loss = sum(self.primary_network.losses)
            loss = loss + regularization_loss

        # Computes the gradient using operations recorded in context of this tape
        grad = tape.gradient(loss, self.primary_network.variables)
        # gradients, _ = tf.clip_by_global_norm(grad, 5.0)
        gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        self.optimizer.apply_gradients(zip(gradients, self.primary_network.variables))
        del tape
        return grad, loss

    def replay(self, episode):
        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(self.memory, self.numbersamples)

            grad, loss = self._train_step(batch)
            if i % store_loss == 0:
                fileLogs.write(".," + '%.9f' % loss.numpy() + ",\n")

        # Soft weights update
        # for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
        #     t.assign(t * (1 - TAU) + e * TAU)

        # Hard weights update
        if episode % copy_weights_interval == 0:
            self.target_network.set_weights(self.primary_network.get_weights())
            # if episode % evaluation_interval == 0:
        #     self._write_tf_summary(grad, loss)
        gc.collect()

    def add_sample(self, env_training, state_action, action, reward, done, new_state, new_demand, new_source,
                   new_destination):
        self.bw_allocated_feature.fill(0.0)
        new_state_copy = np.copy(new_state)

        state_action['graph_id'] = tf.fill([tf.shape(state_action['link_state'])[0]], 0)

        # We get the K-paths between new_source-new_destination
        pathList = env_training.allPaths[str(new_source) + ':' + str(new_destination)]
        path = 0
        list_k_features = list()

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            currentPath = pathList[path]
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate new_demand to bw_allocated
            while (j < len(currentPath)):
                new_state_copy[env_training.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = new_demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute it's qvalues using cummax
            features = agent.get_graph_features(env_training, new_state_copy)

            list_k_features.append(features)
            path = path + 1
            new_state_copy[:, 1] = 0

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        }
        )

        # We store the state with the action marked, the graph ids, first, second, num_edges, the reward, 
        # new_state(-1 because we don't need it in this case), the graph ids, done, first, second, number of edges
        self.memory.append((state_action['link_state'], state_action['graph_id'], state_action['first'],  # 2
                            state_action['second'], tf.convert_to_tensor(state_action['num_edges']),  # 4
                            tf.convert_to_tensor(reward, dtype=tf.float32), tensors['link_state'], tensors['graph_id'],
                            # 7
                            tf.convert_to_tensor(int(done == True), dtype=tf.float32), tensors['first'],
                            tensors['second'],  # 10
                            tf.convert_to_tensor(tensors['num_edges'])))  # 12


if __name__ == "__main__":
    # python train_DQN.py
    # Get the environment and extract the number of actions.
    env_training = gym.make(ENV_NAME)  # 返回训练用环境
    np.random.seed(SEED)
    env_training.seed(SEED)
    env_training.generate_environment(graph_topology, listofDemands)

    env_eval = gym.make(ENV_NAME)  # 返回评估用环境
    # np.random.seed(SEED)
    env_eval.seed(SEED)
    env_eval.generate_environment(graph_topology, listofDemands)

    batch_size = hparams['batch_size']
    agent = DQNAgent(batch_size)

    eval_ep = 0
    train_ep = 0
    max_reward = 0
    reward_id = 0

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    # We store all the information in a Log file and later we parse this file 
    # to extract all the relevant information
    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    # modelssample_DQN_agent
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # Tensorflow的Checkpoint机制将可追踪变量以二进制的方式储存成一个.ckpt文件，储存了变量的名称及对应张量的值。
    #
    # Checkpoint 只保存模型的参数，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。

    # tf.train.Checkpoint() 接受的初始化参数比较特殊，是一个 **kwargs 。
    # 具体而言，是一系列的键值对，键名model可以随意取，值optimizer为需要保存的对象
    checkpoint = tf.train.Checkpoint(model=agent.primary_network, optimizer=agent.optimizer)

    # 例如，在源代码目录建立一个名为 save 的文件夹并调用一次 checkpoint.save('./save/model.ckpt') ，
    # 我们就可以在可以在 save 目录下发现名为 checkpoint 、 model.ckpt-1.index 、 model.ckpt-1.data-00000-of-00001 的三个文件，
    # 这些文件就记录了变量信息。checkpoint.save() 方法可以运行多次，每运行一次都会得到一个. index 文件和. data 文件，序号依次累加。
    # 当在其他地方需要为模型重新载入之前保存的参数时，需要再次实例化一个 checkpoint，同时保持键名的一致。再调用 checkpoint 的 restore 方法。就像下面这样：
    #
    # model_to_be_restored = MyModel()                                        # 待恢复参数的同一模型
    # checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)   # 键名保持为“myAwesomeModel”
    # checkpoint.restore(save_path_with_prefix_and_index)
    # 即可恢复模型变量。 save_path_with_prefix_and_index 是之前保存的文件的目录 + 前缀 + 编号。
    # 例如，调用 checkpoint.restore('./save/model.ckpt-1') 就可以载入前缀为 model.ckpt ，序号为 1 的文件来恢复模型。

    # 当保存了多个文件时，我们往往想载入最近的一个。可以使用 tf.train.latest_checkpoint(save_path) 这个辅助函数f。例如如果 save 目录下有 model.ckpt-1.index
    # 到 model.ckpt-10.index 的 10 个保存文件， tf.train.latest_checkpoint('./save') 即返回 ./save/model.ckpt-10

    # ----------------模型训练阶段-----------------------
    #
    # model = MyModel()
    # # 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
    # checkpoint = tf.train.Checkpoint(myModel=model)
    # # ...（模型训练代码）
    # # 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
    # checkpoint.save('./save/model.ckpt')
    # ----------------模型使用阶段-----------------------
    #
    # model = MyModel()
    # checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
    # checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数

    rewards_test = np.zeros(EVALUATION_EPISODES)

    for eps in range(EVALUATION_EPISODES):

        # 重新初始化函数
        # 在强化学习中，agent需要多次尝试，积累经验，然后从经验中学到好的动作。一次尝试我们称之为一条trajectory或一个episode。
        # 每次尝试都需要从初始状态达到终止状态。
        # 一次尝试结束后，需要从头开始，即重新初始化
        # 源码：
        #
        # def _reset():
        # self.state = self.np_random.uniform(low = -0.05,high=0.05,size=(4,))
        # ##利用均匀随机分布初始化环境的状态
        # self.steps_beyond_done =None
        # ## 设置当前步数为None
        # return np.array(self.state)
        # ## 返回环境的初始化状态

        state, demand, source, destination = env_eval.reset()
        # print('-----------')
        # print(state)
        # print(demand)
        # print(source)
        # print(destination)

        rewardAddTest = 0
        while 1:
            # We execute evaluation over current state
            # demand, src, dst
            action, _ = agent.act(env_eval, state, demand, source, destination, True)

            new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source,
                                                                                      destination)
            # print('*****')
            # print(state)
            # print(demand)
            # print(source)
            # print(destination)
            rewardAddTest = rewardAddTest + reward
            state = new_state
            if done:
                break
        rewards_test[eps] = rewardAddTest

    evalMeanReward = np.mean(rewards_test)
    fileLogs.write(">," + str(evalMeanReward) + ",\n")
    fileLogs.write("-," + str(agent.epsilon) + ",\n")
    fileLogs.flush()

    counter_store_model = 1

    for ep_it in range(ITERATIONS):
        if ep_it % 5 == 0:
            print("Training iteration: ", ep_it)

        if ep_it % 50 == 0:
            now = time.localtime()
            nowTime = time.strftime("%Y-%m-%d %H-%M-%S", now)
            print("Time: ", nowTime)

        if ep_it == 0:
            # At the beginning we don't have any experiences in the buffer. Thus, we force to
            # perform more training episodes than usually
            train_episodes = FIRST_WORK_TRAIN_EPISODE
        else:
            train_episodes = TRAINING_EPISODES
        for _ in range(train_episodes):
            # Used to clean the TF cache
            tf.random.set_seed(1)

            state, demand, source, destination = env_training.reset()

            while 1:
                # We execute evaluation over current state
                action, state_action = agent.act(env_training, state, demand, source, destination, False)
                new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action,
                                                                                                          demand,
                                                                                                          source,
                                                                                                          destination)

                agent.add_sample(env_training, state_action, action, reward, done, new_state, new_demand, new_source,
                                 new_destination)
                state = new_state
                demand = new_demand
                source = new_source
                destination = new_destination
                if done:
                    break

        agent.replay(ep_it)

        # Decrease epsilon (from epsion-greedy exploration strategy)
        if ep_it > epsilon_start_decay and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon *= agent.epsilon_decay

        # We only evaluate the model every evaluation_interval steps
        if ep_it % evaluation_interval == 0:
            for eps in range(EVALUATION_EPISODES):
                state, demand, source, destination = env_eval.reset()
                rewardAddTest = 0
                while 1:
                    # We execute evaluation over current state
                    action, _ = agent.act(env_eval, state, demand, source, destination, True)

                    new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand,
                                                                                              source, destination)
                    rewardAddTest = rewardAddTest + reward
                    state = new_state
                    if done:
                        break
                rewards_test[eps] = rewardAddTest
            evalMeanReward = np.mean(rewards_test)

            if evalMeanReward > max_reward:
                max_reward = evalMeanReward
                reward_id = counter_store_model

            fileLogs.write(">," + str(evalMeanReward) + ",\n")
            fileLogs.write("-," + str(agent.epsilon) + ",\n")

            # Store trained model
            checkpoint.save(checkpoint_prefix)
            fileLogs.write("MAX REWD: " + str(max_reward) + " MODEL_ID: " + str(reward_id) + ",\n")
            counter_store_model = counter_store_model + 1

        fileLogs.flush()

        # Invoke garbage collection
        # tf.keras.backend.clear_session()
        gc.collect()

    for eps in range(EVALUATION_EPISODES):
        state, demand, source, destination = env_eval.reset()
        rewardAddTest = 0
        while 1:
            # We execute evaluation over current state
            # demand, src, dst
            action, _ = agent.act(env_eval, state, demand, source, destination, True)

            new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source,
                                                                                      destination)
            rewardAddTest = rewardAddTest + reward
            state = new_state
            if done:
                break
        rewards_test[eps] = rewardAddTest
    evalMeanReward = np.mean(rewards_test)

    if evalMeanReward > max_reward:
        max_reward = evalMeanReward
        reward_id = counter_store_model

    fileLogs.write(">," + str(evalMeanReward) + ",\n")
    fileLogs.write("-," + str(agent.epsilon) + ",\n")

    # Store trained model
    checkpoint.save(checkpoint_prefix)
    fileLogs.write("MAX REWD: " + str(max_reward) + " MODEL_ID: " + str(reward_id) + ",\n")

    fileLogs.flush()
    fileLogs.close()
