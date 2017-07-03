"""
This python demo is a simplest implement of Paper "Playing Atari with Deep Reinforcement Learning Mnih. 2013".
Only purpose here to achive is writing a backbone program, explain this method clear and clean enough for me to quickly think through another day.
"""
#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import time

N_STATES = 3   # the length of the 1 dimensional world
TERMINAL = N_STATES - 1
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

class DeepQNetwork:
    def __init__(
        self
        , n_actions
        , n_features
        , learning_rate=0.01
        , reward_decay=0.9
        , e_greedy=0.9
        , memory_size=3
        , epsilon=0.9
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.memory_size = memory_size
        self.epsilon = epsilon
        
        # Initialze replay memory D to capacity N. content: [s, a, r, s_]
        self.D = np.zeros((memory_size, n_features*2 + 2))
        self.NN = self._build_net()
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    
    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # for train
        with tf.variable_scope('q_net'):
            c_names = ['q_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0, 0.3)
            b_initializer = tf.constant_initializer(0.1) # for relu
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            self.q_predict = tf.matmul(l1, w2) + b2
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q_predict)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.D[index, :] = transition

        self.memory_counter += 1
    
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_predict, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(self.n_actions)
        return action

    def learn(self):
        y = self.sess.run(self.q_predict, feed_dict={self.s: self.D[:, :self.n_features]})
        for i in range(self.memory_size):
            s, a, r, s_ = self.D[i]
            if s_ != TERMINAL:
                s_tensor = np.array([s_])[np.newaxis, :]
                q_predict = self.sess.run(self.q_predict, feed_dict={self.s: s_tensor})
                y[i][int(a)] = r + self.reward_decay * np.max(q_predict)
            else:
                y[i][int(a)] = r
        _, self.cost = self.sess.run([self._train_op, self.loss]
                                    , feed_dict={self.s: self.D[:, :self.n_features]
                                                , self.q_target: y})
        print("Cost: {}".format(self.cost))
        
        

    def _debug_print_memory(self):
        print(self.D)
    
    def _debug_print_q_table(self):
        """
        Print all states of Q values.
        """
        q_table = np.zeros((N_STATES, len(ACTIONS)))
        for i in range(N_STATES):
            observation = np.array([i])[np.newaxis, :]
            q_table[i, :] = actions_value = self.sess.run(self.q_predict, feed_dict={self.s: observation})
        q_table = pd.DataFrame(q_table, columns=ACTIONS)
        print(q_table)




class ENV:
    def __init__(self, n_states):
        self.n_states = n_states
        self.state = 0

    def step(self, A):
        if A == 'right':
            if self.state == self.n_states - 2:
                S_ = self.state + 1
                R = 1
            else:
                S_ = self.state + 1
                R = 0
        else:
            R = 0
            if self.state == 0:
                S_ = self.state # reach left wall
            else:
                S_ = self.state - 1
        self.state = S_
        return S_, R


    def render(self, episode, step_counter):
        env_list = ['-'] * (self.n_states - 1) + ['T'] # '---------T' our environment
        if self.state == 'terminal':
            interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                                ', end='')
        else:
            env_list[self.state] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(FRESH_TIME)


RL = DeepQNetwork(len(ACTIONS), 1)
RL._debug_print_q_table()

#%%
MAX_EPISODES = 13
env = ENV(N_STATES)
for i in range(RL.memory_size):
    a = np.random.randint(len(ACTIONS))
    s = env.state
    s_, r = env.step(ACTIONS[a])
    print(' -> '.join(['state '+str(s), ACTIONS[a], 'reward '+str(r), 'state '+str(s_)]))
    RL.store_transition(s, a, r, s_)
    if TERMINAL == s_:
        env.state = 0 # restart

RL.learn()
RL._debug_print_q_table()

#%%
for i in range(MAX_EPISODES):
    env.state = 0
    step_count = 0
    while env.state != TERMINAL:
        s = env.state
        a = RL.choose_action(np.array([env.state]))
        s_, r = env.step(ACTIONS[a])
        RL.store_transition(s, a, r, s_)
        step_count += 1
    RL.learn()
    RL._debug_print_q_table()
    print("Used {} steps to get to terminal.".format(step_count))
    

