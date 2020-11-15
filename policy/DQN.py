import tensorflow as tf
import numpy as np
import pandas as pd



class QNET:
    def __init__(self, n_state, n_action, lr, lr_decay):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        self.lr_decay = lr_decay

    def backbone(self, input_tensor):
        x = tf.keras.layers.Dense(128)(input_tensor)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation('relu')(x)

        output_tensor = tf.keras.layers.Dense(self.n_action)(x)

        return output_tensor

    def build(self):
        input_tensor = tf.keras.Input([self.n_state, ])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mse'
        )

        return model

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['state', 'action', 'reward',
                                            'next_state', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

    def clear(self):
        self.__init__(capacity=self.capacity)

class DQNAgent:
    def __init__(self,
                 n_state,
                 n_action,
                 gamma=0.99,
                 epsilon=0.001,
                 replayer_capacity=10000,
                 batch_size=128,
                 batches=5,
                 lr=1e-3,
                 lr_decay=1e-5
                 ):
        self.n_state = n_state
        self.n_action = n_action

        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.batches = batches
        self.lr = lr
        self.lr_decay = lr_decay
        self.replayer = DQNReplayer(replayer_capacity)  # 经验回放

        self.network = QNET(
            n_state=self.n_state,
            n_action=self.n_action,
            lr=self.lr,
            lr_decay=self.lr_decay
        )

        self.eval_qnet = self.network.build()  # 评估网络
        self.target_qnet = self.network.build()  # 目标网络

        self.target_qnet.set_weights(self.eval_qnet.get_weights())

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store(state, action, reward, next_state, done)

        for batches in range(self.batches):
            states, actions, rewards, next_states, dones = self.replayer.sample(self.batch_size)

            next_qs = self.target_qnet(next_states)
            next_max_qs = tf.reduce_max(next_qs, axis=-1)
            us = rewards + self.gamma * (1. - dones) * next_max_qs
            targets = self.eval_qnet(states).numpy()
            targets[np.arange(us.shape[0]), actions] = us
            self.eval_qnet.fit(states, targets, verbose=1)

        if done:
            self.target_qnet.set_weights(self.eval_qnet.get_weights())

    def decide(self, state):  # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            epsilon_qs = np.random.rand(self.n_action)
            return np.argmax(epsilon_qs)

        state_tensor = tf.convert_to_tensor(state, tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        qs = self.eval_qnet(state_tensor)
        return np.argmax(qs[0])

    def save(self, epoch):
        self.eval_qnet.save('./snapshots/' + '_eval_qnet_epoch' + str(epoch) + '.h5')
