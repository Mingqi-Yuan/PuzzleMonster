import tensorflow as tf
import numpy as np
import pandas as pd

class Actor:
    def __init__(self, n_state, n_action, lr, lr_decay, loss):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        self.lr_decay = lr_decay
        self.loss = loss

    def backbone(self, input_tensor):
        x = tf.keras.layers.Dense(128)(input_tensor)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Dense(self.n_action)(x)
        output_tensor = tf.keras.layers.Activation('softmax')(x)

        return output_tensor

    def build(self):
        input_tensor = tf.keras.Input([self.n_state, ])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=self.loss
        )

        return model

class Critic:
    def __init__(self, n_state, n_action, lr, lr_decay, loss):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        self.lr_decay = lr_decay
        self.loss = loss

    def backbone(self, input_tensor):
        x = tf.keras.layers.Dense(128)(input_tensor)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation('relu')(x)

        output_tensor = tf.keras.layers.Dense(1)(x)

        return output_tensor

    def build(self):
        input_tensor = tf.keras.Input([self.n_state, ])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=self.loss
        )

        return model

class PPOReplayer:
    def __init__(self):
        self.memory = pd.DataFrame()

    def store(self, df):  # 存储经验
        self.memory = pd.concat([self.memory, df], ignore_index=True)

    def sample(self, size):  # 回放经验
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field \
                in self.memory.columns)


class PPOAgent():
    def __init__(self,
                 n_state,
                 n_action,
                 clip_ratio=0.1,
                 gamma=0.99, 
                 lambd=0.99, 
                 min_trajectory_length=1000,
                 batches=1, 
                 batch_size=64,
                 lr=1e-3,
                 lr_decay=1e-5
                 ):
        self.n_state = n_state
        self.n_action = n_action

        self.gamma = gamma
        self.lambd = lambd
        self.min_trajectory_length = min_trajectory_length
        self.lr = lr
        self.lr_decay = lr_decay
        self.batches = batches
        self.batch_size = batch_size

        self.trajectory = []  # 存储回合内的轨迹
        self.replayer = PPOReplayer()

        def ppo_loss(y_true, y_pred):
            p = y_pred  # 新策略概率
            p_old = y_true[:, :self.n_action]  # 旧策略概率
            advantage = y_true[:, self.n_action:]  # 优势
            surrogate_advantage = (p / p_old) * advantage  # 代理优势
            clip_times_advantage = clip_ratio * advantage
            max_surrogate_advantage = advantage + tf.where(advantage > 0.,
                                                           clip_times_advantage, -clip_times_advantage)
            clipped_surrogate_advantage = tf.minimum(surrogate_advantage,
                                                     max_surrogate_advantage)
            return - tf.reduce_mean(clipped_surrogate_advantage, axis=-1)

        self.actor_net = Actor(
            n_state=self.n_state,
            n_action=self.n_action,
            lr=self.lr,
            lr_decay=self.lr_decay,
            loss=ppo_loss
        ).build()
        
        self.critic_net = Critic(
            n_state=self.n_state,
            n_action=self.n_action,
            lr=self.lr,
            lr_decay=self.lr_decay,
            loss='mse'
        ).build()

    def decide(self, state):
        probs = self.actor_net.predict(state[np.newaxis])[0]
        action = np.random.choice(self.n_action, p=probs)
        return action

    def learn(self, state, action, reward, done):
        self.trajectory.append((state, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory, columns=['state', 'action', 'reward'])  # 开始对本回合经验进行重构
            states = np.stack(df['state'])
            df['v'] = self.critic_net.predict(states)
            pis = self.actor_net.predict(states)
            df['pi'] = [a.flatten() for a in np.split(pis, pis.shape[0])]

            df['next_v'] = df['v'].shift(-1).fillna(0.)
            df['u'] = df['reward'] + self.gamma * df['next_v']
            df['delta'] = df['u'] - df['v']  # 时序差分误差
            df['return'] = df['reward']  # 初始化优势估计，后续会再更新
            df['advantage'] = df['delta']  # 初始化优势估计，后续会再更新
            for i in df.index[-2::-1]:  # 指数加权平均
                df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
                df.loc[i, 'advantage'] += self.gamma * self.lambd * \
                                          df.loc[i + 1, 'advantage']  # 估计优势
            fields = ['state', 'action', 'pi', 'advantage', 'return']
            self.replayer.store(df[fields])  # 存储重构后的回合经验
            self.trajectory = []  # 为下一回合初始化回合内经验

        if len(self.replayer.memory) > self.min_trajectory_length:
            for batch in range(self.batches):
                states, actions, pis, advantages, returns = \
                    self.replayer.sample(size=self.batch_size)
                ext_advantages = np.zeros_like(pis)
                ext_advantages[range(self.batch_size), actions] = \
                    advantages
                actor_targets = np.hstack([pis, ext_advantages])  # 执行者目标
                self.actor_net.fit(states, actor_targets, verbose=0)
                self.critic_net.fit(states, returns, verbose=0)

            self.replayer = PPOReplayer()  # 为下一回合初始化经验回放