import os
import sys
import warnings

from tqdm import tqdm
import mujoco_py
import gym
import numpy as np
import tensorflow as tf

from src.models import get_actor_model, get_critic_model

SEED = 42

class ExperienceBuffer:
    """Experience Buffer used for experience replay by DeepQLearning
    """
    def __init__(self, max_buffer_size, act_space, obs_space):
        self.max_buffer_size = int(max_buffer_size)
        self.obs_space = int(obs_space)
        self.act_space = int(act_space)

        self.states = np.zeros((self.max_buffer_size, self.obs_space))
        self.actions = np.zeros((self.max_buffer_size, self.act_space))
        self.rewards = np.zeros((self.max_buffer_size,1), dtype=np.float32)
        self.next_states = np.zeros((self.max_buffer_size, self.obs_space))
        self.done_flags = np.zeros(self.max_buffer_size, dtype=bool)

        self.row = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done_flag):
        """Add a state to buffer
        """
        self.states[self.row] = state
        self.actions[self.row] = action
        self.rewards[self.row] = reward
        self.next_states[self.row] = next_state
        self.done_flags[self.row] = done_flag
        
        self.row = (self.row + 1) % self.max_buffer_size
        self.size = max(self.size, self.row)
    
    def get_size(self):
        """Get the current size of the buffer
        """
        return self.size

    def get_batch(self, batch_size):
        """Get a random batch from the buffer
        """
        if batch_size > self.get_size():
            warnings.warn('Batch size is larger than curren buffer size.')

        idx = np.random.choice(
            min(self.size, self.max_buffer_size), batch_size, replace=True
        )
        return (
            self.states[idx], 
            self.actions[idx], 
            self.rewards[idx],
            self.next_states[idx],
            self.done_flags[idx]
        )

class DeepDeterministicPolicyGradient:

    def __init__(self, environ='HalfCheetah-v2', **kwargs):

        self.env = gym.make(environ)
        # self.env.seed(kwargs.get('seed', np.random.randint(1e9)))
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.lower_bounds = self.env.action_space.low
        self.upper_bounds = self.env.action_space.high

        self.memory = ExperienceBuffer(
            kwargs.get('buffer_size', 50000),
            self.num_actions,
            self.num_states
        )
        self.batch_size = kwargs.get('batch_size', 64)

        self.polyak = kwargs.get('polyak', 0.005)
        self.gamma = kwargs.get('gamma', 0.99)
        self.action_noise = kwargs.get('action_noise', 0.2)

        lr_actor = kwargs.get('lr_actor', 0.001)
        lr_critic = kwargs.get('lr_critic', 0.002)

        self.actor_model = get_actor_model(
            self.num_states,
            self.num_actions,
            self.upper_bounds
        )
        self.target_actor_model = get_actor_model(
            self.num_states,
            self.num_actions,
            self.upper_bounds
        )

        self.critic_model = get_critic_model(
            self.num_states,
            self.num_actions
        )
        self.target_critic_model = get_critic_model(
            self.num_states,
            self.num_actions
        )

        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        self.actor_model_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_model_optimizer = tf.keras.optimizers.Adam(lr_critic)


    # @tf.function
    def train_one_batch(
        self, critic_model, actor_model, target_critic_model, target_actor_model
    ):
        
        states, actions, rewards, next_states, _ = \
            self.memory.get_batch(self.batch_size)
        
        with tf.GradientTape() as tape:
            target_actions = target_actor_model(next_states, training=True)
            next_q = target_critic_model(
                [next_states, target_actions], 
                training=True
            )

            target_q = rewards + self.gamma * next_q
            pred_q = critic_model([states, actions], training=True)

            critic_loss = tf.math.reduce_mean(tf.math.square(target_q - pred_q))

        critic_gradient = tape.gradient(
            critic_loss, 
            critic_model.trainable_variables
        )
        self.critic_model_optimizer.apply_gradients(
            zip(critic_gradient, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            pred_actions = actor_model(states, training=True)
            pred_q =critic_model([states, pred_actions], training=True)

            actor_loss = -tf.math.reduce_mean(pred_q)

        actor_gradient = tape.gradient(
            actor_loss,
            actor_model.trainable_variables
        )
        self.actor_model_optimizer.apply_gradients(
            zip(actor_gradient, actor_model.trainable_variables)
        )

        return critic_model, actor_model

    @tf.function
    def polyak_update_target(self, target_weights, weights, polyak):
        for wt, w in zip(target_weights, weights):
            wt.assign(w * polyak + wt*(1-polyak))
    
    def policy(self, state, std_noise):
        # obtain new model
        action = self.actor_model(state)
        # sample noise
        noise = np.random.normal(0, std_noise)
        # Add normal noice to selected action
        action = action + noise
        # Clip action within bounds
        action = np.clip(action, self.lower_bounds, self.upper_bounds)

        return np.squeeze(action)

    def train(self, num_episodes, verbose=True, render_every=sys.maxsize, 
              save=None):

        self.total_training_reward = []
        self.average_training_reward = []

        if verbose:
            pbar = tqdm(range(num_episodes))
        else:
            pbar = range(num_episodes)

        for episode in pbar:

            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:

                if verbose and episode != 0 and episode % render_every == 0:
                    self.env.render()

                state_expanded = tf.expand_dims(state, axis=0)

                action = self.policy(state_expanded, self.action_noise)
                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                self.memory.add(state, action, reward, next_state, done)                

                
                self.critic_model, self.actor_model = self.train_one_batch(
                    self.critic_model, 
                    self.actor_model, 
                    self.target_critic_model, 
                    self.target_actor_model
                )

                self.polyak_update_target(
                    self.target_actor_model.variables, 
                    self.actor_model.variables, 
                    0.005
                )
                self.polyak_update_target(
                    self.target_critic_model.variables, 
                    self.critic_model.variables, 
                    0.005
                )

                state = next_state
            
            self.total_training_reward.append(float(episode_reward))

            if verbose:
                pbar.set_postfix(
                    {
                        'Rolling reward': '{:.1f}'.format(
                            np.mean(self.total_training_reward[-40:])
                        )
                    }
                )
            self.average_training_reward.append(
                np.mean(self.total_training_reward[-40:])
            )

            if (save and episode > 0 and
                self.average_training_reward[-1] > self.average_training_reward[-2]):
                
                self.actor_model.save(os.path.join(save, 'actor'))
                self.critic_model.save(os.path.join(save, 'critic'))
            
        
        import matplotlib.pyplot as plt

        plt.plot(self.average_training_reward)
        plt.xlabel("Episode")
        plt.ylabel("Average Epsiodic Reward")
        plt.show()

    def simulate(self, model=None, episodes=1):

        if model:
            self.actor_model = tf.keras.models.load_model(
                os.path.join(model, 'actor')
            )

        for episode in range(episodes):

            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                self.env.render()

                state_expanded = tf.expand_dims(state, axis=0)

                action = self.policy(state_expanded, self.action_noise)
                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                state = next_state
            
            print('episode {:>3.0f}: {:>5.0f}'.format(episode, episode_reward))



    







