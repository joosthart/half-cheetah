import os
import sys
import warnings

from tqdm import tqdm
import mujoco_py
import gym
import numpy as np
import tensorflow as tf

from src.ddpg.models import get_actor_model, get_critic_model

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
            warnings.warn('Batch size is larger than current buffer size.')

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
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.lower_bounds = self.env.action_space.low
        self.upper_bounds = self.env.action_space.high

        # Set random seeds
        self.seed = kwargs.get('seed', np.random.randint(1, 2**32-1))
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # init buffer
        self.memory = ExperienceBuffer(
            kwargs.get('buffer_size', 50000),
            self.num_actions,
            self.num_states
        )

        # Train parameters
        self.batch_size = kwargs.get('batch_size', 64)
        self.polyak = kwargs.get('polyak', 0.005)
        self.gamma = kwargs.get('gamma', 0.99)
        self.action_noise = kwargs.get('action_noise', 0.2)

        # init models
        lr_actor = kwargs.get('lr_actor', 0.001)
        lr_critic = kwargs.get('lr_critic', 0.002)

        self.actor_model = get_actor_model(
            self.num_states,
            self.num_actions,
            self.upper_bounds,
            hidden = kwargs.get('actor_hidden', [256, 256]),
            init_min=kwargs.get('actor_init_min', -0.003),
            init_max=kwargs.get('actor_init_max', 0.003),
        )
        self.target_actor_model = get_actor_model(
            self.num_states,
            self.num_actions,
            self.upper_bounds,
            hidden = kwargs.get('actor_hidden', [256, 256]),
            init_min=kwargs.get('actor_init_min', -0.003),
            init_max=kwargs.get('actor_init_max', 0.003),
        )

        self.critic_model = get_critic_model(
            self.num_states,
            self.num_actions,
            hidden_state = kwargs.get('critic_hidden_state', [16,32]),
            hidden_action = kwargs.get('critic_hidden_action', [32]),
            hidden_common = kwargs.get('critic_hidden_common', [256, 256]),
        )
        self.target_critic_model = get_critic_model(
            self.num_states,
            self.num_actions,
            hidden_state = kwargs.get('critic_hidden_state', [16,32]),
            hidden_action = kwargs.get('critic_hidden_action', [32]),
            hidden_common = kwargs.get('critic_hidden_common', [256, 256]),
        )

        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        self.actor_model_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_model_optimizer = tf.keras.optimizers.Adam(lr_critic)


    def train_one_batch(
        self, critic_model, actor_model, target_critic_model, target_actor_model
    ):
        """Train actor-critic model for one batch of data sampled from memory 
        buffer.

        Args:
            critic_model (tf.Model): Tensorflow critic model
            actor_model (tf.Model): Tensorflow actor model
            target_critic_model (tf.Model): Tensorflow target critic model
            target_actor_model (tf.Model): Tensorflow target actor model

        Returns:
            (tf.Model, tf.Model): Trained critic and target model
        """
        
        # Sample batch from buffer
        states, actions, rewards, next_states = \
            self.memory.get_batch(self.batch_size)
        
        with tf.GradientTape() as tape:
            # Predict target next Q-values
            target_actions = target_actor_model(next_states, training=True)
            next_q = target_critic_model(
                [next_states, target_actions], 
                training=True
            )

            # Bellman's equation
            target_q = rewards + self.gamma * next_q
            pred_q = critic_model([states, actions], training=True)
            # Calculate critc model loss using Mean Squared Bellman Loss
            critic_loss = tf.math.reduce_mean(tf.math.square(target_q - pred_q))

        # Update critic model
        critic_gradient = tape.gradient(
            critic_loss, 
            critic_model.trainable_variables
        )
        self.critic_model_optimizer.apply_gradients(
            zip(critic_gradient, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            # Predict Q-values for states
            pred_actions = actor_model(states, training=True)
            pred_q = critic_model([states, pred_actions], training=True)

            # Calculate actor loss using inverse mean Q-values
            actor_loss = -tf.math.reduce_mean(pred_q)

        # Update actor model
        actor_gradient = tape.gradient(
            actor_loss,
            actor_model.trainable_variables
        )
        self.actor_model_optimizer.apply_gradients(
            zip(actor_gradient, actor_model.trainable_variables)
        )

        return critic_model, actor_model

    def polyak_update_target(self, target_weights, weights, polyak):
        """Polyak average trained and target weights

        Args:
            target_weights (np.array): target weights
            weights (np.arry]): trained weight
            polyak (float]): Polyak parameter
        """
        for wt, w in zip(target_weights, weights):
            wt.assign(w * polyak + wt*(1-polyak))
    
    def policy(self, state, std_noise):
        """ Policy used to predict an action given a state

        Args:
            state (np.array): State observation array
            std_noise (float): Standard deviation of Gaussian noised added to 
                action.

        Returns:
            np.array: Predicted action
        """
        # obtain new model
        action = self.actor_model(state)
        # sample noise
        noise = np.random.normal(0, std_noise, action.shape)
        # Add normal noice to selected action
        action = action + noise
        # Clip action within bounds
        action = np.clip(action, self.lower_bounds, self.upper_bounds)

        return np.squeeze(action)

    def train(self, num_episodes, verbose=True, render_every=sys.maxsize, 
              save=None):
        """ Train DDPG actor-critic model

        Args:
            num_episodes (int): Number of episodes
            verbose (bool, optional): Whether to print progress. Defaults to 
                True.
            render_every (int, optional): Render environment every i episodes. 
                Defaults to sys.maxsize.
            save (str, optional): If sting give, save model to given path. 
                Defaults to None.
        """

        self.total_training_reward = []
        self.average_training_reward = []

        if verbose:
            pbar = tqdm(range(num_episodes))
        else:
            pbar = range(num_episodes)

        for episode in pbar:

            # set environment seed
            self.env.seed(self.seed+episode)
            state = self.env.reset()

            episode_reward = 0
            done = False

            while not done:

                if verbose and episode != 0 and episode % render_every == 0:
                    self.env.render()

                # Transform state to tensor
                state_expanded = tf.expand_dims(state, axis=0)
                # Get action from policy
                action = self.policy(state_expanded, self.action_noise)
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                # update reward
                episode_reward += reward
                # Add current state to memory buffer
                self.memory.add(state, action, reward, next_state, done)                

                # train actor-critic model for one batch                
                self.critic_model, self.actor_model = self.train_one_batch(
                    self.critic_model, 
                    self.actor_model, 
                    self.target_critic_model, 
                    self.target_actor_model
                )
                # Polyak average train and target model
                self.polyak_update_target(
                    self.target_actor_model.variables, 
                    self.actor_model.variables, 
                    self.polyak
                )
                self.polyak_update_target(
                    self.target_critic_model.variables, 
                    self.critic_model.variables, 
                    self.polyak
                )

                state = next_state
            
            self.total_training_reward.append(float(episode_reward))

            # Display progress
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

            # If model best seen so far, save model
            if (save and episode > 0 and
                self.average_training_reward[-1] > self.average_training_reward[-2]):
                self.actor_model.save(os.path.join(save, 'actor'))
                self.critic_model.save(os.path.join(save, 'critic'))

        # Save training history
        if save:
            path = 'episodic_reward.npy'
            with open(os.path.join(save, path), 'wb') as f:
                np.save(f, self.total_training_reward)
            path = 'mean_episodic_reward.npy'
            with open(os.path.join(save, path), 'wb') as f:
                np.save(f, self.average_training_reward)
                

    def simulate(self, model=None, episodes=1, render=True, verbose=True):
        """Simulate an environment given a trained model

        Args:
            model (tf.Model, optional): Trained model. Defaults to None.
            episodes (int, optional): Number of simulation to run. Defaults to 
                1.
            render (bool, optional): If True, render environment at every step. 
                Defaults to True.
            verbose (bool, optional): Whether to print progress. Defaults to 
                True.

        Returns:
            (np.array, np.array): mean and std of simulation rewards
        """
        if model:
            self.actor_model = tf.keras.models.load_model(
                os.path.join(model, 'actor')
            )
        
        sim_reward = []

        for episode in range(episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Render environment
                if render:
                    self.env.render()
                # Transform state to tensor
                state_expanded = tf.expand_dims(state, axis=0)
                # Get action from policy without noise
                action = self.policy(state_expanded, 0)
                # Take action
                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                state = next_state
            
            sim_reward.append(episode_reward)
            # Display progress
            if verbose:
                print('episode {:>3.0f}: {:>5.0f}'.format(
                    episode, episode_reward))

        return np.mean(sim_reward), np.std(sim_reward)


    







