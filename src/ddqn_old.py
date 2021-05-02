import copy
from tensorflow.python.keras.backend import dtype

from tensorflow.python.keras.engine import training
from tqdm import tqdm
import mujoco_py
import gym
import numpy as np
import tensorflow as tf

from src.models import ActorModel, CriticModel

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
        self.rewards = np.zeros((self.max_buffer_size,1))
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
        # if batch_size > self.get_size():
        #     raise ValueError(
        #         'Batch size is too large. Batch size should be less than or '
        #         'equal to current buffer size.'
        #     )

        idx = np.random.choice(
            min(self.size, self.max_buffer_size), batch_size, replace=True
        )
        return (
            tf.convert_to_tensor(self.states[idx]), 
            tf.convert_to_tensor(self.actions[idx]), 
            tf.cast(tf.convert_to_tensor(self.rewards[idx]), dtype=tf.float32),
            tf.convert_to_tensor(self.next_states[idx]),
            tf.convert_to_tensor(self.done_flags[idx])
        )

class DeepDeterministicPolicyGradient:

    def __init__(self, environ='HalfCheetah-v2', **kwargs):

        self.env = gym.make(environ)
        # self.env.seed(kwargs.get('seed', np.random.randint(1e9)))
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.lower_bounds = self.env.action_space.low[0]
        self.upper_bounds = self.env.action_space.high[0]

        self.memory = ExperienceBuffer(
            kwargs.get('buffer_size', 50000),
            self.num_actions,
            self.num_states
        )

        self.polyak = kwargs.get('polyak', 0.995)
        self.gamma = kwargs.get('gamma', 0.99)

        # Q model
        self.critic_model = CriticModel(
            self.num_states,
            self.num_actions,
            kwargs.get('lr_critic', 0.002),
            [16, 32],
            [32],
            [256, 256],
            'relu',
            'relu',
            'relu'
        )
        self.target_critic = CriticModel(
            self.num_states,
            self.num_actions,
            kwargs.get('lr_critic', 0.002),
            [16, 32],
            [32],
            [256, 256],
            'relu',
            'relu',
            'relu'
        )

        # policy model
        self.actor_model = ActorModel(
            self.num_states,
            self.num_actions,
            [256, 256],
            kwargs.get('lr_actor', 0.001),
            self.upper_bounds,
            'relu',
            'tanh'
        )

        self.target_actor = ActorModel(
            self.num_states,
            self.num_actions,
            [256, 256],
            kwargs.get('lr_actor', 0.001),
            self.upper_bounds,
            'relu',
            'tanh'
        )

        # Sync weights of model and target
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
    
    def policy(self, state, noise_std):
        action = tf.squeeze(self.actor_model.predict(state))
        noise = np.random.normal(0, noise_std, action.shape)
        # Add noise to action
        action = action.numpy() + noise
        # Fix all values within bounds
        action = np.clip(action, self.lower_bounds, self.upper_bounds)

        return [np.squeeze(action)]

    @tf.function
    def polyak_average_update_critic(self):
        target_weights = self.target_critic.model.variables
        critic_weights = self.critic_model.model.variables
        for w_t, w in zip(target_weights, critic_weights):
            w_t.assign(self.polyak*w_t + (1-self.polyak)*w)
    
    @tf.function
    def polyak_average_update_actor(self):
        target_weights = self.target_actor.model.variables
        actor_weights = self.actor_model.model.variables
        for w_t, w in zip(target_weights, actor_weights):
            w_t.assign(self.polyak*w_t + (1-self.polyak)*w)

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def compute_targets(self, reward, next_state, done):
        target_action = self.target_actor.predict(next_state, training=True)
        q_next = self.target_critic.predict(
            [next_state, target_action], training = True
        )
        return reward + self.gamma*(1-done)*q_next

    @tf.function
    def train(self):
        
        # Sample a batch from buffer
        states, actions, rewards, next_states, done_flags = \
            self.memory.get_batch(self.batch_size)
        
        # Compute target values
        # targets = self.compute_targets(rewards, next_states, done_flags)

        # https://keras.io/examples/rl/ddpg_pendulum/
        with tf.GradientTape() as tape:
            # Compute target values
            target_actions = self.target_actor.predict(next_states, training=True)
            q_next = self.target_critic.predict(
                [next_states, target_actions], training=True
            )
            target_q = rewards + self.gamma*q_next#*(1-done_flags)
            # target_q = self.compute_targets(rewards, next_states, done_flags)
            # Get model prdictions from critical model
            pred_q = self.critic_model.predict(
                [states, actions], 
                training=True
            )
            # Calculate the loss
            critic_loss = tf.math.reduce_mean(
                tf.math.square(target_q - pred_q)
            )
        
        critic_grad = tape.gradient(
            critic_loss, 
            self.critic_model.model.trainable_variables
        )
        self.critic_model.optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model.predict(states, training=True)
            pred_q = self.critic_model.predict(
                [states, actions], 
                training=True
            )
            # Caclculate the loss. The minus sign indicates that we effectively 
            # perform gradient `ascent`.
            actor_loss = -tf.math.reduce_mean(pred_q)
        
        actor_grad = tape.gradient(
            actor_loss, 
            self.actor_model.model.trainable_variables
        )
        self.actor_model.optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.model.trainable_variables)
        )


    def fit(self, max_steps, noise, burn_in=64, batch_size=64, verbose=True, 
            render_every=100):
        
        self.batch_size = batch_size
        self.total_training_reward = []
        self.avg_training_reward = []

        if verbose:
            pbar = tqdm(range(int(max_steps)))
        else:
            pbar = range(int(max_steps))
        
        for epoch in pbar:

            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                
                if verbose and epoch != 0 and epoch%render_every==0:
                    self.env.render()

                tensor_state = tf.expand_dims(tf.convert_to_tensor(state), 0) 
                # Select action using policy
                action = self.policy(tensor_state, noise)

                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # accumulate experience
                self.memory.add(state, action, reward, next_state, done)
                
                # if self.memory.get_size() > batch_size: # some condition on training
                    
                    # Train one batch
                self.train()
                # update the target models
                self.update_target(
                    self.target_actor.model.variables, 
                    self.actor_model.model.variables, 
                    0.005
                )
                self.update_target(
                    self.target_critic.model.variables, 
                    self.critic_model.model.variables, 
                    0.005
                )
                    # self.polyak_average_update_critic()
                    # self.polyak_average_update_actor()
                    
                if done:
                    break

                state = next_state

            self.total_training_reward.append(float(episode_reward))

            if verbose:
                pbar.set_postfix({
                    'Rolling reward': '{:.1f}'.format(
                        np.mean(self.total_training_reward[-40:])
                    )
                })
            
            self.avg_training_reward.append(np.mean(self.total_training_reward[-40:]))

        import matplotlib.pyplot as plt

        plt.plot(self.avg_training_reward)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.show()
