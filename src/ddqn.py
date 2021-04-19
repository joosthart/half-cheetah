

from tqdm import tqdm
import gym
import numpy as np
import tensorflow as tf

from src.models import DenseModel

SEED = 42

class ExperienceBuffer:
    """Experience Buffer used for experience replay by DeepQLearning
    """
    def __init__(self, max_buffer_size, obs_space):
        self.max_buffer_size = int(max_buffer_size)
        self.obs_space = int(obs_space)

        self.states = np.zeros((self.max_buffer_size, self.obs_space))
        self.actions = np.zeros(self.max_buffer_size, dtype=int)
        self.rewards = np.zeros(self.max_buffer_size)
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

        self.size = max(self.size, self.row)
        self.row = (self.row + 1) % self.max_buffer_size
    
    def get_size(self):
        """Get the current size of the buffer
        """
        return self.size

    def get_batch(self, batch_size):
        """Get a random batch from the buffer
        """
        if batch_size > self.get_size:
            raise ValueError(
                'Batch size is too large. Batch size should be less than or '
                'equal to current buffer size.'
            )

        idx = np.random.choice(
            min(self.size, self.max_buffer_size), batch_size, replace=False
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

        self.env.seed(kwargs.get('seed', np.random.randint(1e9)))

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.memory = ExperienceBuffer(
            kwargs.get('buffer_size', 1e6),
            self.num_states
        )

        self.polyak = None
        
        self.target_model = None
        self.policy = None
        

    def compute_targets(self, reward, next_state, done):
        q_next = self.target_model(next_state, self.policy(next_state))
        return reward + self.gamma*(1-done)*q_next

    def update_policy(self):
        pass

    def update_q_function(self):
        pass

    def update_target_model(self):
        target_weights = self.target_model.get_weights()
        train_weights = self.train_model.get_weights()
        self.target_model.set_weights(
            self.polyak*target_weights + (1-self.polyak)*train_weights
        )

    def get_action(self, state):
        pass

    def train(self):
        
        
        states, actions, rewards, next_states, done_flags = \
            self.memory.get_batch(self.batch_size)
        
        targets = self.compute_targets(rewards, next_states, done_flags)

        self.model.train(states, targets)


    def fit(self, max_steps, verbose=True):
        
        self.total_training_reward = []
        self.total_training_loss = []

        if verbose:
            pbar = tqdm(range(max_steps))
        else:
            pbar = range(max_steps)
        
        state = self.env.reset()
        for epoch in pbar:
            # Select action
            action = self.get_action(state)

            # Execute action
            next_state, reward, done, _ = self.env.step(action)

            # accumulate experience
            self.memory.add(state, action, reward, next_state, done)

            if done:
                state = self.env.reset()
            else:
                state = next_state
            
            if True: # some condition on training
                for i in []: # some number of updates
                    # Train one batch
                    self.train()
                    # update the target models
                    self.update_target_model()





