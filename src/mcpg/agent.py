import mujoco_py
import gym
import tensorflow as tf

import numpy as np
import os

def MDP(lr, gamma, normalize, n_hidden, seed, Number_episodes = 2000):
    """The Main function that trains the Policy of the Monte-Carlo Policy gradient algorithm
    given the Hyperparameters
        Args:
            lr (float): Learning Rate.
            gamma (float): Discount factor.
            normalize (bool): Booling for using Reward Normalization 
            n_hidden (int): Number of hidden layers in the Neural Net.
            Number_episodes (int, optional): Maximal steps of episode. Defaults to 2000.

        Returns:
            list[int]: obtained rewards
            list[int]: running mean over the rewards
            list[int]: obtained loss
            list[int]: running std over the rewards
    """
    #initialize
    seed            = seed
    np.random.seed(seed)
    env             = gym.make('HalfCheetah-v2')
    env.seed(seed)
    
    #initialize policy
    agent  = PolicyLearning(env, lr, gamma, normalize, n_hidden, seed)

    #allocate memory
    running_mean = []
    running_loss = []
    running_std  = [] 
    last_results = []

    #begin training
    for i_episode in range(Number_episodes):

        #reset variables
        s_old        = env.reset() 
        cum_reward   = 0
        run          = 0
        done         = False

        #begin episode run
        while not done:
            #Get and make action from probabillity distribution
            action, prob_dist = agent.make_move(s_old) #take action given state
            
            
            
            s_new, reward, done, info = env.step(action)

            #store run
            agent.save_step(s_old, action, reward, prob_dist)

            #update
            cum_reward += reward
            run        += 1
            s_old       = s_new
        
        #Store episode
        last_results.append(cum_reward)
        
        #Train network
        loss, mean, std = agent.train(last_results)
        running_mean.append(mean)
        running_std.append(std)
        running_loss.append(loss)

        #Visualize
        if i_episode % 10 == 0:
            print("Running lr{}_gamma{}_normalize{}_hiddenlayers{}".format(lr, 
                                                                          gamma,
                                                                          normalize, 
                                                                          n_hidden))            
            print('EPISODE {}'.format(i_episode), 
                '\t| REWARD: {:.0f}'.format(cum_reward),
                '\t| RUNNING MEAN: {:.0f}'.format(mean))

    return last_results, running_mean, running_loss, running_std

class PolicyLearning():
    """REINFORCE agent for solving the CartPole problem.
    """

    def __init__(self, env, lr, gamma, normalize, n_hidden, seed):
        tf.random.set_seed(seed)
        
        #Set (Hyper-)parameters
        self.LR               = lr
        self.GAMMA            = gamma
        self.NORM             = normalize
        self.noise            = 0.1
        self.num_hidden_layers = n_hidden
        self.action_space     = len(env.action_space.sample())
        self.state_space      = env.observation_space.shape[0]

        print(self.action_space, self.state_space)
        
        #Initialize Policy
        self.model = self.make_policy()

        #for storing
        self.episode_states      = []
        self.episode_probs       = []
        self.episode_rewards     = []
        self.episode_action_diff = []
        self.episode = 0

    def make_policy(self):
        """Initialize Policy Network
        
        Returns:
            class: Neural Network model
        """

        #placeholders
        In_states        = tf.keras.layers.InputLayer(self.state_space, 
                                            name = 'input_states')
        Out_actions       = tf.keras.layers.Dense(self.action_space, 
                                            activation = 'tanh', 
                                            name = 'output_actions')

        #initialize neural network
        model = tf.keras.Sequential()
        model.add(In_states)

        for idx in range(self.num_hidden_layers):
            model.add(
                tf.keras.layers.Dense(  
                    256,
                    activation= 'relu',
                    kernel_initializer='random_normal',
                    use_bias=True,
                    name='dense{}'.format(idx+1)
                )
            )

        #Create Output layer
        model.add(Out_actions)

        model.compile(
            optimizer= tf.keras.optimizers.Adam(self.LR),
            loss="categorical_crossentropy"
        )

        return model
    
    def make_move (self, S):
        """Perform action based on the probabillity distribution
        
        Returns:
            list: contain action taken [zero or one] 
            list: contain probabillity of action taken [prob, prob]
        """
       
        #get action probabillities
        probs = self.model.predict(S.reshape([1, self.state_space])).flatten()
        
        #added Noise to action to get out of minimia
        action = np.random.normal(probs, self.noise, size = len(probs))
        
        action = np.clip(action, -1, 1)        
        return action, probs

    def save_step(self, S, A, R, P):
        """Store the state, action, reward, and Prob. of the action """
        
        action_ = A
        self.episode_action_diff.append(action_ - P)
        self.episode_probs.append(P)
        self.episode_rewards.append(R)
        self.episode_states.append(S)

    def train(self, last_results):
        """Train the model with the REINFORCE loss function.
        
        Args:
            last_results(array): array of all cumulative rewards. 
        
        Returns:
            flaot: loss value of the model with it's new paramters.
            flaot: running mean over last 100 cumulative rewards.
            float: running std over last 100 cumulative rewards.
        
        """
        
        #initialize model distribution parameters
        states        = np.vstack(np.array(self.episode_states))
        action_diff   = np.vstack(np.array(self.episode_action_diff))            
        state_rewards = self.get_reward_trace()   

        action_diff   = action_diff * state_rewards.reshape(-1,1)
        action_diff   = np.vstack([action_diff])+self.episode_probs
        
        #perform update
        loss = self.model.train_on_batch(states, action_diff)

        #calculate mean and std
        if len(last_results) < 100:
            mean = np.mean(last_results)
            std  = np.std(last_results) 
        else:
            mean = np.mean(last_results[-100:])
            std  = np.std(last_results[-100:])

        #reset
        self.episode_states      = []
        self.episode_probs       = []
        self.episode_rewards     = []
        self.episode_action_diff = []

        self.episode += 1

        return loss, mean, std

    def get_reward_trace(self):
        """Perform the gradient descent step
        
        Returns:
            array: array of the discounted rewards per time step of a single roll
            out
        """
        
        #allocate memory
        grad   = np.zeros_like(self.episode_rewards)
        epochs = range(len(self.episode_rewards))
        R      = 0

        #perform discounted rewards
        for t in reversed(epochs):
            R       = self.episode_rewards[t] + self.GAMMA*R
            grad[t] = R
        
        # Perform Baseline Subtraction. 
        if self.NORM == True:
            grad = grad - np.mean(grad)
            grad = grad / np.std(grad)
            return grad
        else:
            return grad
