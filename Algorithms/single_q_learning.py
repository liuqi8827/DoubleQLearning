from collections import defaultdict
from tqdm import tqdm as _tqdm
import numpy as np

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SingleQLearning(object):

    def __init__(self):
        pass    

    class EpsilonGreedyPolicy(object):
        """
        A simple epsilon greedy policy.
        """
        def __init__(self, Q, epsilon):
            self.Q = Q
            self.epsilon = epsilon
        
        def sample_action(self, obs):
            """
            This method takes a state as input and returns an action sampled from this policy.  

            Args:
                obs: current state

            Returns:
                An action (int).
            """
            
            action_values = self.Q[obs]
            if (np.random.uniform(0, 1) <= self.epsilon):
                action = np.random.randint(0, len(action_values))
            else:
                action = np.argmax(action_values)
                            
            return action

    def single_q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy
        
        Args:
            env: OpenAI environment.
            policy: A behavior policy which allows us to sample actions with its sample_action method.
            Q: Q value function
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            
        Returns:
            A tuple (Q, stats).
            Q is a numpy array Q[s,a] -> state-action value.
            stats is a list of tuples giving the episode lengths and returns.
        """
        
        # Keeps track of useful statistics
        stats = []
        
        for i_episode in tqdm(range(num_episodes)):
            i = 0
            R = 0
            
            state = env.reset()
            
            while True:
                
                action = policy.sample_action(state)
            
                transition = env.step(action)
                next_state = transition[0]
                reward = transition[1]
                
                max_action = np.argmax(Q[next_state])
                Q[state, action] = Q[state, action] + alpha * (reward + discount_factor * 
                                                            Q[next_state, max_action] - Q[state, action])
                
                state = next_state
                
                R += reward
                i += 1
                
                done = transition[2]
                if done:
                    break
            
            stats.append((i, R))
        episode_lengths, episode_returns = zip(*stats)
        return Q, (episode_lengths, episode_returns)