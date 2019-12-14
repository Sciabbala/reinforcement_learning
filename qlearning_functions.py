from random import uniform
import numpy as np

def agent_training(env, qtable, total_episodes=50000, max_steps=99, learning_rate=1,
                   gamma=1, epsilon=1, max_epsilon=1, min_epsilon=0.01, decay_rate=0.01):
    
    '''This function will train an agent to play the Taxi-v3 environment
    
    Input:  
        env: the Taxi-v3 environment
        qtable: a qtable with nrows = n of possible game states and ncolums = n of possible actions
        total_episodes: number of iterations for training
        max_steps: number of steps before stopping the training
        learning_rate: affects the effect that rewards have on the learning
        gamma: discount rate
        epsilon: starting exploration rate
        max_epsilon: maximum exploration rate
        min_epsilon: minimum exploration rate
        decay_rate: exploration decay rate
        
    Output:
        trained_qtable: returns trained q-table'''
    
    trained_qtable = qtable.copy()
    
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
    
        for step in range(max_steps):
            exp_exp_tradeoff = uniform(0,1)

            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])

            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            trained_qtable[state, action] = trained_qtable[state, action] + learning_rate * (reward + gamma *
                                        np.max(trained_qtable[new_state, :]) - trained_qtable[state, action])

            state = new_state

            if done:
                break


        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    return trained_qtable


def agent_testing(env, qtable, tot_test_episodes=1000, max_steps=99, render=False):
    '''
    The functions tests an agent in a given environment and with a given qtable and returns the mean score over all
    the tests.
    
    Input:
        env: the Taxi-v3 environment
        qtable: the trained qtable
        tot_test_episode: number of iterations tot test
    '''
    
    # Creating a variable to store the rewards of all test episodes
    rewards = []

    for episode in range(tot_test_episodes):
        # Reset the environment
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        
        if render:
            # Visually separating the renders of the environment states
            print("-------------------------------------------------------------------------------")
            print("Episode number ", episode)

        # Executing actions based on the highest Q-value for each state
        for step in range(max_steps):
            action = np.argmax(qtable[state, :])
            
            if render:
                env.render()
                print(action)
            
            new_state, reward, done, info = env.step(action)

            total_rewards += reward

            # If agent finishes the current game append the score and break the loop
            if done:
                rewards.append(total_rewards)

                break

            state = new_state

    env.close()
    score_over_time = str(sum(rewards)/tot_test_episodes)
    
    return score_over_time