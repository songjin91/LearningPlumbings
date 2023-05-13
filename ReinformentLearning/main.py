import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import random

from models import Actor, Critic
from utils import *

seed = 123
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# Parameters
SEED = 123               # random seed
LR = 0.0005              # learning rate
GAMMA = 0.99             # discount factor for future rewards
MAX_STEP = 40            # maximum number of steps in an episode before termination
MAX_EPISODE = 10000      # maximum number of episodes

# Initialize a global copy of actor-critic shared_model.
model_a = Actor()
model_c = Critic()

def train(pid, shared_actor, shared_critic, learning_rate): 
    torch.manual_seed(SEED+pid)

    # Initialize a local copy of actor-critic shared_model.
    local_actor = Actor()
    local_critic = Critic()

    # Instantiate optimization algorithm.
    optimizerA = torch.optim.Adam(shared_actor.parameters(), lr=learning_rate)
    optimizerC = torch.optim.Adam(shared_critic.parameters(), lr=2*learning_rate)

    episode = 0
    episode_rewards = []
    
    while True:
        episode += 1

        # initial random state
        if episode % 2 == 1:
            state = random_graph(n_max=10)
            for _ in range(15):
                node_index = np.random.randint(len(state.x))
                move_type = np.random.randint(1, 4)
                updown = 1
                sign = random.choice([1, -1])
                _, state = neumann_move(state, node_index, move_type, updown, sign)
        else:
            n_append = np.random.randint(1,4)
            state = random_graph(n_max=10-n_append-1)
            for _ in range(n_append):
                state = append_twos(state)
        
        init_n = 5* len(state.x) + torch.sum(torch.abs(state.x))
        min_n = init_n
        done = False
        
        # Run episode
        step = 1        
        step_rewards = []
        step_log_probas = []
        step_state_values = []
        # Load the model parameters from global copy.
        local_actor.load_state_dict(shared_actor.state_dict())
        local_critic.load_state_dict(shared_critic.state_dict())

        while not done:
            # Get action from the current policy
            action_proba, log_proba = local_actor(state)
            state_value = local_critic(state)

            # Randomly sample an action from the probability distribution
            action = choose_action(action_proba)
            next_state = state
            # Take action on the environment and get reward, next_state
            if action[2] == 1: # blow-up
                move_done, next_state = neumann_move(next_state, node_index=action[0], type=action[1], 
                                    updown=action[2], sign=action[3])
            else: # blow-down
                for k in range(3):
                    move_done, next_state = neumann_move(next_state, node_index=action[0], type=k+1, 
                                    updown=action[2], sign=action[3])
                    if move_done:
                        break
            next_n = 5* len(next_state.x) + torch.sum(torch.abs(next_state.x))
            if next_n < min_n:
                min_n = next_n
                reward = init_n - next_n
            elif not move_done:
                reward = -10                
            else:
                reward = min_n - next_n
            
            if len(next_state.x) <= 10:
                done = True
                
            state = next_state
            step += 1

            # TODO: Check for max episode length
            if step > MAX_STEP:
                done = True

            # Store data for loss computation
            step_rewards.append(reward)
            step_log_probas.append(log_proba[action[4]])
            step_state_values.append(state_value)

        # Calculate loss over the trajectory
        R = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        policy_loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        value_loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        for idx in reversed(range(len(step_rewards) - 1)):
            R = GAMMA * R + step_rewards[idx]
            advantage = R - step_state_values[idx]
            value_loss = value_loss + advantage.pow(2)
            policy_loss = policy_loss - step_log_probas[idx]*advantage

        # Reset gradient
        optimizerA.zero_grad()
        optimizerC.zero_grad()

        # Calculate gradients by combining actor and critic loss.
        value_loss.backward(retain_graph=True)
        policy_loss.backward(retain_graph=True)

        episode_rewards.append(sum(step_rewards[:-1]))

        # Copy the gradients on local_model to the shared_model.
        for local_param, global_param in zip(local_actor.parameters(),
                                             shared_actor.parameters()):
            global_param.grad = local_param.grad
        for local_param, global_param in zip(local_critic.parameters(),
                                             shared_critic.parameters()):
            global_param.grad = local_param.grad

        # Backprop the gradients.
        optimizerA.step()
        optimizerC.step()
        
        # Log metrics.
        if episode % 100 == 0:
            episode_rewards_ = episode_rewards[-100:]
            avg_reward = sum(episode_rewards_)/len(episode_rewards_)           
            print(f"EPISODE: {episode}")
            print(f"{pid} - Average Reward: {avg_reward}")
            stat_file.close()

        if episode > MAX_EPISODE:
            plt.figure()
            plt.title("Average reward")
            plt.plot(range(0, episode), episode_rewards)
            plt.xlabel("episode")
            plt.ylabel("Average Reward per 100 episode")
            plt.savefig("./rewards.png")
            break
    return 0


print("*********START TRAINING************")


if __name__ == '__main__':
    # Set the method to start a subprocess.
    # Using `spawn` here because it handles error propagation across multiple
    # sub-processes. When any of the subprocess fails it raises exception on
    # join.
    mp.set_start_method('spawn')

    # Initialize actor-critic model here. The parameters initialized here
    # will be shared across all the sub-processes for both policy and value
    # network.
    model_a = Actor()
    model_a.share_memory()
    model_c = Critic()
    model_c.share_memory()

    processes = []
    for pid in range(8):
        # Start process
        p = mp.Process(target=train, args=(pid, model_a, model_c, LR,))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()    

    print("*********END OF TRAINING************")
        
    torch.save(model_a.state_dict(),'model_a.pth')
    torch.save(model_c.state_dict(),'model_c.pth')
    print("### SAVED ACTOR AND CRITIC MODELS. ###")
