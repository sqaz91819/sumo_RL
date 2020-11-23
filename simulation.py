import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
from Neuron_Network import TrainModel
# import tensorflow as tf

class Simulation:

    def __init__(self, sumocfg):
        self.gamma = 0.99
        self.max_episode = 1000
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.max_step = 5400
        self.num_actions = 2
        self.sumoBinary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo"
        self.sumoCmd = [self.sumoBinary, "-c", sumocfg + ".sumocfg"]
        self.eps = np.finfo(np.float32).eps.item()
        self.sumo_process("single_route")



    def get_state(self):
        state = np.zeros((1,7))
        traci.simulationStep()
        car_ids = traci.vehicle.getIDList()
        time = 0
        for car_id in car_ids:
            time += traci.vehicle.getAccumulatedWaitingTime(car_id)
            if traci.vehicle.getLaneID(car_id) == "gneE_0":
                pos = traci.vehicle.getLanePosition(car_id)
                if pos / 7 <= 6:
                    state[pos/7] = 1
                else:
                    state[6] = 1


        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        return state, time



    def sumo_process(self):

        for episode in range(self.max_episode):
            state = []
            eipsode_reward = 0
            with tf.GradientTape() as tape:
                
                traci.start(self.sumoCmd)

                for step in range(self.max_step):
                    
                    state, delay = self.get_state()

                    action_probs, critic_value = model(state)
                    self.critic_value_history.append(critic_value[0, 0])

                    action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
                    self.action_probs_history.append(tf.math.log(action_probs[0, action]))

                    state, delay2 = self.get_state()
                    reward = delay - delay2
                    self.rewards_history.append(reward)
                    episode_reward += reward

                traci.close()
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

                returns = []
                discounted_sum = 0
                for r in self.rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.action_probs_history, self.critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        keras.losses.Huber()(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )


                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Clear the loss and reward history
                self.action_probs_history.clear()
                self.critic_value_history.clear()
                self.rewards_history.clear()

                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode))


if __name__ == "__main__":
    t = Simulation("single_route")
    exit()
    