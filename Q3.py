import numpy as np
import data_collector

def data_collection(car_env, N=100000):

    collector = data_collector.DataCollector(car_env)
    states, actions, rewards, next_states, done_flags = collector.collect_data(N)

    mean = np.mean(next_states, axis=0)
    std = np.std(next_states, axis=0)
    print('The mean position: ' + str(mean[0]))
    print('The STD of the position: ' + str(std[0]))
    print('The mean speed: ' + str(mean[1]))
    print('The STD of the speed: ' + str(std[1]))
