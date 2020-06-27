import numpy as np
import matplotlib.pyplot as plt
from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
from lspi import compute_lspi_iteration


def run_lspi(samples_to_collect, seed):

    number_of_kernels_per_dim = [12, 10]
    gamma = 0.99
    w_updates = 20
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000
    np.random.seed(seed)

    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print(f'success rate {data_success_rate}')
    # standardize data
    data_transformer = DataTransformer()
    data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    states = data_transformer.transform_states(states)
    next_states = data_transformer.transform_states(next_states)
    # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # encode all states:
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    # set a new linear policy
    linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
    # but set the weights as random
    linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
    # start an object that evaluates the success rate over time
    evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
    success_per_iter = [evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)]
    for lspi_iteration in range(w_updates):
        print(f'starting lspi iteration {lspi_iteration}')

        new_w = compute_lspi_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        )
        norm_diff = linear_policy.set_w(new_w)
        success_per_iter.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
        if norm_diff < 0.00001:
            break
    print('done lspi')

    return success_per_iter

def lspi_eval(samples_to_collect, seeds, save=True):
    success_per_iter = []
    max_len = 0
    for seed in seeds:
        res = run_lspi(samples_to_collect, seed)
        max_len = max(max_len, len(res))
        success_per_iter.append(res)

    for arr in success_per_iter:
        while len(arr) < max_len:
            arr.append(arr[-1])

    avg_success_per_iter = np.mean(np.asarray(success_per_iter), axis=0)
    if save:
        np.savez('Answers/data_Q3_5', data=avg_success_per_iter)
        plt.figure()
        plt.plot(range(len(avg_success_per_iter)), avg_success_per_iter)
        plt.xlabel("Iteration")
        plt.ylabel("Success Rate")
        plt.title("Mean Success Rate as a Function of Iteration")
        plt.grid()
        plt.savefig("Answers/Q3_5")
        plt.close()
    else:
        return avg_success_per_iter

def lspi_data_depandence(samples_to_collect_arr, seeds):
    success_per_iter_per_samples = []
    for samples_num in samples_to_collect_arr:
        success_per_iter_per_samples.append(lspi_eval(samples_num, seeds, False))

    np.savez('Answers/data_Q3_6', data1=np.asarray(success_per_iter_per_samples[0])
                                , data2=np.asarray(success_per_iter_per_samples[1])
                                , data3=np.asarray(success_per_iter_per_samples[2]))
    # data = np.load('Answers/data_Q3_6.npz')
    # success_per_iter_per_samples = []
    # success_per_iter_per_samples.append(data['data1'])
    # success_per_iter_per_samples.append(data['data2'])
    # success_per_iter_per_samples.append(data['data3'])
    fig, ax = plt.subplots()
    line1, = ax.plot(range(len(success_per_iter_per_samples[0])), success_per_iter_per_samples[0],
                     label='samples = ' + str(samples_to_collect_arr[0]), linewidth=1, color='black')
    line2, = ax.plot(range(len(success_per_iter_per_samples[1])), success_per_iter_per_samples[1],
                     label='samples = ' + str(samples_to_collect_arr[1]), linewidth=1, color='red')
    line3, = ax.plot(range(len(success_per_iter_per_samples[2])), success_per_iter_per_samples[2],
                     label='samples = ' + str(samples_to_collect_arr[2]), linewidth=1, color='blue')

    plt.xlabel('Iteration')
    plt.ylabel("Success Rate")
    plt.title("Mean Success Rate as a Function of Iteration")
    plt.grid()
    ax.legend()
    plt.savefig('Answers/Q3_6')
    plt.close()


if __name__ == '__main__':
    samples_to_collect_arr = [10000, 100000, 150000]
    seeds = [123, 234, 345]
    lspi_eval(samples_to_collect_arr[1], seeds)
    # lspi_data_depandence(samples_to_collect_arr, seeds)



