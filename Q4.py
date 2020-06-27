import numpy as np
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor

from q_learn_mountain_car import Solver
from q_learn_mountain_car import run_episode

def run_Q_learning(seed, epsilon_current=0.1, max_episodes=10000):
    env = MountainCarWithResetEnv()
    np.random.seed(seed)
    env.seed(seed)

    gamma = 0.99
    learning_rate = 0.01
    epsilon_decrease = 1.
    epsilon_min = 0.05

    solver = Solver(
        # learning parameters
        gamma=gamma, learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
    )

    bottom_state = np.asarray([-0.5, 0])
    bottom_state_val = []
    success_rates = []
    episodes_gain = []
    episodes_bellman_err = []
    for episode_index in range(1, max_episodes + 1):
        episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)
        episodes_gain.append(episode_gain)
        # reduce epsilon if required
        epsilon_current *= epsilon_decrease
        epsilon_current = max(epsilon_current, epsilon_min)
        episodes_bellman_err.append(mean_delta)
        bottom_state_features = solver.get_features(bottom_state)
        bottom_state_max_action = solver.get_max_action(bottom_state)
        bottom_state_val.append(solver.get_q_val(bottom_state_features, bottom_state_max_action))

        # print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')

        # termination condition:
        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
            mean_test_gain = np.mean(test_gains)
            success_rates.append(np.mean(np.asarray(test_gains) > -200))
            print(f'tested 10 episodes: mean gain is {mean_test_gain}')
            if mean_test_gain >= -75.:
                print(f'solved in {episode_index} episodes')
                break

    return episodes_gain, success_rates, bottom_state_val, episodes_bellman_err

def Q_learning_eval(seeds, epsilon_current=0.1, max_episodes=10000):
    bottom_state_val = []
    success_rates = []
    episodes_gain = []
    episodes_bellman_err = []
    episodes = max_episodes

    for seed in seeds:
        gain, rate, val, err = run_Q_learning(seed, max_episodes=episodes)
        episodes = min(episodes, len(gain))
        episodes_gain.append(gain)
        success_rates.append(rate)
        bottom_state_val.append(val)
        episodes_bellman_err.append(err)

    for ii in range(len(episodes_gain)):
        episodes_gain[ii] = episodes_gain[ii][:episodes]
        success_rates[ii] = success_rates[ii][:int((episodes+1)/10)]
        bottom_state_val[ii] = bottom_state_val[ii][:episodes]
        episodes_bellman_err[ii] = episodes_bellman_err[ii][:episodes]

    episodes_gain = np.mean(np.asarray(episodes_gain), axis=0)
    success_rates = np.mean(np.asarray(success_rates), axis=0)
    bottom_state_val = np.mean(np.asarray(bottom_state_val), axis=0)
    episodes_bellman_err = np.mean(np.asarray(episodes_bellman_err), axis=0)

    episodes_bellman_err_avg = []
    for ii in range(episodes):
        tmp = np.asarray(episodes_bellman_err[max(0, ii-100):ii])
        episodes_bellman_err_avg.append(np.mean(tmp))

    # data = np.load('Answers/data_Q4_3.npz')
    # episodes_gain = data['episodes_gain']
    # success_rates = data['success_rates']
    # bottom_state_val = data['bottom_state_val']
    # episodes_bellman_err = data['episodes_bellman_err']
    # episodes = len(episodes_gain)

    np.savez('Answers/data_Q4_3', episodes_gain=episodes_gain, success_rates=success_rates
                        , bottom_state_val=bottom_state_val, episodes_bellman_err=episodes_bellman_err
                        ,episodes_bellman_err_avg=episodes_bellman_err_avg)

    x1 = range(1, episodes+1)
    x2 = range(1, len(success_rates)*10 + 1, 10)

    list = [episodes_gain, success_rates, bottom_state_val, episodes_bellman_err_avg]
    titles = ['Total Reward',
              'Performance',
              'Bottom State Value',
              'Total Bellman Error']
    x = [x1, x2, x1, x1]

    for ii, y in enumerate(list):
        plt.figure()
        plt.plot(x[ii], y)
        plt.xlabel("Episode")
        plt.ylabel(titles[ii])
        plt.title(titles[ii] + ' Vs. Training episodes')
        plt.grid()
        plt.savefig("Answers/Q4_3_" + titles[ii].replace(' ', '_'))
        plt.close()

def Q_learning_exploration_eval(seeds, epsilons, max_episodes=10000):

    episodes_gain = []

    for epsilon in epsilons:
        episodes_gain_tmp = []
        episodes = max_episodes
        for seed in seeds:
            gain, rate, val, err = run_Q_learning(seed, epsilon_current=epsilon, max_episodes=episodes)
            episodes = min(episodes, len(gain))
            episodes_gain_tmp.append(gain)

        for ii in range(len(episodes_gain_tmp)):
            episodes_gain_tmp[ii] = episodes_gain_tmp[ii][:episodes]

        episodes_gain.append(np.mean(np.asarray(episodes_gain_tmp), axis=0))

    np.savez('Answers/data_Q4_4', episodes_gain=episodes_gain)

    #
    # # save np.load
    # np_load_old = np.load
    # # modify the default parameters of np.load
    # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    #
    # data = np.load('Answers/data_Q4_4.npz')
    #
    #
    # # restore np.load for future normal usage
    # np.load = np_load_old
    #
    # episodes_gain = data['episodes_gain']

    fig, ax = plt.subplots()
    line = []
    for ii, eps in enumerate(epsilons):
        line.insert(ii, ax.plot(range(len(episodes_gain[ii])), episodes_gain[ii],
                                label='$\\epsilon = $' + str(eps), linewidth=1))

    plt.xlabel('Episode')
    plt.ylabel("Total Reward")
    plt.title("Total Reward Vs. Training episodes")
    plt.grid()
    ax.legend()
    plt.savefig('Answers/Q4_5')
    plt.close()


if __name__ == '__main__':
    max_episodes = 500
    seeds = [123] #, 234, 123]
    epsilons = [1, 0.75, 0.5, 0.3, 0.01]
    # Q_learning_eval(seeds, max_episodes=max_episodes)
    Q_learning_exploration_eval(seeds, epsilons, max_episodes=max_episodes)

