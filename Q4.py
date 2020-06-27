import numpy as np
import time

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor

#from q_learn_mountain_car import Solver
import q_learn_mountain_car
from q_learn_mountain_car import run_episode

if __name__ == "__main__":
    env = MountainCarWithResetEnv()
    seed = 123
    # seed = 234
    # seed = 345
    np.random.seed(seed)
    env.seed(seed)

    gamma = 0.99
    learning_rate = 0.01
    epsilon_current = 0.1
    epsilon_decrease = 1.
    epsilon_min = 0.05

    max_episodes = 100000

    solver = q_learn_mountain_car.Solver(
        # learning parameters
        gamma=gamma, learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
    )

    for episode_index in range(1, max_episodes + 1):
        episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)

        # reduce epsilon if required
        epsilon_current *= epsilon_decrease
        epsilon_current = max(epsilon_current, epsilon_min)

        print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')

        # termination condition:
        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
            mean_test_gain = np.mean(test_gains)
            print(f'tested 10 episodes: mean gain is {mean_test_gain}')
            if mean_test_gain >= -75.:
                print(f'solved in {episode_index} episodes')
                break

    run_episode(env, solver, is_train=False, render=True)

# eval_crat = np.mean(np.asarray(test_gains) > -200)
# print('eval {}'.format(eval_crat))
