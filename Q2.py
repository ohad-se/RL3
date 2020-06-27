from mountain_car_with_data_collection import MountainCarWithResetEnv
import radial_basis_function_extractor as rbfe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

def preliminary(car_env):
    n_pos = 100
    n_speed = 100
    min_pos = car_env.min_position
    max_pos = car_env.max_position
    max_speed = car_env.max_speed
    position = np.linspace(min_pos, max_pos, n_pos)
    speed = np.linspace(-max_speed, max_speed, n_speed)

    def get_features():
        number_of_kernels_per_dim = [12, 10]
        extractor = rbfe.RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        states = [x for x in itertools.product(position, speed)]
        features = np.asarray(extractor.encode_states_with_radial_basis_functions(states))
        changes = np.max(features,axis=0) - np.min(features,axis=0)
        max_changing_features_index = np.argsort(changes)

        feature_1 = np.reshape(features[:, max_changing_features_index[-1]], (n_pos, n_speed))
        feature_2 = np.reshape(features[:, max_changing_features_index[-2]], (n_pos, n_speed))

        return feature_1, feature_2

    def plot_fiture(feature, num_of_feature):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xs, ys = np.meshgrid(position, speed)

        surf = ax.plot_surface(xs.T, ys.T, feature, cmap=cm.jet, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('Position', fontsize=10, rotation=150)
        ax.set_ylabel('Speed', fontsize=10)
        ax.set_zlabel('Value', fontsize=10, rotation=60)
        ax.set_title('Feature ' + str(num_of_feature))
        plt.savefig('Answers/Q2_features_' + str(num_of_feature) + '.png')
        # plt.show()

    feature_1, feature_2 = get_features()
    plot_fiture(feature_1, 1)
    plot_fiture(feature_2, 2)

if __name__ == '__main__':
    car_env = MountainCarWithResetEnv()
    preliminary(car_env)

