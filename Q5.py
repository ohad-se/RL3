from Q4 import Q_learning_eval

if __name__ == '__main__':
    max_episodes = 200
    seeds = [123]
    epsilons = [1]
    epsilon_decrease = 1
    Q_learning_eval(seeds, epsilons, max_episodes=max_episodes, epsilon_decrease=epsilon_decrease,
                                start_at_bottom=True, save=False)