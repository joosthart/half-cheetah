



from src.ddpg.agent import DeepDeterministicPolicyGradient
from src.ddpg.experiments import run_all_experiments as ddpg_experiment_run
from src.mcpg.experiments import run_all_experiments as mcpg_experiment_run

if __name__ == '__main__':

    # polyak()

    # params_Lillicrap = {
    #     'lr_actor': 1e-4,
    #     'lr_critic': 1e-3,
    #     'gamma': 0.99,
    #     'polyak': 0.001,
    #     'batch_size': 64,
    #     'buffer_size': 1e6,
    #     'actor_hidden': [300, 400],
    #     'critic_hidden_state': [300],
    #     'critic_hidden_action': [],
    #     'critic_hidden_common': [400],
    #     'critic_init_min':-3e-4,
    #     'critic_init_max':3e-4

    # }

    
    # estimator.train(150, render_every=1e4, save='models/Lillicrap')

    print('lillicrap_no_action_layer')
    print(50*'-')
    for idx, p in enumerate([0.001, 0.005, 0.01, 0.05]):
        estimator = DeepDeterministicPolicyGradient()
        model = 'models/lillicrap_no_action_layer/polyak={}/'.format(p)
        mean, std = estimator.simulate(model, 100, render=False, verbose=False)

        print(p, ': ', mean, '+/-', std)

    print('lillicrap_action_layer')
    print(50*'-')
    for idx, p in enumerate([0.001, 0.005, 0.01, 0.05]):
        estimator = DeepDeterministicPolicyGradient()
        model = 'models/lillicrap_action_layer/polyak={}/'.format(p)
        mean, std = estimator.simulate(model, 100, render=False, verbose=False)

        print(p, ': ', mean, '+/-', std)
        