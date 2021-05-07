
from src.ddpg.agent import DeepDeterministicPolicyGradient

ENVIRONMENTS = [
    'Ant-v2',
    'Hopper-v2',
    'Swimmer-v2',
    'Walker2d-v2',
    'HalfCheetah-v2'
]

BEST_MODEL_PARAMS = {
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'polyak': 0.05,
        'batch_size': 64,
        'buffer_size': 1e6,
        'actor_hidden': [256, 256],
        'critic_hidden_state': [256],
        'critic_hidden_action': [],
        'critic_hidden_common': [256],
        'action_noise': 0.2
}

PARAMS_LILLICRAP_NO_ACTION_LAYER = {
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 1e6,
        'actor_hidden': [256, 256],
        'critic_hidden_state': [256],
        'critic_hidden_action': [],
        'critic_hidden_common': [256],
    }

PARAMS_LILLICRAP_ACTION_LAYER = {
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'gamma': 0.99,
    'batch_size': 64,
    'buffer_size': 1e6,
    'actor_hidden': [256, 256],
    'critic_hidden_state': [256],
    'critic_hidden_action': [128],
    'critic_hidden_common': [256],
}


def polyak_grid_search(
        parameter_sets=[
            PARAMS_LILLICRAP_NO_ACTION_LAYER, 
            PARAMS_LILLICRAP_ACTION_LAYER
        ],
        polyak_grid = [1e-3, 5e-3, 1e-2, 5e-2],
        save=['lillicrap_no_action_layer', 'lillicrap_action_layer'],
    ):

    for p in polyak_grid:
        for params, name in zip(parameter_sets, save):
            base_estimator = DeepDeterministicPolicyGradient(
                polyak=p, **params
            )
            base_estimator.train(
                208, save='models/{}/polyak={}'.format(name,p)
            )

def train_environemnts_different_seeds(
        envs=ENVIRONMENTS, params=BEST_MODEL_PARAMS
    ):
    for env in envs:
        for seed in range(5):
            model = DeepDeterministicPolicyGradient(
                environ=env, seed=seed,**params
            )

            save_dir = 'models/{}/best_parmas/seed={:.0f}'.format(env, seed)

            model.train(
                208, save=save_dir,
            )

def run_all_experiments():
    polyak_grid_search()
    train_environemnts_different_seeds()