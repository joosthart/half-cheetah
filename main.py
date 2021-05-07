import argparse
import os

from src.ddpg.agent import DeepDeterministicPolicyGradient
from src.ddpg.experiments import run_all_experiments as ddpg_experiment_run
from src.mcpg.experiments import run_all_experiments as mcpg_experiment_run

parser = argparse.ArgumentParser(
    description='MuJoCo Reinforcement learning project.'
)

parser.add_argument(
    '-r', '--run-experiments', 
    default='',
    type=str,
    help='Run experiments. Options: "ALL", "DDPG", "MCPG"'
)

parser.add_argument(
    '-s', '--simulate', 
    action='store_true',
    help='Display simulations of DDPG algorithms on HalfCheetah-v2.'
)

parser.add_argument(
    '-e', '--episodes', 
    default=3,
    type=int,
    help='Number of episodes to run simulation.'
)

args = parser.parse_args()

if __name__ == '__main__':

    if not os.path.exists('models/'):
        os.makedirs('models/')

    if args.simulate:
        model = DeepDeterministicPolicyGradient()
        model.simulate('demo', args.episodes)

    if args.run_experiments and args.run_experiments.lower() == 'all':
        ddpg_experiment_run()
        mcpg_experiment_run()
    elif args.run_experiments and args.run_experiments.lower() == 'ddpg':
        ddpg_experiment_run()
    elif args.run_experiments and args.run_experiments.lower() == 'mcpg':
        mcpg_experiment_run()
    