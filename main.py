



from src.ddqn import DeepDeterministicPolicyGradient


if __name__ == '__main__':
    estimator = DeepDeterministicPolicyGradient()
    estimator.train(1000, render_every=100, save='models/pendulum')

    estimator.simulate('models/pendulum', 5)