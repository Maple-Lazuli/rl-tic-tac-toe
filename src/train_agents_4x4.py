from src.agent import Agent
from src.environment import Environment


def main(num_episodes=1, lr=0.01, epsilon=(0.01, 1.0), name='Agent'):

    agent = Agent(state_size=3, action_size=16, name=name, lr=lr, epsilon=epsilon)
    env = Environment(size=4)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()['state']
        board = env.get_state()['board']
        turn = env.get_state()['turn']

        while not env.is_done():
            action, action_square = agent.act(board, state, turn)

            env.update(action_square)

            next_state = env.get_state()['state']

            agent.step(state, action, next_state)

            state = next_state

        agent.update(env.get_state()['reward'])

        if episode % 10 == 0:
            try:
                agent.print_metrics()
            except:
                pass

    agent.save()


if __name__ == "__main__":
    main(num_episodes=10000, lr=0.1, epsilon=(0.01, 1.0), name='Agent2_4x4')


