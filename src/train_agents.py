from src.agent import Agent
from src.environment import Environment


def main():
    num_episodes = 1000000

    agent = Agent(state_size=3, action_size=9, num_episodes=num_episodes)
    env = Environment(size=3)

    for episode in range(num_episodes):
        episode_memory = []
        env.reset()
        state = env.get_state()['state']
        board = env.get_state()['board']
        turn = env.get_state()['turn']

        while not env.is_done():
            action, action_square = agent.act(board, state, turn)

            env.update(action_square)

            next_state = env.get_state()['state']

            episode_memory.append([state, action, next_state])

            state = next_state

        agent.update(env.get_state()['reward'])

        if episode % 100000 == 0:
            try:
                agent.print_metrics()
            except:
                pass

if __name__ == "__main__":
    main()