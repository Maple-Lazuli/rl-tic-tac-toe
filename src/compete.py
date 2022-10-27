import argparse
import random

random.seed(101011)

from src.agent import Agent
from src.environment import Environment


def compete(name1, name2, games=1001):
    agent1 = Agent(3, 9, name=name1)
    agent1.load()
    agent2 = Agent(3, 9, name=name2)
    agent2.load()

    env = Environment(size=3)

    results = {
        'pos1': name1,
        'pos1-wins': 0,
        'pos2': name2,
        'pos2-wins': 0,
    }
    for _ in range(games):
        env.reset()
        if random.randint(0, 1) == 1:
            # Agent 1 goes first
            while not env.is_done():
                state = env.get_state()['state']
                board = env.get_state()['board']
                turn = env.get_state()['turn']
                action, action_square = agent1.perform(board, state, turn)

                env.update(action_square)

                state = env.get_state()['state']
                board = env.get_state()['board']
                turn = env.get_state()['turn']
                action, action_square = agent2.perform(board, state, turn)

                env.update(action_square)
            if env.get_state()['reward'] == 1:
                results['pos1-wins'] += 1
            else:
                results['pos2-wins'] += 1
        else:
            # Agent 2 goes first
            while not env.is_done():
                state = env.get_state()['state']
                board = env.get_state()['board']
                turn = env.get_state()['turn']
                action, action_square = agent2.perform(board, state, turn)

                env.update(action_square)

                state = env.get_state()['state']
                board = env.get_state()['board']
                turn = env.get_state()['turn']
                action, action_square = agent1.perform(board, state, turn)

                env.update(action_square)

            if env.get_state()['reward'] == 1:
                results['pos2-wins'] += 1
            else:
                results['pos1-wins'] += 1

        results['winner'] = name1 if results['pos1-wins'] > results['pos2-wins'] else name2
    return results


def main(flags):

    print(compete(name1=flags.agent1, name2=flags.agent2, games=flags.games))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--agent1', type=str,
                        default="Agent1",
                        help="The name of the second agent")

    parser.add_argument('--agent2', type=str,
                        default="Agent2",
                        help="The name of the second agent")

    parser.add_argument('--games', type=int,
                        default=1001,
                        help="The number of games to compete with")

    flags, unparsed = parser.parse_known_args()

    main(flags)

