python -m src.train_agents --name Agent1 --episodes 1000000 --gamma 0.001 --epsilon_high 1.0 --epsilon_low 0.01 --size 3
python -m src.train_agents --name Agent2 --episodes 1000000 --gamma 0.1 --epsilon_high 1.0 --epsilon_low 0.01 --size 3
python -m src.train_agents --name Agent3 --episodes 1000000 --gamma 0.9 --epsilon_high 1.0 --epsilon_low 0.01 --size 3

python -m src.train_agents --name Agent4 --episodes 1000000 --gamma 0.01 --epsilon_high 1.0 --epsilon_low 0.001 --size 3
python -m src.train_agents --name Agent5 --episodes 1000000 --gamma 0.01 --epsilon_high 1.0 --epsilon_low 0.01 --size 3
python -m src.train_agents --name Agent6 --episodes 1000000 --gamma 0.01 --epsilon_high 1.0 --epsilon_low 0.1 --size 3


python -m src.train_agents --name Agent1_4x4 --episodes 1000000 --gamma 0.001 --epsilon_high 1.0 --epsilon_low 0.01 --size 4
python -m src.train_agents --name Agent2_4x4 --episodes 1000000 --gamma 0.1 --epsilon_high 1.0 --epsilon_low 0.01 --size 4
python -m src.train_agents --name Agent3_4x4 --episodes 1000000 --gamma 0.9 --epsilon_high 1.0 --epsilon_low 0.01 --size 4

python -m src.train_agents --name Agent4_4x4 --episodes 1000000 --gamma 0.01 --epsilon_high 1.0 --epsilon_low 0.001 --size 4
python -m src.train_agents --name Agent5_4x4 --episodes 1000000 --gamma 0.01 --epsilon_high 1.0 --epsilon_low 0.01 --size 4
python -m src.train_agents --name Agent6_4x4 --episodes 1000000 --gamma 0.01 --epsilon_high 1.0 --epsilon_low 0.1 --size 4
