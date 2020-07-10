# tetris_gym_env
openAI gym environment and how I trained the model used in challenge AI mode [here](https://github.com/froprintoai/tetris).

# how to install tetris environment.
```
pip install -e gym-tetris
```

# how to test your env
After modifiying [gym-tetris/gym_tetris/envs/tetris_env.py](https://github.com/froprintoai/tetris_gym_env/blob/master/gym-tetris/gym_tetris/envs/tetris_env.py),
such as changing reward systems defined in step method, you can test your environment graphically by executing the following code.
```
python env_test.py
```

# how I trained tetris AI
What I trained in train.py and model.py is the state value function, which takes as inputs the field comibined with next minos, a current mino, and a holding mino.

Loss function = MSE(v_net(state), (reward + v_net(next state)))

The agent always pick the action which maximizes v_net(next state) + reward.
