This folder contains Python scripts for my Deep Q-learning Network learning.

Using OpenAI gym, the environment being played is Atari Breakout.
Here, I used Tensorflow 2, but with TF graphs created using v1 backward compatibility. This sped up training as compared to using Keras.

`dqn_breakout_train.py` is used to train the DQN.
`dqn_breakout_play.py` is used to test the trained DQN.
`tf_dqn_weights.npz` is a numpy file containing the weights of each of the DQN layers.
`attempt2.mp4` is a video of the trained DQN playing Breakout.
