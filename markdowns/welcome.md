# AlphaZero like implementation for Oware Abapa game (www.Codingame.com)

This tutorial can help you implement a working AlphaZero (AZ) like bot  for competing on a limited resources environment (1 vCPU, 100KB of total size and 50ms turn time). Most AZ implementations have many Convolutional/Dense layers, and the model can weight more than 20 MB. This code uses just a tiny Neural Network(NN) of about 80KB in size to fit in the 100KB limit.

See https://github.com/marchete/CGZero for more info and source code.

## Difficulty Level: 
Very Hard. There are many different components that must be tied together.

## Previous Knowledge:
- C++, for the bot part
- Python, for the training part
- Monte Carlo Tree Search
- Experience on AI bot creation (Codingame multiplayer games)

## Estimated Time:
3 hours. Also +5 hours for training the model. Training and doing tests with Neural Networks are time consuming.

## What will I learn?
All the components for an AZ like bot. Please note that this implementation isn't exactly a pure AZ bot, it's tweaked to be much simpler.

## Why should I learn?
In many games AZ bots are the most powerful right now. 
They are interesting because they don't need domain knowledge, they learn from self play.

## Main objectives on this playground

My premises while creating the bot was:
* To be a competitive AI. The bot isn't one of those "it reached 67% winrate vs a random bot!". It's not a fantastic AI but with enough training it reached top 6th at Oware Abapa. That isn't a easy task, top 10 players are the best of the best, high level bots.
* Using a proven Machine Learning (ML) framework for the training part, Tensorflow in my case.
* Don't use domain knowledge, only the Gamestate and valid moves per turn, and a EndGame score (-0.8 or 0.8, with some bonus for early wins and score difference).
* A pure self-learning AI, it only learns by playing against itself (or previous NN versions). For the whole training it never competes against any other AI bot (only itself), neither any kind of expert knowledge, opening books or anything.
* On the documents I'll try to keep it as simple as possible. NN documentation is really hard to understand for some of us. Please keep in mind that this tutorial is aimed for generalistic programmers, not ML experts.
* No external libraries, just C++17 standards. Compilable both in Windows and Linux.
* Fit in 100KB to submit to CG's servers (including binary code and NN weights).
* The code is probably bugged, and not orthodox to Machine Learning standards. But it can be used as an example of different parts that makes a working Alphazero bot.

I'd like to thank CG people on chat (Jacek, Robostac, Wontonimo and others) for helping me with general NN knowledge.

## Disclaimer
The bot is able to learn how to play correctly in about 10 generations (around 4 hrs in a Core-i7 without GPU training). Improved versions of this code was able to reach 5th place with 1hr of training.
