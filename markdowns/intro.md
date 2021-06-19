# Introduction

## Overview: 
There is a lot of documentation in Internet about AlphaZero, I won't repeat it. AZ bots get their strength from selfplay, thanks to the statistical nature of MCTS. MCTS rollouts brings information from future turns and give win probabilities, as well as average scoring. These rollouts reinforce the NN model, so on succesive selfplay+train cycles the NN gives better values. Many papers state that doubling the MCTS rollout count per turn (800 to 1600) can increase winrate noticeably.

This is the design used on CGZero (this AlphaZero implementation).

![CGZero Diagram](/images/CGZero_Diagram.png)


## Modules:
The implementation consists on several parts:

- A game simulator. It's important to have an accurate simulation, It must simulate a turn and give all the legal moves for the current player. Written in C++
- A modified MCTS tree, that uses the NN as evaluation and exploration. Written in C++
- Different workers: Selfplay (play games and save states as samples), Pitplay (get winrates between generations), Sampler (prepare samples for training), submit (to send it to CG), written in C++.
- A trainer. Written in Python, it uses Jupyter Notebook and uses Tensorflow as the ML framework.
- A NN model. This is the definition of the Neural Network that will be used both in the C++ and the Python trainer. It has layers (Inputs, Dense Layers, and outputs).
- Some utility to pack the binary and model, to send it to CG for submitting. I have it in C#.

In my implementation I have 5 main files:

- `CGZero.cpp`: It does all the C++ roles except the Sampler.
- `NNSampler.cpp`: Sampler Worker. Takes all samples, and average unique gamestates (it reduces samples size count). Right now is game agnostic. It takes inputs+outputs+count from a binary float file.
- `NN_Mokka.h`: NN Inference engine.
- `Train.ipynb`: Jupyter Notebook that does all the training.
- `ENCODER16k.cs`: C# utility. It packs N files and creates a text file that can be pasted in CG IDE. It can compress up to 60%. 


## CGZero Pipeline:

- Define the NN model, both in C++ an Python. Ensure that both codes give the same prediction from the same input. Ensure that the weight fits on the 100KB limit. Recompile the C++ code as it's hardcoded.
- All other steps are completely done at `Train.ipynb` notebook.
- Load parameters for training (number of threads, number of matches per cycle, training parameters, etc)
- Resume previous training state, or create a generation 0 initial state. The trainer will have 2 best models, this is to avoid overfitting against a single model. At any point I have `best1`/`best2`/`candidate` models. All of them are saved on disk.
- Start the training cycle, the whole training is sequential but it uses threads to improve game generation.
   - Generate self-plays. It can be between `best1`/`best2` or a random generation. The percentage of matches on each option is defined based on the previous winrate (i.e. if I lose a lot against `best2`, I'll try to focus on creating replays for `best2`).
   - Use Sampler to get a random subset of samples. The sampler has intelligence, it averages the samples if they have the same gamestate. I.e. the initial gamestate will be repeated in all games, so the sampler will average all training values to create a single sample for that gamestate. It also adds some decay to older generations, to give more importance to samples of recent generations. Generation 0 is too random, I try to remove its samples as soon as possible.
   - Read the samples and train in Tensorflow. I don't do minibatches, but I guess it could be better.
   - Save this `candidate` generation as `gen<generation>.w32`(weights only for C++ code) and as `gen<generation>.h5` (for Tensorflow, to resume training). 
   - Pit play `candidate` against `best1` and `best2`, get winrates. If winrate `candidate` vs `best1`> 55%, then update best models. `best1` and `best2` are always different.
   - Increase Generation (generation++)