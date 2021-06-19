# Differences between CGZero and Alphazero

## Simple NN Model
Original AlphaZero NN design has many (80+) Convolutional layers, Fully Connected layers, Batch Normalization layers, skip connection layers (adding inputs again after each residual layer) and more. 
 CGZero is a toy model. I haven't implemented neither Convolutional nor Batch Normalization layers. I have like 3 or 4 layers of Fully Connected layers, nothing more.
  Check https://adspassets.blob.core.windows.net/website/content/alpha_go_zero_cheat_sheet.png for some diagram of Alpha Zero networks. That can't fit on normal competitive games, and it won't have enough CPU time to do anything good.

## Synchronous training

AlphaZero used 4 first-generation [TPUs](https://cloud.google.com/tpu/docs/tpus) and 44 [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) cores. My setup is modest, just a single Intel Core i7, training while I do other stuff at the PC. I don't have distributed workers (can be really simple to delegate selfplay workers to external PC's though).
 So all the training pipeline is just linear, one task after another. I use threading to greatly improve selfplay game generation and pitplay winrate results.

## Self-play generation

AlphaZero states 25000 games against itself. I just do around 400-1500 each generation. They always use the best model, I use 3 different options:

1. `Best1` vs `Best1`
2. `Best1` vs `Best2`
3. `Best1 or Best2` vs `random generation` (a recent one)

The amount of the third option is always a 10% of the games to generate. **1** and **2** will have variable percentage, depending on the result of the last training. Having `winrate1` and `winrate2` (winrates of candidate vs each best model) I'll generate more games to the lowest winrate.

## Samples
Alphazero stores `| Inputs | Policy | EndGame Value |` as a sample. They store samples for the last 500k games, I just store last X00k samples (much less games).

My Samples have 4 parts:

    | Inputs | SumPolicy | SumValue | Count |

Samples will be dictionaries with inputs as keys, so I make unique samples and sum the policy and values, and increase the count by 1.  Dividing by `Count` will average samples.
AlphaZero always uses EndGame Value for all samples. I mix what you can read on: 
https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628
In that document they declare **`Z`** and **`Q`** as possible objectives for value training.  **`Z`** is the endgame score (-1, 0 or 1) that will be used on all samples (taking into account that samples from the loser player has -endgameScore). **`Q`** is just the sumValue/visits you get while doing MCTS search (i.e. the mean Score of the root node).
 I use a mix of both.  `SampleValue =  k * Z + (1-k)*Q` . **`k`** will linearly increase each turn, so initial samples have some little EndGame Score (because I think it's too far to declare a position as good or bad), and final moves will have a big percentage of EndGame Score (it's more clear that this position is good or bad).
I had many problems with the sign of the Samples value. I struggled to get the correct sign. It seems naive, but to evaluate a game state the root node have a -sum(childrenValues). This is not what I wanted. If as player 0 I have a sure win (1.0), then all children from root will have 1.0 as mean score. But as the way the score backpropagates the root node will have a -1.0 (it's seen as the enemy POV). I neeeded the score for the player 0, so I tested a lot with signs until I got what I expected.

Alphazero picks random minibatches of 2048 positions from last 500k games. I imagine it's just a random selection of 2048 samples. But according to my samples, without deduplication/averaging of samples you'll end up with a lot of repetitive initial samples, most of them having contradictory training targets.

My subset selection is much bigger, I take around 200k-800k samples from last 500k-N000k samples as a subset. Also the subset selection is weighted. I order samples by `Count`, on descending order.  The 20% of my subset will come from the first 20% of that ordered sample list, the next 20% from the first 40% and so on. This way I ensure that most frequent samples usually appears on the subset. 
Finally I add some decay weight to older samples. I get all files, sortered by name descending (so higher generations are read first). For each file I reduce 0.02 the decay factor (starting with 1.0, limited to 0.7 as min decay). So samples from an old generation are counted but they add less to the final `SumPolicy` and `SumValue`
 
 ## Training
 Alphazero uses minibatches of 2048 samples. I use a big subset with M00k samples, and the training function does N passes (EPOCH between 5 and 20, depending on how much it takes). I do it on a synchronized way.
 AZ do the evaluation of the network each 1000 minisamples, I do after 1 training call (but that call has N passes as EPOCH).

The AZ loss function is `Cross entropy loss` + `Mean squared loss` + `regularisation`. I don't know exactly what that regularisation is, or how to calculate it, so I ignored it.
 Also the `Cross entropy loss` is completely incorrect for my approach. https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy or any other crossentropy loss in tensorflow seems to work for categorization (i.e. they expect only one true value, with a 1.0 on it and the rest with zero). In my tests I was getting huge losses as categorical crossentropy, even with near perfect predictions (0.0003 differences between the prediction and the expected policy values).
 Finally I used https://www.tensorflow.org/api_docs/python/tf/keras/metrics/kl_divergence , that is exactly what I was looking for (a measure of how one probability distribution is different from a second)

 ## Evaluation
 I evaluate against two best models, `best1` and `best2`. When a new candidate have a winrate >55%, `best1`  is passed to `best2`, and the new candidate is promoted as `best1`

 ## Move selection after MCTS Search ended
While training AlphaZero uses some temperature for selection. It seemed obscure and overly complicated.

I went to a simpler mode. I rely on dirichlet noise to have some randomness on visit count. Also for the first 11 turns I have some chance to randomly pick a move, regardless its stats. This random move invalidates the sample generation for that position, to avoid breaking the statistics.
I don't pick the move with most visits. I used visits and score (similar to https://tech.io/playgrounds/55004/best-first-minimax-search-with-uct).  `SelectMoveScore = move's eval value + log(move's visits)`