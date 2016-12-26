# NN-Gridworld Experiment
<p align="center">
  <img src="../../images/graphs/nn-graph.png", width="50%"/>
</p>

In comparison to the previous [Q-Gridworld](../q-gridworld) experiment, this experiment attempts to solve [Gridworld](../games/gridworld.py) using a Neural Network to estimate the Q-values instead of storing them in a table.

The state of the game is defined as three matricies of the field (```3 x [field_size, field_size]```) - one with the position of the actor, one with the position of the pit and one with the position of the goal.

At each step, the flattened state ```s``` (defined as a single row of size ```3 * field_size * field_size```) is fed into the Neural Network. The network then outputs a row of Q-values, each representing the values of the actions in the state ```s```. The  agent then performs an action according to its policy  ```π(s) = max(Q(s,a))``` or another random action given the exploration rate ```ε```.

Once the agent has taken its action, the Q-values of the next state ```s'``` are estimated, and the ```max(Q(s',a'))``` is then used to update the previous Q-value ```Q(s,a)``` as stated in [Q-Gridworld](../q-gridworld).

## Network 

The network in this experiment has two fixed hidden layers, but with customizable number of neurons in each hidden layers.

```
input_size = 3*field_size*field_size
action_size = 4 (up, down, left, right)
```

| Input          | Hidden L1|Hidden L2   | Output          |
|----------------|----------|------------|-----------------|
|```input_size```| 80 [C]   |80 [C]      |```action_size```|

[C] - Customizable


## Performance



