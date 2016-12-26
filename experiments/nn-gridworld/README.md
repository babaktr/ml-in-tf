# NN-Gridworld Experiment
<p align="center">
  <img src="../../images/graphs/nn-graph.png", width="50%"/>
</p>

In comparison to the [Q-Gridworld](../q-gridworld) experiment, we're now going train an agent to play [Gridworld](../games/gridworld.py) by utilizing a Neural Network instead of storing all Q-values in a table. By sending the state of the game as an input, the network will be trained to estimate the Q-values of that state.

The state of the game is defined as three zero-filled matricies of the field (```3 x [field_size, field_size]```) where the coordinate of the actor, pit and the goal is marked by setting that element of the matrix to 1.

At each step, the flattened state ```s``` (defined as a single row of size ```3 * field_size * field_size```) is fed into the Neural Network. The network then outputs a row of Q-values, each representing the values of the actions in the state ```s```. The  agent then performs an action according to its policy  ```π(s) = max(Q(s,a))``` or another random action given the exploration rate ```ε```.

Once the agent has taken its action, the Q-values of the next state ```s'``` are estimated, and the ```max(Q(s',a'))``` is then used to update the previous Q-value ```Q(s,a)``` as stated in [Q-Gridworld](../q-gridworld).

## Network 

The network in this experiment has two fixed hidden layers, but with customizable number of neurons in each hidden layers.
Given the variables ```input_size``` and ```action_size```, the network structure can be described as:

* ```input_size = 3 * field_size * field_size```
* ```action_size = 4 (up, down, left, right)```

<center>

| Input          | Hidden L1| Hidden L2  | Output          |
|:--------------:|:--------:|:----------:|:---------------:|
|```input_size```| 80 [c]   | 80 [c]     |```action_size```|

</center>
[c] - Customizable


## Performance

<p align="center">
  <img src="../../images/plots/nn-gridworld-plot1.png", width="70%"/>
  <img src="../../images/plots/nn-gridworld-plot2.png", width="70%"/>
</p>

The plots above show the agents training progress running with all parameters set to their default values.

As you can see in the plot, something happend to the performance of the agent just after 600 episodes. From that point and beyond, it really got the gist of it and performed really well thereafter. The result of running 100 test runs with the fully trained agent can be seen in the table below.

|            | Average  |Max  | Min |
|:-----------|:--------:|:---:|:---:|
| **Steps**  | 2.66	    | 9   | 1   |
| **Rewards**| 0.48     | 1   | 0.1 |

