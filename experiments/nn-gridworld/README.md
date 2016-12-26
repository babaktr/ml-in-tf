# NN-Gridworld Experiment

In comparison to the previous [Q-Gridworld](../q-gridworld) experiment, this experiment attempts to solve [Gridworld](../games/gridworld.py) using a Neural Network to estimate the Q-values instead of storing them in a table.

The state of the game is defined as three matricies of the field - one with the position of the actor, one with the position of the pit and one with the position of the goal.

At each step, the flattened state ```s``` (defined as a single row) is fed into the Neural Network. The network then outputs a row of Q-values, upon which the agent acts according to (with the policy  ```π(s) = max(Q(s,a))``` or random action given the exploration rate ```ε```).

Once the agent has taken its action, the Q-values of the next state ```s'``` are estimated, and the ```max(Q(s',a'))``` is then used to update the Q-value for that state-action pair as stated in [Q-Gridworld](../q-gridworld).

## Graph
<p align="center">
  <img src="../../images/graphs/nn-graph.png", width="50%"/>
</p>

