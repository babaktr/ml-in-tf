# Q-Gridworld Experiment

The purpose of this experiment is to show how a basic implementation of [Gridworld](../games/gridworld.py) could be solved with Q-Learning by storing and using all Q-values, _Q(s,a)_, in a table.

For a nonterminal state, it uses the update function:
``` 
Q(s,a) ⟵ Q(s,a) + η(r + ɣmax(Q(s',a') - Q(s,a))
```
And for a terminal state:
``` 
Q(s,a) ⟵ r
```

You'll notice when you play around with the parameters (more specifically the field size) that it will take longer and longer for the agent to perform be able to perform well. In these cases, it might be smart to move away from having to keep all the states and Q-values in the memory and approach this problem from a different angle. 

How about using a Neural Network? Take a look at [this example](../nn-gridworld)!
## Plot
<p align="center">
  <img src="../../images/plots/q-gridworld-plot.png", width="70%"/>
</p>