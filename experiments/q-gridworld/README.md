# Q-Gridworld Experiment

The purpose of this experiment is to show how a basic implementation of [Gridworld](../games/gridworld.py) could be solved with Q-Learning by storing and using all Q-values, _Q(s,a)_, in a table.

It uses the update function:
```
Q(s,a) <- Q(s'a) + η(r + ɣmax(Q(s',a') - Q(s,a))
```