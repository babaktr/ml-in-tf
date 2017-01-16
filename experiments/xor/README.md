# XOR Experiment

<p align="center">
  <img src="../../images/graphs/xor-graph.png", width="50%"/>
</p>


This experiment implements a small Neural Network (or a Multilayer Perceptron) with two hidden layers that solves the [XOR](https://en.wikipedia.org/wiki/Exclusive_or) problem [<a name="myfootnote1">1</a>].



<p align="center">
  <img src="../../images/networks/xor-network.png", width="50%"/>
</p>

## Get Started
To get started, use the terminal to navigate to ```ml-in-tf/experiments/xor/```and run ```python xor.py```.

To see the graph and plots using ```tensorboard```, use the terminal to navigate to ```ml-in-tf/``` and run ```tensorboard --logdir logs/```. Wait for the following message:

```
Starting TensorBoard on port <port>
```
And then open up a browser and go to ```localhost:<port>```.

## Network
The network in this experiment has two fixed hidden layers, but with customizable number of neurons in each hidden layers. Given the variables input_size and action_size, the network structure can be described as:

* ```binary output = 0 or 1```

| Input   | Hidden L1 |Hidden L2| Output            |
|:-------:|:---------:|:-------:|:-----------------:|
|   2  	| 2 [c]     |  2 [c]  |```binary output```|

[c] - Customizable

## Parameters
The customizable parameters of this experiment - and their default values - are as follows:

* ```batches``` - ```10000``` - Number of batches (epochs) to run the training on.
* ```hidden_n``` - ```2``` -  Number of nodes to use in the two hidden layers.
* ```learning_rate``` - ```0.1``` - Learning rate of the optimizer.
* ```status_update``` - ```1000``` - How often to print an status update.
* ```optimizer``` - ```gradent_descent``` - Specifices optimizer to use [adam, rmsprop]. Defaults to gradient descent
* ```run_test``` - ```True``` - If the final model should be tested.

## Performance

<p align="center">
  <img src="../../images/plots/xor-plot.png", width="70%"/>
</p>

Example graph showing the error over batches/epochs one hidden node (orange), two hidden nodes (turquoise) and three hidden nodes (purple).

#
<sup>[1](#myfootnote1)</sup> [Multi-layer perceptrons (feed-forward nets), gradient descent, and back propagation](http://ecee.colorado.edu/~ecen4831/lectures/NNet3.html)
