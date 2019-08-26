# Genetic Algorithms for Neural Architecture Search

## Introduction
This project performs neural architecture search via various evolution strategies (crossover, mutation and selection).  It performs an NAS on a shallow CNN to be trained on the CIFAR-10 dataset.

This code can be used to find the perfect NN for your specific dataset.

## Requirements

    python3
    keras
    tensorflow
    
I also recommend using a GPU. 

## How to run

    python ga_nas.py
    
or run

    ga_nas.ipynb

## Programming Strategies
This project uses some strategies with respect to how the DNA is translated to its phenotype.

![Translation scheme](https://github.com/justinmacp/NAS_via_GeneticAlgorithms/blob/master/GenotoPheno.png)

The genotype of an individual consists of 1 + 3 * N values, where N is the network depth. Since the problem is an image recogntion problem, it reqires a feature extraction stage, then a flattening layer and then a fully connected stage to perform the classifcation.

The first entry in the DNA determines the layer number before which the flattening occurs. In the current implementation this is the ultimate or penultimate layer and the first entry is either a 1 or a 0. Each subsequent triplet determines a layer and its parameters. The following table indicates the phenotype translation of a triplet.

First value of the triplet: determines layer type depending on the stage it is in (feature extraction or classification):

| Value | Feature extraction | Classifiaction |
| --- | --- | --- |
| 0 | Conv | Dense |
| 1 | Dropout | Dropout |
| 2 | Pooling | Gaussian Noise |

The second value of the triplet determines a parameter of the layer depending on the layer. It's a number from 0 to 9:

| Layer type | Parameter (p) |
| --- | --- | 
| Conv | 2 * p = number of filters |
| Dense | 10 * p = number of neurons |
| Dropout | 0.1 * p = dropout rate |
| Pooling | 2 + p = stride |
| Gaussian Noise | 0.1 * p = standard deviation |

The third value of the triplet determines another parameter of the layer. It is a number from 0 to 3:

| Layer type | Parameter (q) |
| --- | --- | 
| Conv | 2 * q + 1 = kernel size |
| Pooling | pooling type (max if less than 2, average otherwise) |

The other layers dont require a second parameter. 

The at the beginning of ga_nas.py there are some hyperparameters predefined. These can be adjusted according to the user's needs. 

## Future improvements

1. a more stable genotype encoding strategy. A single mutation has the possibility to change an entire layer. This is not condusive to successful evolution.
2. incorporate various other factors into the fitness evaluation. Often it's not only accuracy that determines the quality of a network, but also speed (i.e. real time image recognition).
3. encode more hyperparameters of the network into DNA. The learning rate, batch size etc. are good candidated for optimization via genetic algorithms as well
4. variable network depth. Randomly select number of layers of an individual in the population.
