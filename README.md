# Assignment 05

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

#### First layer

$`Z^{[1]} = W^{[1]} X + b^{[1]}`$ : $`X`$ denotes the input data

$`A^{[1]} = g^{[1]}(Z^{[1]})`$ : $`g^{[1]}`$ is the activation function at the first layer

#### Second layer

$`Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}`$

$`A^{[2]} = g^{[2]}(Z^{[2]})`$ : $`g^{[2]}`$ is the activation function at the second layer

#### Third layer

$`Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}`$

$`A^{[3]} = g^{[3]}(Z^{[3]})`$ : $`g^{[3]}`$ is the activation function at the third (output) layer

## Neural Network Architecture

- The neural network architexture should be designed to have 3 layers
- The activation function should be applied to each layer
- You can use any activation function at each layer

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using only the training set
- The classifier should be tested using only the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own functions for the followings:
    - compute the forward propagation
    - compute the backward propagation
    - compute the loss
    - compute the accuracy
    - compute the gradient of the model parameters with respect to the loss
    - update the model parameters
    - plot the results

## Optimization

- You should apply the full gradient descent algorithm with your choice of learning rates
- You should apply enough number of iterations that lead to the convergence of the algorithm
- You should use the vectorization scheme in the computation of gradients and the update of the model parameters
- You can initialize the model parameters with your own algorithm

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Do not print out text message per each iteration. It should be illustrated by graphs
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss at convergence with training and validation datasets
    - training loss (at convergence)
    - validation loss (at convergence)
    - validation loss (when the best validation accuracy is achieved over all the iterations)
    - training accuracy (at convergence)
    - validation accuracy (at convergence)
    - validation accuracy (when the best validation accuracy is achieved over all the iterations)

## Grading

- The grading is given by the best validation accuracy over all the iterations (10 digits after the decimal point)
- top 50% would get the score 5 and bottom 50% would get the score 4 (only complete submissions will be considered)
- The maximum score for incomplete submissions will be the score 3

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit













