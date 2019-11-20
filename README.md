# Assignment 04

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

## Activation Function

- Sigmoid

    $`g(z) = \frac{1}{1 + \exp^{-z}}`$

- tanh

    $`g(z) = \frac{\exp^{z} - \exp^{-z}}{\exp^{z} + \exp^{-z}}`$

- ReLU

    $`g(z) = \max(0, z)`$

- Leaky ReLU

    $`g(z) = \max(\alpha z, z), \quad \alpha \in \mathbb{R}^+`$

## Neural Network Architecture

- The sizes of the hidden layers and the output layer should be determined with respect to the validation accuracy obtained by the network architecture with all the activation functions being sigmoid functions. ($`g^{[1]} = g^{[2]} = g^{[3]} =`$ Sigmoid)
- Apply different activation functions at all the layers except the output layer that should be Sigmoid function
- Apply different activation functions at different layers except the output layer that should be Sigmoid function

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using the training set
- The classifier should be tested using the validation set
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

- Apply the gradient descent algorithm with an appropriate learning rate
- Apply the number of iterations that lead to the convergence of the algorith
- Use the vectorization scheme in the computation of gradients and the update of the model parameters

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

## Output

- Do not print out text message per each iteration. It should be illustrated by graphs.
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss with training and validation datasets with your best neural network architecture as below:

##### $`g^{[1]}, g^{[2]}, g^{[3]}`$ are Sigmoid (from the previous assignment)

- Learning curves
- Loss and Accuracy table

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### $`g^{[1]}, g^{[2]}`$ are tanh and $`g^{[3]}`$ is Sigmoid

- Learning curves
- Loss and Accuracy table

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### $`g^{[1]}, g^{[2]}`$ are ReLU and $`g^{[3]}`$ is Sigmoid

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### $`g^{[1]}, g^{[2]}`$ are Leaky ReLU with your choice of $`\alpha`$ and $`g^{[3]}`$ is Sigmoid

- Learning curves
- Loss and Accuracy table

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit











