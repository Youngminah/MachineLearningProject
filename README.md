# Assignment 06

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

## Neural Network Architecture

- The neural network architexture should be designed to have 3 layers
- The activation function should be applied to each layer
- Sigmoid function is used for an activation function at each layer

#### First layer

$`Z^{[1]} = W^{[1]} X + b^{[1]}`$ : $`X`$ denotes the input data

$`A^{[1]} = g(Z^{[1]})`$

#### Second layer

$`Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}`$

$`A^{[2]} = g(Z^{[2]})`$

#### Third layer

$`Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}`$

$`A^{[3]} = g(Z^{[3]})`$

## Activation Function

- Sigmoid : 
    $`g(z) = \frac{1}{1 + \exp^{-z}}`$

## Loss function with a regularization term based on $`L_2^2`$ norm

$`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i + \frac{\lambda}{2} \left( \| W^{[1]} \|_F^2 + \| W^{[2]} \|_F^2 + \| W^{[3]} \|_F^2 \right)`$

- Cross Entropy : 
    $`f_i = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

- Frobenius Norm : 
    $`\| W \|_F = \left( \sum_i \sum_j w_{ij}^2 \right)^{\frac{1}{2}}`$

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
- Demonstrate the role of regularization with varying parameter $`\lambda`$ for the tradeoff between bias and variance
- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss at convergence with training and validation datasets
    - training loss (at convergence)
    - validation loss (at convergence)
    - training accuracy (at convergence)
    - validation accuracy (at convergence)

##### Bias (large $`\lambda`$)

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### Variance (small $`\lambda`$)

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

##### Best Generalization (appropriate $`\lambda`$)

- Learning curves
- Loss and Accuracy table 

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Grading

- The grading is given by the validation accuracy for the best generalization (10 digits after the decimal point)
- top 50% would get the score 5 and bottom 50% would get the score 4 (only complete submissions will be considered)
- The maximum score for incomplete submissions will be the score 3

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit











