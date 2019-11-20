# Assignment 03

```
Build a binary classifier based on 3 layers neural network using the human versus horse dataset 
```

## Binary classification based on 3 layers neural network

$`(x_i, y_i)`$ denotes a pair of a training example and $`i = 1, 2, \cdots, n`$

$`p_i = \sigma(u^T x_i + a)`$ (hidden layer)

$`q_i = \sigma(v^T p_i + b)`$ (hidden layer)

$`\hat{y}_i = \sigma(w^T q_i + c)`$ (output layer)

The logistic function $`\sigma`$ is defined by $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$

The loss function is defined by $`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i(u, v, w, a, b, c)`$

$`f_i(u, v, w, a, b, c) = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

## Neural Network Architecture

- The sizes of the hidden layers and the output layer should be determined with respect to the validation accuracy

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using the training set
- The classifier should be tested using the validation set
- Vectorize an input image matrix into a column vector

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You can use any libarary
- You have to write your own implementation for the followings:
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

- Plot the training loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the validation loss at every iteration (x-axis: iteration, y-axis: loss)
- Plot the training accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Plot the validation accuracy at every iteration (x-axis: iteration, y-axis: accuracy)
- Present the table for the final accuracy and loss with training and validation datasets with your best neural network architecture as below:

| dataset    | loss       | accuracy   | 
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit
