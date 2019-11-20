# Assignment 01

## General Instruction

#### Jupyter Notebook

```console
- Write programming codes in python
- Use Jupyter Notebook for writing codes
- Include comments and intermediate results in addition to the codes
- Export the Jupyter Notebook file in PDF format
- Turn in the PDF file at Google Classroom (late submission is not allowed)
```

#### History of git commits

```console
- Create a private repository at github 
- Commit intermediate status of working file at given steps
- Export the history of commits in PDF format
- Turn in the PDF file at Google Classroom (late submission is not allowed)
```

## Binary Classification based on Logistic Regression

> - $`(x_i, y_i)`$ denotes a pair of a training example and $`i = 1, 2, \cdots, n`$
> - $`\hat{y}_i = \sigma(z_i)`$ where $`z_i = w^T x_i + b`$ and $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$
> - The loss function is defined by $`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i(w, b)`$
> - $`f_i(w, b) = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$

### 1. Plot two clusters of points for training dateset

- Generate two sets of separable random point clusters in $`\mathbb{R}^2`$
- Let $`\{ x_i \}_{i=1}^n`$ be a set of points and $`\{ y_i \}_{i=1}^n`$ be their corresponding labels
- Plot the point clusters in the training dataset using different colors depending on their labels

### 2. Plot two clusters of points for testing dataset

- Generate two sets of separable random point clusters in $`\mathbb{R}^2`$ for a testing dataset using the same centroid and the standard deviation of random generator as the training dataset
- Plot the point clusters in the testing dataset using different colors depending on their labels (different colors from the training dataset)

### 3. git commit

```console
$ git commit -a -m "Plot the training and testing datasets"
$ git push -u origin master
```

### 4. Plot the learning curves

- Apply the gradient descent algorithm
- Plot the training loss at every iteration
- Plot the testing loss at every iteration
- Plot the training accuracy at every iteration
- Plot the testing accuracy at every iteration

### 5. git commit

```console
$ git commit -a -m "Plot the learning curves"
$ git push -u origin master
```










