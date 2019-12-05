# Assignment 08

```
Develop a denoising algorithm based on an auto-encoder architecture using pytorch library in the supervised learning framework 
```

## Image denoising problem

- Denoising aims to reconstruct a clean image from a noisy observation
- We use a simple additive noise model using the Normal distribution:

    $`f = u + \eta`$
    
    where $`f`$ denotes a noisy observation, $`u`$ denotes a desired clean reconstruction, and $`\eta`$ denotes a noise process following the normal distribution:

    $`\eta \sim N(0, \sigma^2)`$

    where $`N(0, \sigma^2)`$ denotes the normal distribution with mean 0 and standard deviation $`\sigma`$

## Neural Network Architecture

- Build an auto-encoder architecture based on the convolutional neural network using pytorch
- The dimension of the network input should be the same as the dimension of the network output
- You can design your neural network architecture as you want

## Loss function 

- You can design your loss function for computing a dissimilarity between the output and the ground truth
- The evaluation of the algorithm is given by the mean squared error:

    $`\ell(h, \hat{h}) = \| h - \hat{h} \|_2^2`$

    where $`h`$ denotes a clean ground truth and $`\hat{h}`$ denotes an output of the network

## Dataset

- The dataset consists of training and testing images that are small pieces taken from images 
- The dimension of image is 120x80
- The number of training images is 4400
- The number of testing images is 400
- The range of training images is [0.2, 0.8]
- The range of testing images is [0.0601, 0.9744]
- The training images are clean
- Test testing images are noisy
- The ground truth for the noisy testing images is not given
- The ground truth for the noisy testing images is used for the evalution
- The noise levels of the testing images are 0.01, 0.02, 0.03 and 0.04
- Example images are shown with different degrees of noise $`\sigma = 0.01, 0.02, 0.03, 0.04`$ from the left as below:

![](std_0.01_clean1.png) ![](std_0.02_clean1.png) ![](std_0.03_clean1.png) ![](std_0.04_clean1.png)
![](std_0.01_noise1.png) ![](std_0.02_noise1.png) ![](std_0.03_noise1.png) ![](std_0.04_noise1.png)

![](std_0.01_clean2.png) ![](std_0.02_clean2.png) ![](std_0.03_clean2.png) ![](std_0.04_clean2.png)
![](std_0.01_noise2.png) ![](std_0.02_noise2.png) ![](std_0.03_noise2.png) ![](std_0.04_noise2.png)

## Implementation

- Write codes in python programming
- Use pytorch libarary
- Use ```jupyter notebook``` for the programming environment
- You can use any python libraries
- Write your own code for your neural network architecture
- Write your own code for the training procedure
- Write your own code for the testing procedure

## Code for reading and writing image data

```python
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# custom dataloader for .npy file
class numpyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    NUM_EPOCH       = 2
    
    transform       = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                ])

    # for training
    traindata       = np.load('train.npy')
    traindataset    = numpyDataset(traindata, transform)
    trainloader     = DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=2)

    for epoch in range(NUM_EPOCH):
        for batch_idx, data in enumerate(trainloader):
            image   = data[0]
            to_img  = transforms.ToPILImage()
            image   = to_img(image)

            fig     = plt.figure()
            ax      = fig.add_subplot(1, 1, 1)
            ax.imshow(image, cmap='gray')

            '''
            your code for train
            '''

    # for testing
    testdata        = np.load('test.npy')
    testdataset     = numpyDataset(testdata, transform)
    testloader      = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=2)

    result_for_submit = None    # this is for submit file

    for batch_idx, data in enumerate(testloader):

        result_of_test = data

        if batch_idx == 0:
            result_for_submit = result_of_test
        else:
            try:
                result_for_submit = torch.cat([result_for_submit, result_of_test], dim=0)

            except RuntimeError:
                transposed = torch.transpose(result_of_test, 2, 3)
                result_for_submit = torch.cat([result_for_submit, transposed], dim=0)
        
    # the submit_file.shape must be (400,1,120,80) 
    submit_file = result_for_submit.detach().numpy()
    np.save('your_name.npy', submit_file)
```

## Optimization

- You can use any optimization techniques

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

#### Output (text)

- Print out the followings at each epoch
    - The average of the training loss over mini-batch iterations at each epoch
    - [epoch #####] loss: (training) ########

#### Output (graph)

- Plot the average of the training loss over mini-batch iterations at each epoch
- Plot the standard deviation of the training loss over mini-batch iterations at each epoch

#### Output (file)

- Save the output of the network for the given training images as a file

## Grading

- The grading is given by the performance of the algorithm based on the evaluation criterion (mean squared error) among the complete ones
    - up to top 25% : score 10
    - up to top 50% : score 8
    - up to top 75% : score 6
    - up to top 100% : score 4
    - incomplete : maximum 3
    
## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit
- A data file of the denoising results for the testing images (give a filename: yourname.npy) 
