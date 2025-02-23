# Report on Data Augmentation Techniques

## Objective

The purpose of this report is to evaluate the impact of data augmentation techniques—specifically, Bootstrapping and Data Synthesis with Noise Addition—on the hotel bookings dataset. These methods are applied to increase the dataset size, preserve the underlying distribution, and enhance model training robustness.


## Overview of Techniques

### Bootstrapping:

- Description: Bootstrapping involves creating additional samples by randomly sampling from the original dataset with replacement. This helps simulate the original data distribution, as the new samples are simply variations of existing data points.

- Parameters: Number of additional samples = 5,000

### Implementation:

Random samples were drawn with replacement from the original dataset.
Additional samples were concatenated with the original dataset.
Data Synthesis with Noise Addition:

- Description: Synthetic data was generated by adding small, normally distributed random noise to the numeric columns of bootstrapped samples. This approach introduces subtle variations while maintaining the dataset's general structure.

- Parameters:
Number of synthetic samples = 5,000
Noise Mean = 0, Noise Standard Deviation = 5% of each column’s standard deviation

### Implementation:

Samples were drawn from the original dataset.
Noise was added to numeric features, while categorical features were retained as-is.
Synthetic samples were concatenated with the original dataset.

