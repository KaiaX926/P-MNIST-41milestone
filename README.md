# P-MNIST-milestone

The MNIST database of handwritten digits is one of the most commonly used dataset for training various image processing systems and machine learning algorithms. MNIST has a training set of 60,000 examples, and a test set of 10,000 examples. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
MNIST is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. The original black and white (bilevel) images from NIST were size normalized. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. The images were centered in a 28 × 28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28 × 28 field.

In the first two questions completed before, I explored the simple learners, including KNN, SVM, and Decision tree. Some of them already have great performance. In the second phase, I used machine-learning algorithms to predict the results. Using different tricks like dropout, batch normalization, and momentum, I found the model that will outperform the simple learners.

Three observations are obtained during the whole process based on the training and the model behaviors.
1. The seed will influence the behavior of models.
2. Convolutional layers do improve the behaviors of the neuron network.
3. Momentum and learning rate is crucial to the model performance. 
