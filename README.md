# Deep_Convolutional_GANS

## Abstract

Empirical learning models are limited by the number of observations that can
be used to teach them. The real world has a limited amount of data pertaining
to a given phenomenon, thereby making it hard for us to keep up with the high
demand for observations. In light of this issue, Generative Adversarial
Networks prove to be a novel method to generate datasets that mimic the
distributions of the real world. In this project, we aim to examine the
performance of GANs when used to create fake, but realistic images. We run
a Deep Convolutional GAN on two datasets — MNIST and Fashion MNIST
to see how the model fares with a slight increase in pattern complexity. As a
result, we find Deep Convolutional GANs to be extremely capable of image
generation if we develop methods to optimize GAN training to prevent
stagnation of performance.

## Introduction

Machine learning, as it currently stands, is arguably a domain that is dependent on
empirical learning. Most supervised models are based on variations of the following definition of
learning:

Based on N (possibly noisy) observations X = {(x(i), y(i))} of the input and
output of a fixed though unknown system f(x), construct an estimator f’(x; θ)
so as to minimize, E[(L(f (x) − f’(x; θ))]

Although this simple definition has given rise to a diverse plethora of ML models, it still
faces a critical issue — learning cannot be independent of the (possibly noisy) observations. This is a pragmatic 
complication in our endeavour to create smart machines since there might not be
enough data (or examples) to ensure that they are ready to be deployed into the real world. This
is most apparent if we consider the high volume of examples demanded by the Bayes classifier.
Therefore, there is a need for us to generate datasets that reflect the real world in order to ensure
the development of accurate stochastic machines. Generative Adversarial Networks (GAN)
could provide a novel solution for our predicament. Through GANs, we have the ability to try
and replicate distributions within existing datasets. The concept of dataset generation can have a
wide range of potential applications. For example, autonomous cars that are being trained would
not be required to drive on the road in order to collect data, thereby ensuring the safety of others.
Instead, the entire training process could be done by simulating the entire dataset through
generation by GANs.

### Introducing GANS

In Ian Goodfellow’s seminal paper titled Generative Adversarial Networks (Goodfellow
et al.), he introduces the framework in which a generative model is utilized against an adversary;
a discriminative model that learns to classify a sample to be either from the model distribution or
the data distribution. As an analogy, the generative model can be thought of as a team of
counterfeiters attempting to produce fake notes and the discriminative model is similar to the
police attempting to detect the counterfeit samples. The driving intuition behind this framework
is that “competition in this game drives both teams to improve their methods until the
counterfeits are indistinguishable from the genuine articles.” (Goodfellow et al.)

In this framework, to learn the generator’s distribution pg over data x, the authors use an input noise variables pz(z) and then describe a mapping to the data space as G(z; θg), where G is a differentiable function represented by a multilayer perceptron with parameter θg. The second multilayer perceptron (discriminator model), D(x; θd) outputs a single scalar in which D(x) represents the probability that x came from the data and not pg. Finally, D is trained to maximize the probability of assigning the correct label to both training examples and samples from G. Simultaneously, G is trained to minimize log(1 − D(G(z))). Therefore, D and G play the following two-player minimax game with value function V (G, D) in which the global optimum of is at pg = pdata (Goodfellow et al.) 

## Types of GANS

### Conditional GAN

In the unconditional GAN as mentioned above, there is no control over modes of the data
being generated. In contrast, Conditional Generative Adversarial Networks (CGAN) learn to
generate fake samples under a given condition — class label or different modality. (Mirza et al.)

### Wasserstein GAN

Arjovsky et al. have developed Wasserstein GANs as an attempt to teach a machine to
learn a probability distribution. In this method, the loss function is made to include the
Wasserstein distance. Therefore, loss functions associated with WGANs are also correlated with
image quality. Additionally, there is a significant improvement in training stability and
independence from the underlying architecture. This is a great achievement given the fact that
GANs are notoriously hard to train. (Arjovsky et al.)

### Deep Convolutional GAN

Deep Convolutional GANs take advantage of CNNs in order to create a stable framework
for training GANs as well as generating high-quality samples. These commonly used alongside
the aforementioned types in order to generate fake, yet realistic images. (Radford, et al.)

## Approach

### Dataset Used

We use two datasets from the MNIST Database - MNIST handwritten digits and Fashion
MNIST. Both these datasets contain 60,000 training examples and 10,000 testing examples. Each
example is a 28x28 grayscale image and is associated with a class from 10 labels. Although the
number of labels is the same in these two datasets, examples from the Fashion MNIST dataset
are more complex than the handwritten digits. In any case, because of the grayscale property and
size of each image, our choice of datasets makes the image generation process far less
complicated. Before we move on to constructing the convolution and deconvolution networks,
we perform a pre-processing step of normalization.

### Methodology

Since our task is to generate images, we use an Unconditional Deep Convolution GANS
(DCGAN) model. This model consists of two parts - the discriminator and the generator.
The discriminator is a Convolution Net while the generator is a De-Convolution Net.

The code has been heavily based on Tensorflow’s generative model tutorial and Siraj
Raval’s GitHub repository.
Discriminator’s task is to distinguish fake images from real ones. 

Layer | Shape | Activation Function
---------- | ------------- | --------
Convolution Layer | Batch Size, 64, 28, 28 | LeakyRELU
Convolution Layer | Batch Size, 128, 14, 14 | LeakyRELU
Dense | Batch Size, 512, 1, 1 | LeakyRELU

The discriminator is taking a 28x28 grayscale image as its input and downsampling the
image into feature maps which are then passed into an activation function. This process is one
convolution layer. From the table above, we can see that this discriminator has two convolution
layers and so, this process is repeated. However, this time the input of the second layer are the
feature maps from the first layer. These are further downsampled into smaller feature maps and
passed through an activation function. So, we can consider each cycle of downsampling an
image and running it through an activation function as one convolution block.
At the end of the two convolution blocks, we flatten the feature maps into a
one-dimensional vector. This one-dimensional vector is now passed through a regular neural
network and the end product is a probability whether the image is fake or real.
Generator’s task, on the other hand, is to create convincing images that will fool the
discriminator. For this, we construct a deconvolution net. As the term suggests, the structure of a
deconvolution net is the inverse of a convolution net.

Layer | Shape | Activation Function
---------- | ------------- | --------
Dense | Batch Size, 256, 7, 7 | LeakyRELU
Convolution Layer | Batch Size, 128, 14, 14 | LeakyRELU
Convolution Layer | Batch Size, 64, 28, 28 | LeakyRELU

We start off with a one-dimensional vector with random values, call it, X. This vector
undergoes a process of upsampling to create a 28 x 28 grayscale image.
This process of upsampling happens by inverting the convolution blocks to output an 28
x 28 image. The first convolution block is the dense layer which takes as input X and outputs
256 small images. Similarly, the second convolution block creates fewer and larger images till
we get one final 28 x 28 image.
After constructing our convolution and deconvolution nets, we compute the losses for the
discriminator and the generator. We use the loss functions that Ian Goodfellow introduces in his
paper.

![][https://media.giphy.com/media/iiV6XdLAQAIkQoaT8r/giphy.gif]

## Conclusion

The Deep Convolutional Generative Adversarial Network proves to be a great method to
generate fake,but realistic images that seem like they belong to the original dataset. The
applications for such a framework are endless since it could assist in creating more accurate ML
models in virtually every domain. It should be noted that DCGANs are not devoid of any
limitations. In fact, its inability to scale performance has been a critical issue. This is
characteristic of GANs which are known to be hard to train. Some of the reasons are as follows:

Non Convergence — The generator’s attempt to find the “best” image keeps changing
because of our choice of optimization. This introduces the possibility of non-convergence
where optimized values are never attained and it becomes a never-ending game.

Mode Collapse — After creating one good image, the generator may cheat and create the
same image over and over again. This would theoretically deem optimization as
successful but the actual output may not be anywhere close to realistic. This concept of
creating similar images is called mode collapse.

## Future Scope of Study

However, there is a large scope for improvement in this area since we are aware of the
reasons behind difficult in training GANS. A few methods to avoid non-convergence and
mode collapse are as follows:

○ Feature Matching - Feature Matching changes the cost function to minimize the
difference between the features of the real images and the generated images. This
compels the generator to match features with the real images and so it avoids the
possibility of non-convergence.

○ Minibatch Discrimination - To mitigate mode collapses, the discriminator can be
given the ability to penalize the generator if the similarity of images increases.
This penalty is charged by calculating a similarity score between a particular
generated image and the entire batch of generated images. This method ensures
variety in each batch of images created by the generator.

● DCGANs can be used to generate coloured images in precise detail. NVIDIA had
recently created a GAN to create images of fake celebrities using a dataset of existing
celebrities. There is a large scope for study in this field, especially if the machine is able
to classify and label each generated image so that it can be used later on for supervised
learning.

● GANs also can be used to create music or different sounds from numerous datasets. The
applications could be similar to that of images.

● Finally, fake images can be circulated throughout our well-connected world swiftly. This
is alarming given the rise in fake content and news throughout the globe. Therefore, there
is scope in identifying and tagging images that are generated by GANs so that a viewer is
aware of the difference and does not fall prey to fake content.
