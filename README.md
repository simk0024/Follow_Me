[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project #

In this project, a Fully Convolutional Network (FCN) was  designed and trained, making a patrolling drone to look for and follow a specific person in interest in a simulated environment.

[ft]: ./misc/F_followTarget.JPG
[wt]: ./misc/F_withoutTarget.JPG
[pt]: ./misc/F_patrolTarget.JPG
[tc]:./misc/F_TrainingCurves.JPG
[fs1]: ./misc/finalscore1.JPG
[fs2]: ./misc/finalscore2.JPG
[fa]: ./misc/fcnArchitecture.jpg
[fe]: ./misc/fcnEncoder.jpg
[fc]: ./misc/fcnConvolution.jpg
[fd]:./misc/fcnDecoder.jpg
[f1]: ./misc/fcnM1.png
[f2]: ./misc/fcnM2.png
[f3]: ./misc/fcnM3.png
[t1]:./misc/training1.jpg
[t2]: ./misc/training2.jpg
[sc]: ./misc/skipConnection.jpg

## Network Architecture

In this project, classifying the existence of person of interest in the feed images as well as identifying the location of person of interest was done. Thus, the drone controller can take actions like move closer when person of interest walking far, or turn when person of interest not in the center of image. This is known as "semantic segmentation".

In contrast to classic Convolutional Network that classifies the probability a determined class is presented in the image, FCN preserve the spatial information throughout the entire network outputting a map of probabilities corresponding to each pixel of the input image.

A FCN was created and consisting of 3 parts:

1. Encoder network: transforms an image input into feature maps
2. 1x1 convolution: combines the feature maps
3. Decoder network: up-samples the result from previous layer back to the original

![alt_text][fa]



###Encoder Block

Encoder block add features detectors capable of transforming the input image into semantic representation. it squeezes the spatial dimensions at the same time that it increases the depth / number if filters maps, by using a series of convolution layers, forcing the network to find generic representations of the data. 

![][fe]

Code below shows the encoder block:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def encoder_block(input_layer, filters, strides):
    # Creates a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides=strides)
    
    return output_layer
```



### 1x1 Convolution

In between the encoder block and the decoder block is a 1x1 convolution layer that computes a semantic representation by combining the feature maps from the encoder.  It acts like a fully connected layer where the number of kernels is equivalent to the number of outputs of a fully connected layer, and spatial information is preserved.

![][fc]

Code below shows the 1x1 convolution layer:

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(input_layer)
output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
conv1 = conv2d_batchnorm(encoder3, filters=128, kernel_size=1, strides=1)
```



### Decoder Block

Decoder block/ Transposed convolutions up-samples the output from the 1x1 convolution back to the original input format, through the use of a series of transpose convolution layers. What it does is swapping the order of forward and backward passes of convolution, the property of differentiability is retained.

![][fd]

Code below shows the decoder block:

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concatenated_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concatenated_layer, filters)
    
    return output_layer
```



### Skip Connection

Last spatial technique that FCN used is skip connection. Encoder blocks narrow down the scope by looking closely at some picture, and at the same time lose the bigger picture. Even if with decoder blocks to decode the output to original image size, some information has been lost. 

Skip connection connects output of one layer to a non-adjacent layer to retain the information, allows the network to use information from multiple resolutions.

![][sc]

Skip connection sounds magic. However, adding too many skip connections can lead to the explosion in the size of model.

## Build the model

After completing the 3 components of FCN, the FCN architecture has to be designed to get the optimal training result. Starting from a simple network and then incrementally making it more complex by adding more layers.

FCN Models below were evaluated using default hyperparameters as below:

```python
learning_rate =	 0.1
batch_size = 	1
num_epochs = 	20
steps_per_epoch = 200
validation_steps = 	50
workers = 	2
```



### FCN Model #1

1 encoder-decoder pair:

![][f1]

```python
def fcn_model(inputs, num_classes):
    #  Encoder Blocks. 
    enc1 = encoder_block(inputs, 8, 2)

    # 1x1 Convolution layer
    conv1 = conv2d_batchnorm(enc1, 16, 1, 1)
    
    # Decoder Blocks 
    dec1 = decoder_block(conv1, inputs, 8)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(dec1)
```

The final score of model 1 was bad -- **0.0368371760277**.



### FCN Model #2

2 encoder-decoder pairs:

![][f2]

```python
def fcn_model(inputs, num_classes):
    #  Encoder Blocks. 
    enc1 = encoder_block(inputs, 8, 2)
    enc2 = encoder_block(enc1, 16, 2)

    # 1x1 Convolution layer
    conv1 = conv2d_batchnorm(enc2, 32, 1, 1)
    
    # Decoder Blocks 
    dec1 = decoder_block(conv1, enc1, 16)
    dec2 = decoder_block(dec1, inputs, 8)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(dec2)
```

The final score of Model 2 was better than Model 1 -- **0.0987216228404**. 



### FCN Model #3

3 encoder-decoder pairs:

![][f3]

```python
def fcn_model(inputs, num_classes):
    #  Encoder Blocks. 
    enc1 = encoder_block(inputs, 8, 2)
    enc2 = encoder_block(enc1, 16, 2)
    enc3 = encoder_block(enc2, 32, 2)
    
    # 1x1 Convolution layer
    conv1 = conv2d_batchnorm(enc3, 64, 1, 1)
    
    # Decoder Blocks 
    dec1 = decoder_block(conv1, enc2, 32)
    dec2 = decoder_block(dec1, enc1, 16)
    dec3 = decoder_block(dec2, inputs, 8)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(dec3)
```

The final score of Model 3 was better than Model 2 -- **0.344768970662**.



### FCN Model #4

4 encoder- decoder pairs were used, but the final score wasn't better than Model 3. Thus, Model 3 is used for the following experiment.



## Hyperparameters

This is the most time consuming part of this project when fine tuning the parameter and training. Training was done on Amazon EC2 to hasten the processing. Even though, lots of time were spent in finding the optimal hyperparameter, and to hit the final score > 40%.

The hyperparameters includes: 

- **learning rate**: controls how much we are adjusting the weights of our network with respect the loss gradient. 
- **batch_size**: number of training samples/images that get propagated through the network in a single pass.
- **num_epochs**: number of times the entire training dataset gets propagated through the network.
- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. 
- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. 
- **workers**: maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. 

To make life easier, only **learning rate, batch size & num_epochs** were fine tuned, the rest remained unchanged. Several sets of hyperparameter were tried to determine the influence of each hyperparamter and relation between them.

![][t1]

![][t2]

Here listed some findings from the experiment above:

1. The smaller the learning_rate, the higher num_epochs are required to get better score.
2. The smaller batch_size, the shorter training time required. 

Finally, with

```
learning_rate = 0.0008
batch_size = 10
num_epochs = 100
```

the model achieve final score of 42.71%.

![][tc]

![][ft]

![][wt]

![][pt]



## Verification in Simulation

With the trained FCN Model, video below show the result of "Follow Me" activity:

<iframe width="560" height="315" src="https://www.youtube.com/embed/VNN4CajrW3E" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>



## Limitation and Future Development

Supporting more kinds of object of interest. Despite the fact that the trained FCN Model does a good job in following the person of interest, it will not follow any other untrained object, i.e. animal, other person, vehicle, etc. To have a model that works with more categories of object, it's necessary to collect and label images with enough examples for each classes. 

Since the search for hyperparameters was an extremely tedious process I would like to try some sort of automated solution like the Amazon SageMaker Hyperparameter Optimization feature and let it fine tune the parameters automatically.

Adding dropout to the model to prevent overfitting as well as pooling layers. Pooling would be interesting as it would provide a better way to reduce the spatial dimensions without loosing as much information as convolution strides.

