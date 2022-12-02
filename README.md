# Biases in AI Emotional Recognition

## Abstract
Algorithmic facial emotional recognition has been under scrutiny for some time. Its potential use cases reach from HR and online shopping to security or healthcare. But its potential pitfalls are equally manifold. This study investigates racial biases in a simple facial emotional recognition model trained by the FER-2013 dataset. Yet, we are not able to sustain our hypotheses that the model is less accurate for positively recognizing minorities or that there are discriminating patterns within the False Positives. Reasons for that might be …

## Introduction
The importance of facial emotional recognition (FER) is well-acknowledged in fundamental human communication. Machine learning algorithms can be trained and tested on suitable datasets to carry out the process of recognizing emotions. FER technology belongs to the multidisciplinary family of “affective computing” that is often attributed towards the advent of Artificial Intelligence. Cameras and allied technological changes have provided impetus for these algorithms to become crucial.

FER technology seeks to achieve categorization of emotions from images. Deep learning allows notable accuracy in this task. The technology is used in applications such as robotics, medicine, surveillance for safety from crime and road accidents, marketing, etc (Khanzada et al. 2020). Vemou et al. broadly summarise uses as follows:
· Provision of custom services
· Customer behaviour analysis and advertising
· Healthcare
· Employment
· Public safety
· Crime detection
· Other

Yet, accuracy and robustness still remain sought-after, due to variations in the faces (heterogeneous features, cultural differences of expressions, etc.) as well as conditions of the images (natural conditions exhibit diverse poses and lighting) (Khaireddin and Chen 2021).
Human-computer interaction as a field is greatly supplemented by emotional recognition. The process has three important steps: face detection, feature extraction and classification module (Song 2021). Machines first need to detect the faces from the environment around it. Most FER models rely on colouring the images into grayscale, which we also do in our code. This allows for a better performance. Furthermore, either window-based or pixel-based techniques are utilised (Khan 2022).

Of course, defining emotion itself comes with a long history of debate from before the time of Greek philosophers. Most definitions refer to it as an impulse that prompts behaviour in an organism adapted to meet its needs. Human emotions are sometimes very subtle and expressions minimal. Psychologists, neuroscientists and other behavioural research scholars have not yet agreed upon universal features of specific emotions, and this serves as a first caution to the usage of FER systems.

Usually, models focus on areas of the face that show most contortion, i.e., around the mouth and eyes (Raut 2018). ML can be trained thus to recognize patterns of these features for classification. Further, in videos, changes in the position of facial landmark muscles are also identified. The following image shows some of the usual landmarks used by most models (Raut).Introduction


![emotional_recognition](image.jpg)
