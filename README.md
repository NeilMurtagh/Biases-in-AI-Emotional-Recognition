# Biases in AI Emotional Recognition





## Abstract

Algorithmic facial emotional recognition has been under scrutiny for some time. Its potential use cases reach from HR and online shopping to security or healthcare. But its potential pitfalls are equally manifold. This study investigates racial biases in a simple facial emotional recognition model trained by the FER-2013 dataset. Yet, we are not able to sustain our hypotheses that the model is less accurate for positively recognizing minorities or that there are discriminating patterns within the False Positives. Reasons for that might be low accuracy, flaws in the dataset or sorting difficulties. Nevertheless, the algorithm is better at recognizing anger, fear, sadness and surprise for non-minorities.



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

Usually, models focus on areas of the face that show most contortion, i.e., around the mouth and eyes (Raut 2018). ML can be trained thus to recognize patterns of these features for classification. Further, in videos, changes in the position of facial landmark muscles are also identified. The following image shows some of the usual landmarks used by most models (Raut).

![emotional_recognition](Bild2.png)

FER2013 was designed for a competition to promote research in FER systems, and displays the varied spectrum of differing natural conditions. It contains 35,887 images of 7 expressible emotions (fear, contempt, disgust, anger, surprise, sad, happy, and neutral) in an unbalanced distribution. There is an increasing amount of research and development in this area. Human performance on the dataset

itself is estimated as around 65.5%. Researchers have attained a higher algorithmic accuracy rate of over 75% on it through their multi-layered approaches.

Regardless of background, such as race, gender, nationality, psychological ability, and even culture, cross-group recognition of emotions is fairly easy for humans. It is tougher to identify emotions without considering other cues that are available to the human senses in everyday communication. These can be language and its intonations and body movements. Scholars have found that mainly, language, voice, facial expressions, and gestures, are the building blocks that help us understand mental states. However, facial expressions are a key component and primary determiners in non-verbal communication. According to Song, facial expression information accounts for about 55 percent of the information conveyed in a social situation.



## Literature review 

Existing literature relates to various domains. There are also specific subdomains that need further work, in the intensity of the emotion for example (Mehta et al. 2019). Some major research is based around cross-cultural identification and predictions, differences in predictions made by various groups of humans (women for instance tend to predict emotions better), and on clinical effects (Ekman et al. 1987). The last speaks of the highly nuanced area of diagnosing and monitoring of physiological and mental health diseases including personality and mood disorders, that Kohler et al. (2003) have expanded well upon. This is important because the emotional state of mind and the verbal expression of emotions might not be correlatable in perception, especially in Autism Spectrum Disorders (Lewis and Dunn 2017). Another use case, as per Consoli (2010) is in marketing where consumer feedback can be monitored through the user’s state of interaction with a product. For legal uses, the intent of an offender (Quintero et al. 2018) as well as prevention from risk including drowsy drivers (Wu et al. 2018) are prominent.

Facial emotions lead non-verbal communication and make verbal communication more effective, both for the encoder and the decoder. It is imperative to understand the complex nature of this task, like identification of the faces, features extraction, and guarding from issues arising due to the external environment. On the other hand, language and speech also convey emotional information about a person. Thus, in-depth analyses and studies have also been conducted for speech-based emotional recognition as well.

There are also several challenges that have been researched and further need work, in using such a technology. The necessary use of sensitive data, such as biometric data, is associated with risks. Some media traction has been gained especially in the past few years, owing to its intrusive nature, and deployment by public authorities. Further, its accuracy has been called into question often, which also becomes one of the reasons we wish to delve into this in this paper. The context in which an emotion has occurred can be subject to being disregarded into a single data point. Algorithms have been found to be biased against certain minority groups, such as on the basis of race, that we find out here in this paper. For instance, several instances of research have found that Black people’s faces have been associated more strongly with negative emotions (especially anger). For example, a study on NBA players showed the disparate impact (Rhue, 2018). There are serious dangers of such discrimination, that could occur simply due to a dataset that has some groups underrepresented. The fear of misclassification demands inclusion of ethical principles. FER makes an individual subject to profiling, and consequently being manipulated for purposes that they may not consent to. It can affect decision making and induce behavioural change. Much like usual surveillance, fears of a societal chilling effect and other lateral problems can arise.

We highlight the importance of our present work, despite the plethora of research being done, by bringing forward two alarming recent cases surrounding facial emotion recognition deployment:

1. June 2022: Private multinational corporation Microsoft, following adoption of its Responsible AI Standards, is “phasing out public access” to its AI based emotion analysis tool Azure. They admit that these tools have been under critique due to their lack of scientific backing, in relating a scowl to anger, for instance. Generalization and inferences based upon these have brought concern, states Microsoft’s chief responsible AI officer, Natasha Crampton. (Vincent)

2. May 2021: The government in China has reportedly been testing FER models on dataset obtained from the minority Uighyrs Muslim population forcefully in police stations. According to the informant, “the software creates a pie chart, with the red segment representing a negative or anxious state of mind.” This probably adds to the government’s desire to predict behaviour of their citizens especially in the Xinjiang region. China is estimated to be home to half of the world's almost 800 million surveillance cameras. (Wakefield)



## Methodology

### Training
To train a facial emotional recognition algorithm, the FER-2013 dataset was used. It is one of the most common and easily accessible datasets for Facial Emotional Recognition, used for example by several developers on Github1. The version of the dataset that our model uses can be downloaded as an excel file via this link from Kaggle. It comprises about 30,000 of similarly-looking 48x48 pixel images depicting human faces.

For this paper we used a model pre-trained on said FER-2013 dataset. It defines seven different categories of facial emotion: angry, disgust, fear, happy, sad, surprise and neutral. To build the model, the entire FER-2013 dataset was loaded, reshaped and randomly split it into a training and a test group. Yet, only eight percent of the data was attributed to the test group, which led us to use another dataset with the double amount of testing pictures for our analysis below. After 114 epochs of training using the keras library, the final model is stored as a h5 document. (Faiz99khan, n.d.

### Preparing the training dataset
To test the model, we used the Natural Human Face Images for Emotion Recognition dataset´s testing folder. Besides its accessibility, the dataset is very attractive due to its high similarity to the FER-2013 dataset the model was trained on (black and white, face close-up, includes all seven facial emotions). Drawings, cartoon pictures and the additional “contempt” category were not used while we did not sort out duplications.

We then separated the pictures by racial minority and non-minority. This very critical distinction was made to the best of our knowledge and belief. We are well aware of the difficulties this bears, as disparities between self-identification and the identification made by us cannot be ruled out. The concept of race in a visual context is highly ambiguous and the fact that the pictures were small and in black and white contributed further to this ambiguity. Where we were unsure about marking a picture as minority or non-minority, we left them in the non-minority category.

This manual sorting could have been skipped, had there been a similar and large enough dataset only containing marginalized groups. Yet, the existence of such a dataset would have posed grave moral difficulties considering the potential for misuse.

### Testing
In order to test the model and the FER-2013 dataset it was trained on, we adapted code provided by Clairvoyant´s Arsh Chowdry (2021). It was originally designed to enable users to analyse their own facial expressions using their webcam. The code therefore already includes the processing and resizing of images. Our adapted algorithm uses tensorflow, keras and cv2 to define a function facial_recognition which, when fed with an image, analyses the depicted individual´s facial

expressions. The result is given back as a bar plot. The facial emotion which scores the highest is saved and added to a counter. This counter is finally used to calculate accuracy and misidentification rates.

![emotional_recognition](Bild3.png)

The testing was conducted in a Google Colab. As the Natural Human Face Images for Emotion Recognition dataset provides the pictures as PNGs divided by emotion, we imported them in this format instead of using an excel file. The fact that the data was available as pictures has greatly facilitated the division between minority and non-minority, as comparable excel files only offered as pairs of numbers without a visual equivalent.



## Results 

Overall, we analysed 5286 pictures, 4618 of them non-minority pictures and 678 of them minority pictures. 358 of all minority pictures were accurate, against 2265 of all non-minority pictures. The accuracy rate2 for minority pictures is therefore 53% and the rate for non-minority pictures is 49% respectively. This contradicts our hypothesis that a facial recognition algorithm trained with the FER-2013 dataset would be less accurate at recognizing minorities´ emotions. Furthermore, the model´s overall accuracy rate (49.5%) is disappointingly low, considering other studies featuring accuracy rates of 70% or even 90% (Debnath, Reza, Rahman, Beheshti, Band & Alinejad, 2022, Song, 2021).

![emotional_recognition](Bild4.png)
Figure 1 – True positive rates for each emotion 

As shown in figure one, for minorities, the model resulted in higher true positive rates predicting disgust and neutrality3. Meanwhile for anger, fear, sadness and surprise, the algorithm performed better on non-minorities. The differences for each emotion usually do not exceed six percent. For happiness, the difference is even close to zero. Only for fear, the algorithm detects non-minorities 16% better.

![emotional_recognition](Bild5.png)
Figure 2 – False positive rates of pictures that have been misidentified per emotion

Figure 2 shows as what inaccurately identified emotions were identified instead by using a false positive rate4. When looking for patterns in misidentification, there are no major differences between minority and non-minority, either. For example, the rate of minority pictures misidentified as angry (6.8%) is lower than the rate of non-minority pictures misidentified as angry (7.4%). This contradicts Rhue´s (2018) findings of black individuals´ emotions being perceived as more angrily by Microsoft´s and Face++´s emotional recognition algorithms.

The algorithm seems very good (and eager) at recognizing happiness. It is the emotion with the by far highest true positive rate and the most False Positives (21.2% for minorities and 20.4% for non-minorities). Smiling, especially while showing teeth, might be an easily recognizable feature. Nevertheless, this association between showing teeth and happiness also bears its difficulties. For example, this person (not part of the training nor testing datasets) is unequivocally considered to be happy while he is clearly not

![emotional_recognition](Bild6.png)



## Discussion/Conclusion

While we have found no trace of racial biases in the program’s accuracy, even finding better results for minorities, the overall accuracy is rather low. These imprecise results invite reflexion concerning the FER-2013 dataset, and provides leads in how we could further improve the program used. We must start by reiterating an important point: according to the overwhelming majority of behavioural research, human emotions are hard to divide, define, and can be reflected in profoundly differing ways on individual faces in function of natural facial structure, cultural practices or norms, and the intensity of the emotion. Thus, this system in its current state, and with the limited amount of training we could procure, is not close to being considered operational in any official capacity.

The only major problem we found concerning racial bias was a certain difficulty from the program in perceiving negative emotions like anger, fear and sadness among minority people, as the true positive rate is lower compared to non-minority people. However, one 2018 study (Rhue, 2018) showed that a sadly common bias was to perceive black basketball players as globally “angrier” than their white counterparts without empirical evidence to prove it. So, there could also be a productive or efficient side effect to the program not reproducing human biases. That said, the unpredictability of this features could present dire consequences in practical uses, hence the necessity for a major improvement in the future.

One solution that might be explored, due to the wide array of emotions and emotional facial signals, could be to limit the algorithm to only differentiate between two emotions, such as happy and sad, or similar opposites. This would work towards fixing the very limited racial bias, as well as improving overall accuracy. An even simpler approach could consist in selecting two emotions with dissimilar facial indicators that both already have low misidentification rates. For example, anger (often characterized by pinched lips or bared teeth, and furrowed brows) and surprise (often illustrated in raised eyebrows and an opened mouth), both baring respectively approximately 7% and 1% rates of misidentification. By prioritizing quality over quantity, in a sense, it could be possible to rise the accuracy of the program above. We could thus try to re-train the algorithm with a simplified objective (detecting a range of two emotions) instead of the current one (detective several).

![emotional_recognition](anger (888).png)
![emotional_recognition](images (100)_face.png)


