# Crowd-Behavior-Analysis-Based-Automated-Surveillance-System
Surveillance systems based on CCTV cameras and other sensors are quite common now
but they remain quite limited in their effectiveness and efficiency. Due to all judgement
and decision-making being left to human observers, it is quite common to see the observer
misread the situation and by the time they take an action, the situation has already gotten
quite worse. The proposed system provides surveillance systems a much needed deeplearning based monitoring system that can detect violence outbreaks quickly and effectively,
hence allowing human observers to diffuse the situation before things get any worse. The
architecture of the proposed system has four components:
• A 3D-Convolutional Neural Network trained on a dataset of violence outbreaks in.
• A Django-based frontend that allows users to detect violence and aggression in their
CCTV camera/ live feed,
• as well as by uploading videos.
• The system also shows logs of recorded videos and the current time of video in which
violence was detected.
In order to successfully create the proposed system, initially a deep learning architecture(3DDenseNet) is selected. The chosen model is also compared with existing solutions which
include but are not limited to:
• Conv-
• BiConv-LSTM
• Conv2D systems
and the results have been recorded to display the effectiveness of each architecture. Then
all the data in the dataset is perprocessed. Ater preprocessing, the data is converted into
tensors and model is trained. When model accuracy is satisfactory,the states are saved and
then used in the front-end application.
