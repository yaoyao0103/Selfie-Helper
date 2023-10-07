## Introduction
This system implements selfie detection to automatically count down the selfie, and combines gesture detection to switch custom filters. It reduces the manual operation of the device, allowing users to quickly switch filters and take selfies conveniently. It also combines CNN deep learning technology to implement selfie detection and hand posture detection models. The countdown starts when a selfie is detected. 

In addition, combined with AI Makeup, custom filters can be implemented. Custom filters can be switched through gestures in the camera, and specific gestures can be used to start or end selfies.

## Motivation
- Replaces the trouble of manually setting the countdown when taking a selfie
- I always forget to charge my selfie stick before going out. If the selfie stick runs out of battery during the journey, it will cause trouble. Through selfie detection, I can count down and take pictures without any operation.
- Selecting a filter before taking a photo always requires manual control for a long time. After selecting the filter, it may not be suitable for the actual application on the face, and you have to select it again. This system allows you to preset several commonly used filters. When the lens is directly facing your face, you can change the preset filters directly through hand gestures, saving you a lot of trouble.
- You can set a variety of filters in advance and apply them directly when taking pictures, which reduces the trouble of retouching pictures one by one later.
- Use AI makeup technology to avoid the trouble of novice photo retouchers who are not good at photo retouching.
- Automatically start/stop taking selfies through specific gestures.

## Display

### Original Image(Left), image applied filter(Right)
![image](https://github.com/yaoyao0103/Selfie-Helper/assets/76504560/1f048fe7-f247-40b9-86f9-1ced49f66803)

### Selfie is not detected(Left), selfie is detected and then start to count down, and detects gestures to apply filter No. 1 (Right)
![image](https://github.com/yaoyao0103/Selfie-Helper/assets/76504560/89400437-f5b4-403b-a552-8af2eb6aa231)


## Implementation Steps
- Find selfie dataset and train selfie detection model.
- Find hand sign dataset and train hand sign detection model.
- Use opencv to capture photos within the lens.
- Use cvzone's PostDetector to capture body objects.
- Calculate the proportion of body objects in the screen through the data of body objects (called body_rate here).
- Use MTCNN for face detection and capture faces.
- If the detected face and body_rate match, selfie detect will be performed. If they match, the selfie will be counted down.
- Use cvzone's HandDetector to capture hand objects.
- During the countdown stage, if you make a gesture with a specific number, you can switch to the filter of that mode.
- Modify the ready-made AIMakeUp program to apply facial filters to the faces captured by MTCNN.

## Reference
- AIMakeUp GitHub: https://github.com/QuantumLiu/AIMakeup

## Usage
- When the user faces the camera and detects the Selfie action, it will automatically count down for 10 seconds. The countdown can also be controlled by hand movements to start and end the countdown.
- When the user's face and hands are captured at the same time, the closest face and hands will be matched as the same person, and the hand movements will be detected and the corresponding gesture filter will be applied to the face.
- Users can set several customized filters in advance.
