from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
import keras

# import age_gender pre_trained model
from pre_trained_model_age_gender import WideResNet
model_age_gender = WideResNet(64, depth=16, k=8)()
model_age_gender.load_weights('/Users/matthew/Desktop/weights.28-3.73.hdf5')

layer_name_age_gender = 'average_pooling2d_1'
conv_base_age_gender = keras.Model(inputs=model_age_gender.input, outputs=model_age_gender.get_layer(layer_name_age_gender).output)  

# import images dataset
train_image_age_gender = []
image_list_age_gender = []
img_dir = '/Users/matthew/Desktop/Thesis_Test'
for _, _, files in os.walk(img_dir):
    for file in files:
        image_list_age_gender.append(file)
        img = image.load_img(os.path.join(img_dir, file), target_size=(64, 64, 3))
        img = image.img_to_array(img)
        img = img/255.
        train_image_age_gender.append(img)
Y = np.array(train_image_age_gender)


# extract age and gender features with pre_trained age_gender model
extractor_age_gender = conv_base_age_gender.predict(Y)
# reshape 
extractor_age_gender = extractor_age_gender.reshape((extractor_age_gender.shape[0], -1))
extractor_age_gender = list(extractor_age_gender)
age_gender_dict = {}
for image, age_gender_feature in zip(image_list_age_gender, extractor_age_gender):
    age_gender_dict[image] = age_gender_feature


# import emotion pre_trained model
from Emotion_Prediction_Pretrained_Model import mini_XCEPTION
model_emotion = mini_XCEPTION((48, 48, 1), 7)
model_emotion.load_weights('/Users/matthew/Desktop/KSU_Honglai Peng/Dr Han/20200327/emotion/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
layer_name_emotion = 'conv2d_23'
conv_base_emotion = keras.Model(inputs=model_emotion.input, outputs=model_emotion.get_layer(layer_name_emotion).output)

# import images dataset 
from keras.preprocessing import image
train_image_emotion = []
image_list_emotion = []
img_dir = '/Users/matthew/Desktop/Thesis_Test'
for _, _, files in os.walk(img_dir):
    for file in files:
        image_list_emotion.append(file)
        img = image.load_img(os.path.join(img_dir, file), color_mode='grayscale', grayscale=True, target_size=(48, 48, 1))
        gray_image = np.squeeze(img)
        gray_image = gray_image.astype('uint8')
        img = image.img_to_array(gray_image)
        img = img/255.
        train_image_emotion.append(img)
X = np.array(train_image_emotion)

# extract emotion features
extractor_emotion = conv_base_emotion.predict(X)
# fletten outputs into one dimensional 
extractor_emotion = extractor_emotion.reshape((extractor_emotion.shape[0], -1))
extractor_emotion = list(extractor_emotion)
emotion_dict = {}
for image, emotion_feature in zip(image_list_emotion, extractor_emotion):
    emotion_dict[image] = emotion_feature
emotion_age_gender_dict = {}
for image, emotion_feature in emotion_dict.items():
    age_gender_feature = age_gender_dict[image]
    age_gender_feature = list(age_gender_feature)
    emotion_feature = list(emotion_feature)
    emotion_feature.extend(age_gender_feature)
    emotion_age_gender_dict[image] = emotion_feature

# race feature extraction with race pre_trained model
import feature_extraction
run feature_extraction
race = '/Users/matthew/Desktop/Race_Thesis_Test.csv'
extractor_race = pd.read_csv(race)

extractor_race = extractor_race.rename(columns={'Unnamed: 0' : 'file_name'})
race_dict = extractor_race.set_index('file_name').T.to_dict('list')
emotion_age_gender_race_dict = {}
for image, emotion_age_gender_feature in emotion_age_gender_dict.items():
    race_feature = race_dict[image]
    race_feature = list(race_feature)
    emotion_age_gender_feature = list(emotion_age_gender_feature)
    emotion_age_gender_feature.extend(race_feature)
    emotion_age_gender_race_dict[image] = emotion_age_gender_feature

emotion_age_gender_race_arr = []
for key, val in emotion_age_gender_race_dict.items():
    emotion_age_gender_race_arr.append(val)

image_list = []
for key, val in emotion_age_gender_race_dict.items():
    image_list.append(key)

emotion_age_gender_race_arr = np.array(emotion_age_gender_race_arr)


# load our own gender model
from keras.models import load_model
model_gender = load_model('/Users/matthew/Desktop/Thesis_pre/model_gender.h5')
gender_prediction = model_gender.predict(emotion_age_gender_race_arr)

gender_list = []
for i in range(len(gender_prediction)):
    if gender_prediction[i][0] > 0.5:
        gender_list.append('M')
    else:
        gender_list.append('F')

gender_tuples = list(zip(image_list, gender_list))
df_gender = pd.DataFrame(gender_tuples, columns=['Image', 'Gender'])

# load our own emotion model
model_emotion =  load_model('/Users/matthew/Desktop/Thesis_Pre/model_emotion.h5')
emotion_prediction = model_emotion.predict(emotion_age_gender_race_arr)

# here we can directly build one dictionary as followed
# emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
from datasets import get_labels
emotion_labels = get_labels('fer2013')

emotion_list = []
for subArray in emotion_prediction:
    emotion_label_arg = np.argmax(subArray)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_list.append(emotion_text)

emotion_tuples = list(zip(image_list, emotion_list))
df_emotion = pd.DataFrame(emotion_tuples, columns=['Image','Emotion'])

# load our own race model
model_race = load_model('/Users/matthew/Desktop/Thesis_pre/model_race.h5')
race_prediction = model_race.predict(emotion_age_gender_race_arr)

Race = ['Asian','Black','White']
race_list = []
for subArray in race_prediction:
    race_label_arg = np.argmax(subArray)
    race_text = Race[race_label_arg]
    race_list.append(race_text)
race_tuples = list(zip(image_list, race_list))

df_race = pd.DataFrame(race_tuples, columns=['Image', 'Race'])

# load our own age model
model_age = load_model('/Users/matthew/Desktop/Thesis_pre/model_age.h5')

age_prediction = model_age.predict(emotion_age_gender_race_arr)
age_list = []
age_label = np.arange(0, 96)
for subArray in age_prediction:
    age_label_arg = np.argmax(subArray)
    age_text = age_label[age_label_arg]
    age_list.append(age_text)
age_tuples = list(zip(image_list, age_list))
df_age = pd.DataFrame(age_tuples, columns=['Image', 'Age'])

df_emotion = df_emotion.drop(columns=['Image'])
df_race = df_race.drop(columns=['Image'])
df_age = df_age.drop(columns=['Image'])
df_all = pd.concat([df_gender, df_emotion], axis = 1, sort = False)
df_all = pd.concat([df_all, df_race], axis=1, sort = False)
df_all = pd.concat([df_all, df_age], axis=1, sort=False)
df_all.to_csv('/Users/matthew/Desktop/Thesis_Test_All.csv')
