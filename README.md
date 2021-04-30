# facial_emotion_based_song_playback
In this project emotion detection vgg16 model is trained and saved as hdf5 file and loaded into the opencv dnn module, faces are detected and passed to this model for prediction, based on emotion detected a random song from specific folder is played

# how to run
1. create a songs folder in this directory 
2. ![image](https://user-images.githubusercontent.com/47672757/116651116-05142580-a9a0-11eb-9383-b37aeab2ec91.png)
folder hirarchy should be like this and copy any songs of your choice in those folders.

used pygame to play a song, tkinter to create a simple gui app to start detecting
