import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import sounddevice as sd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import speech_recognition as sr
import scipy.io.wavfile as wavfile
from gtts import gTTS
import os
st.title("Smart Home Voice Locking System")
# Load the model and data
model = joblib.load('voice_lock_system_ln_up.pkl')
df = pd.read_csv('voice_loking2 (4).csv')

y = df.iloc[:,-1].values
en = LabelEncoder()
en.fit_transform(y)

# Initialize session state for the flag
if "flag" not in st.session_state:
    st.session_state.flag = 0

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def recognize_speaker(file_path):
    mfccs = extract_features(file_path)
    mfccs = mfccs.reshape(1, -1)
    return mfccs

def record_audio():
    st.write("🎙️ Listening...")
    audio = sd.rec(int(4 * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    st.write("🎤 Completed verifying....")
    return audio

if st.button("Speak"):
    r = record_audio()

    # Convert audio to int16 format
    audio_int16 = np.int16(r * 32767)
    wavfile.write("text_con.wav", 44100, audio_int16)

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile("text_con.wav") as source:
        audio_data = recognizer.record(source)
    res = ""

    # Recognize the speech in the audio
    try:
        res = recognizer.recognize_google(audio_data)
        print("Recognized Text:", res)
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


    sf.write("test5.wav", r, 44100)
    r = recognize_speaker('test5.wav')
    pred = model.predict(r)
    pred = en.inverse_transform(pred)
    pred = list(pred)
    print(pred)

    # Define the text for different responses
    text2 = "Sorry,voice notmatching, your access is not granted"
    text = "Welcome Rohith, your access is granted"
    text1 = "Hi Rohith, welcome. The door is open now"
    text3 = "Thank you Rohith. The door is closed now. See you again"
    text4 = "Sorry, I can't understand your command Rohith"

    flag1 = "The door is already open"
    flag0 = "The door is already closed"

    # Create gTTS objects
    tts1 = gTTS(text=text1, lang='en')
    tts3 = gTTS(text=text3, lang='en')
    tts4 = gTTS(text=text4, lang='en')
    tts2 = gTTS(text=text2, lang='en')
    tts_flag1 = gTTS(text=flag1, lang='en')
    tts_flag0 = gTTS(text=flag0, lang='en')

    # Save the audio files
    tts2.save("output2.mp3")
    tts1.save("output1.mp3")
    tts3.save("output3.mp3")
    tts4.save("output4.mp3")
    tts_flag1.save("flag_con1.mp3")
    tts_flag0.save("flag_con0.mp3")

    if pred[0] == 'Rohith':
        if 'open' in res:
            if st.session_state.flag == 0:
                st.write("# WELCOME TO HOME ROHITH")
                os.system("start output1.mp3")
                st.session_state.flag = 1
            else:
                st.write("# WELCOME TO HOME ROHITH")
                os.system("start flag_con1.mp3")
        elif 'close' in res:
            if st.session_state.flag == 1:
                st.write("# Bye i See You Again.")
                os.system("start output3.mp3")
                st.session_state.flag = 0
            else:
                st.write("# Bye i See You Again.")
                os.system("start flag_con0.mp3")
        else:
            st.write("# Sorry try once again.")
            os.system("start output4.mp3")
    else:
        st.write("# try again.")

        os.system("start output2.mp3")

print(st.session_state.flag)
