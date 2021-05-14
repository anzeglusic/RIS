#!/usr/bin/python3

import roslib
import time
import rospy

import speech_recognition as sr



class SpeechTranscriber:
    def __init__(self):
        rospy.init_node('speech_transcriber', anonymous=True)
        
        # The documentation is here: https://github.com/Uberi/speech_recognition

        # The main interface to the speech recognition engines
        self.sr = sr.Recognizer()
        
        # These are the methods that are available to us for recognition.
        # Please note that most of them use an internet connection and currently they are using
        # a default API user/pass, so there are restrictions on the number of requests we can make.
        # recognize_bing(): Microsoft Bing Speech
        # recognize_google(): Google Web Speech API
        # recognize_google_cloud(): Google Cloud Speech - requires installation of the google-cloud-speech package
        # recognize_houndify(): Houndify by SoundHound
        # recognize_ibm(): IBM Speech to Text
        # recognize_sphinx(): CMU Sphinx - requires installing PocketSphinx
        # recognize_wit(): Wit.ai
        
        # An interface to the default microphone
        self.mic = sr.Microphone()
        
        # You can get the list of available devices: sr.Microphone.list_microphone_names()
    # You can set the fault microphone like this: self. mic = sr.Microphone(device_index=3)
    # where the device_index is the position in the list from the first command.

    def recognize_speech(self):
        with self.mic as source:
            print('Adjusting mic for ambient noise...')
            self.sr.adjust_for_ambient_noise(source)
            print('SPEAK NOW!')
            audio = self.sr.listen(source)
           
        print('I am now processing the sounds you made.')
        recognized_text = ''
        try:
            recognized_text = self.sr.recognize_google(audio)
        except sr.RequestError as e:
            print('API is probably unavailable', e)
        except sr.UnknownValueError:
            print('Did not manage to recognize anything.')
            
        return recognized_text


if __name__ == '__main__':
    
    st = SpeechTranscriber()
    
    while not rospy.is_shutdown():
        text = st.recognize_speech()
        print('I recognized this sentence:', text)
        time.sleep(4)


