from gtts import gTTS
import io
import pygame
import time

def text_to_speech(text):
    # Initialize the gTTS engine
    if text:
        tts = gTTS(text=text, lang='en', slow=False)

        # Create an in-memory file-like object
        audio_stream = io.BytesIO()
        
        # Save the speech to the in-memory file-like object
        tts.write_to_fp(audio_stream)

        # Play the audio from the in-memory file-like object using pygame
        audio_stream.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

# if __name__ == "__main__":
#     text = "Hello, how are you today?"

#     # Play audio without saving to local file
#     text_to_speech(text)




#******************using pydub****************

# import pyttsx3
# from pydub import AudioSegment
# from pydub.playback import play

# def text_to_speech(text):
#     # Initialize the text-to-speech engine
#     engine = pyttsx3.init()

#     # Set properties (optional)
#     engine.setProperty('rate', 150)  # Speed of speech

#     # Convert text to speech
#     engine.say(text)
#     engine.runAndWait()

# def text_to_audio_segment(text):
#     # Initialize the text-to-speech engine
#     engine = pyttsx3.init()

#     # Set properties (optional)
#     engine.setProperty('rate', 150)  # Speed of speech

#     # Save the speech as an in-memory audio segment
#     audio_segment = AudioSegment.from_mp3(engine.save_to_file(text, 'temp.mp3'))
    
#     return audio_segment

# def play_audio_segment(audio_segment):
#     # Play the in-memory audio segment
#     play(audio_segment)

# if __name__ == "__main__":
#     text = "Hello, how are you today?"

#     # Play audio without saving to local file
#     text_to_speech(text)

    # Alternatively, you can use the following method to play audio without saving to a local file
    # audio_segment = text_to_audio_segment(text)
    # play_audio_segment(audio_segment)




