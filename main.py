import cv2
import numpy as np
import pyaudio
import wave

def record():


    # Set the number of channels, sample rate, and recording duration
    num_channels = 16
    sample_rate = 44100
    duration = 5  # in seconds

    # Set the video dimensions and frame rate
    width = 640
    height = 480
    fps = 30

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new audio stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=num_channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    # Initialize video recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('recording.mp4', fourcc, fps, (width, height))

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Start recording
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
        # Capture video frame at the same time as audio
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            video_out.write(frame)

    # Stop recording and close the audio and video streams
    stream.stop_stream()
    stream.close()
    audio.terminate()
    video_out.release()
    cap.release()

    # Save the recorded audio to a WAV file
    wave_file = wave.open("recording.wav", "wb")
    wave_file.setnchannels(num_channels)
    wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b"".join(frames))
    wave_file.close()





def recordAudio():

    # Set the number of channels, sample rate, and recording duration
    num_channels = 16
    sample_rate = 44100
    duration = 30  # in seconds

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new audio stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=num_channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    # Start recording
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    # Stop recording and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    wave_file = wave.open("recording.wav", "wb")
    wave_file.setnchannels(num_channels)
    wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b"".join(frames))
    wave_file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    recordAudio()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
