import cv2
import numpy as np
import pyaudio
import wave
import csv
import time

def record():
    # Set the number of channels, sample rate, and recording duration
    num_channels = 16
    sample_rate = 44100
    duration = 5 # in seconds

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new audio stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=num_channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    # Initialize video recording
    width = 640
    height = 480
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('recording.mp4', fourcc, fps, (width, height))

    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Initialize CSV writer
    with open('audio_video_timestamps.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Time (s)', 'Video Frame Number', 'Audio Frame Number'])

        # Start recording
        frames = []
        start_time = time.perf_counter()
        audio_frame_num = 0
        video_frame_num = 0
        audio_frames = []
        num_audio_frames = int(duration * sample_rate / 1024)
        for i in range(num_audio_frames):
            # Capture audio frame
            data = stream.read(1024)
            audio_frames.append(data)
            audio_frame_num += 1
            audio_time = time.perf_counter() - start_time

            # Capture video frame
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                video_out.write(frame)
                frames.append(frame)
                video_frame_num += 1
                # Save the corresponding time of the audio frame in a CSV file
                csv_writer.writerow([audio_time, video_frame_num, audio_frame_num])

        # Stop recording and close the audio and video streams
        stream.stop_stream()
        stream.close()
        audio.terminate()
        video_out.release()
        cap.release()
        cv2.destroyAllWindows()

        # Save the recorded audio to a WAV file
        wave_file = wave.open("recording.wav", "wb")
        wave_file.setnchannels(num_channels)
        wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(b"".join(audio_frames))
        wave_file.close()

        # Check the length of the audio and video data arrays
        print("Length of audio data:", len(audio_frames))
        print("Length of video data:", len(frames))

        # Check the actual frame rate of the video capture device
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print("Actual frame rate:", actual_fps)

        # Get the total number of video frames captured
        total_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total video frames:", total_video_frames)



def recordAudio():

    # Set the number of channels, sample rate, and recording duration
    num_channels = 16
    sample_rate = 44100
    duration = 90  # in seconds

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
    record()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


#TODO Thread for audio and video, global variable for audio frame