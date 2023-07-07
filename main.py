import cv2
import numpy as np
import pyaudio
import wave
import csv
import time
import threading

# Global variable for current audio frame
current_audio_frame = 0

# Set the number of channels, sample rate, and recording duration
num_channels = 16
sample_rate = 44100
duration = 5  # in seconds

# Video settings
width = 640
height = 480
fps = 25

def record_audio(stream, audio_frames, num_audio_frames):
    global current_audio_frame, sample_rate
    for i in range(num_audio_frames):
        # Capture audio frame
        data = stream.read(1024)
        audio_frames.append(data)
        current_audio_frame += 1

def record_video(cap, video_out, frames, num_video_frames, csv_writer):
    global current_audio_frame, sample_rate, width, height, fps
    for i in range(num_video_frames):
        # Capture video frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            video_out.write(frame)
            frames.append(frame)
            # Calculate the corresponding time of the audio frame
            if fps == 0:
                audio_time = 0
            else:
                audio_time = (i / fps) + (current_audio_frame / sample_rate)
            # Save the timestamps to the CSV file
            csv_writer.writerow([audio_time, i+1, int(audio_time * sample_rate)])


def record():
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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Initialize CSV writer
    with open('audio_video_timestamps.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Time (s)', 'Video Frame Number', 'Audio Frame Number'])

        # Start recording audio in a separate thread
        audio_frames = []
        num_audio_frames = int(duration * sample_rate / 1024)
        audio_thread = threading.Thread(target=record_audio, args=(stream, audio_frames, num_audio_frames))
        audio_thread.start()

        # Start recording video in a separate thread
        frames = []
        num_video_frames = int(duration * fps)
        video_thread = threading.Thread(target=record_video, args=(cap, video_out, frames, num_video_frames, csv_writer))
        video_thread.start()

        # Wait for both threads to finish
        audio_thread.join()
        video_thread.join()

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

        # Close the CSV file
        csv_file.close()

if __name__ == '__main__':
    record()
