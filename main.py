import time
import cv2
import numpy as np
import pyaudio
import wave
import moviepy.editor as mp

def record():


    # Set the number of channels, sample rate, and recording duration
    num_channels = 16
    sample_rate = 48000
    duration = 5  # in seconds

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new audio stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=num_channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=2048)

    # Initialize video recording
    width = 640
    height = 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('recording.mp4', fourcc, fps, (width, height))

    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Start recording
    audio_frames = []
    num_audio_frames = int(duration * sample_rate / 2048)
    start_time = time.time()
    for i in range(num_audio_frames):
        # Synchronize audio and video streams
        current_time = time.time() - start_time
        expected_frame_time = i * 2048 / sample_rate
        delay = expected_frame_time - current_time
        if delay > 0:
            time.sleep(delay)

        # Capture audio frame
        data = stream.read(2048)
        audio_frames.append(data)

        # Capture video frame
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
    wave_file.writeframes(b"".join(audio_frames))
    wave_file.close()

    # Check the length of the audio and video data arrays
    print("Length of audio data:", len(audio_frames))

    # Check the actual frame rate of the video capture device
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Actual frame rate:", actual_fps)


    # # Load the audio and video files
    # audio = mp.AudioFileClip("recording.wav")
    # video = mp.VideoFileClip("recording.mp4")
    #
    # # Synchronize the audio and video streams
    # synced_audio = audio.set_start(video.start)
    # synced_video = video.set_audio(synced_audio)
    #
    # # Write the synchronized video to a file
    # synced_video.write_videofile("synced_video.mp4", fps=video.fps)

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



# wip command for capture
# ffmpeg -f dshow -i video="Logi C270 HD WebCam" -f dshow -i audio="Line (MCHStreamer Multi-channels)" -map 0:v -map 1:a -c:v libx264 -c:a aac -b:a 384k -ac 16 test.mp4


