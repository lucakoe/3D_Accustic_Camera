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
    duration = 20  # in seconds

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

    # Start recording
    audio_frames = []
    video_frames = []
    start_time = time.monotonic()
    for i in range(int(duration * fps)):
        # Synchronize audio and video streams using a common clock
        expected_frame_time = start_time + i / fps
        delay = expected_frame_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        # Capture audio frame
        data = stream.read(1024)
        audio_frames.append(data)

        # Capture video frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            video_frames.append(frame)
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
    print("Length of video data:", len(video_frames))

    # Use moviepy to combine the audio and video streams into a single file with synchronized timing
    audio_clip = mp.AudioFileClip("recording.wav")
    video_clip = mp.ImageSequenceClip(video_frames, fps=fps)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("synced_video.mp4", fps=fps)




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
# ffmpeg -f dshow -r 25 -i video="Logi C270 HD WebCam" -use_wallclock_as_timestamps 1 -f dshow -i audio="Line (MCHStreamer Multi-channels)" -map 0:v -map 1:a -b:a 384k -ac 16 -async 0 -max_muxing_queue_size 1024 test.mp4



