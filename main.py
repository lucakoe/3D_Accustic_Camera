# TODO make documetation of recording setup
# camera calibration


import shutil
import avsync
import os
import datetime
import recording
import usvcam_main.usvcam.analysis as analysis
import calibration
import pyaudio

# Audio settings
num_channels = 16
sample_rate = 44100  # careful use the same as the audio device in the system settings
chunk = 1024

# Video settings
trigger_device = "Dev1/port1/line0"
width = 640
height = 480
fps = 20
cam_delay = 0.0

# General settings
data_path = './data'
audio_recording_out_filename = 'audio.wav'
video_recording_out_filename = 'vid.mp4'
syncfile_filename = 'sync.csv'
output_filename = 'video_audio.mp4'
parameter_filename = 'param.h5'
camera_calibration_path = os.path.join(data_path, 'cam_calibration.h5')
mic_calibration_path = os.path.join(data_path, 'micpos.h5')
temp_path = os.path.join(data_path, 'temp')
mic_array_devices = [2, 1]  # device number of microphones in order of output files
mic_array_position = [[-0.063, 0.032, 0.005], [0, 0, 0]]  # relative to camera
mic_array_layout_default = [[], [], [], [], [11, 4, 13, 2], [], [], [], [], [], [], [], [], [], [], [], [],
                            [8, 9, 6, 7, 10, 11, 4, 5, 12, 13, 2, 3, 14, 15, 0,
                             1]]  # layout of used channels depending on number of channels used
mic_array_layout = [mic_array_layout_default[num_channels],
                    mic_array_layout_default[
                        num_channels]]  # mic channels arranged from left top to right bottom (when looking from behind)
new_calibration = False  # if set true, a new calibration for the calib_path gets initiated


def record_audio_trigger(data_path):
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Create a new directory for this recording
    recording_dir = os.path.join(data_path, timestamp)
    os.makedirs(recording_dir)
    recording.record(recording_dir, audio_recording_out_filename, syncfile_filename, trigger_device, fps,
                     mic_array_devices,
                     num_channels, sample_rate,
                     chunk)

    audio_recording_out_path = []
    for i in range(len(mic_array_devices)):
        if i == 0:
            audio_recording_out_path.append(os.path.join(recording_dir, audio_recording_out_filename))
        else:
            audio_recording_out_path.append(f"{os.path.join(recording_dir, audio_recording_out_filename)[:-4]}_{i}.wav")

        recording.rearrange_wav_channels(audio_recording_out_path[i], mic_array_layout[i],
                                         audio_recording_out_path[i])

    # USV segmentation
    input("Stop Video Recording and press enter. File gets read from temp folder")
    while (not os.path.exists(os.path.join(temp_path, video_recording_out_filename))):
        input("No video file found in temp folder, please move it there and check if the name is correct")
    if os.path.exists(os.path.join(temp_path, video_recording_out_filename)):
        shutil.move(os.path.join(temp_path, video_recording_out_filename),
                    os.path.join(recording_dir, video_recording_out_filename))
    else:
        print("File not found. Cannot move the file.")

    avsync.combine_vid_and_audio(os.path.join(recording_dir, audio_recording_out_filename),
                                 os.path.join(recording_dir, video_recording_out_filename),
                                 os.path.join(recording_dir, syncfile_filename),
                                 os.path.join(recording_dir, output_filename), fps, sample_rate, cam_delay)

    return recording_dir


# makes analysis based on the default names. only processes first file without _i extention.
def analysis_default(recording_dir):
    calibration.wav2dat(recording_dir)
    calibration.create_paramfile(recording_dir, camera_calibration_path, width, height, sample_rate, num_channels, mic_array_position[0])
    analysis.dat2wav(recording_dir, num_channels)
    # USV segmentation
    input(recording_dir + "\n" + "Do USV segmentation and press Enter to continue...")

    if new_calibration:
        SEG, P = calibration.my_pick_seg_for_calib(recording_dir)

        data = {"SEG": SEG, "P": P}

        # with open('calibdata.pickle','wb') as f:
        #      pickle.dump(data, f)
        #
        # with open('calibdata.pickle', 'rb') as f:
        #     data = pickle.load(f)

        SEG = data["SEG"]
        P = data["P"]
        calibration.my_calc_micpos(recording_dir, SEG, P, h5f_outpath='./micpos.h5')

    analysis.create_localization_video(recording_dir, mic_calibration_path, color_eq=False)


if __name__ == "__main__":
    # recording.get_microphone_info()
    recording_dir = record_audio_trigger(data_path)
    # analysis_default(recording_dir)
