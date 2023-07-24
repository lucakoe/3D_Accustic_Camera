# TODO make documetation of recording setup
# camera calibration


import shutil

import h5py
import cv2
import avsync
import os
import datetime
import recording
import usvcam_main.usvcam.analysis as analysis
import misc.camera_calib.camera_calib as cameracalib
import calibration
from PIL import Image, ImageTk
from tkinter import Tk, Label
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
camera_devices = [0, 1]  # number doesnt matter, just length; implemented for possible extension
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
        audio_recording_out_path.append(os.path.join(recording_dir, modify_file_name(audio_recording_out_filename, i)))
        recording.rearrange_wav_channels(audio_recording_out_path[i], mic_array_layout[i],
                                         audio_recording_out_path[i])

    video_recording_out_path = []
    for i in range(len(camera_devices)):
        video_recording_out_path.append(os.path.join(recording_dir, modify_file_name(video_recording_out_filename, i)))

    # USV segmentation
    input("Stop Video Recording and press enter. File gets read from temp folder")

    video_recording_out_path = []
    video_recording_temp_path = []
    for i in range(len(camera_devices)):
        video_recording_out_path.append(os.path.join(recording_dir, modify_file_name(video_recording_out_path, i)))
        video_recording_temp_path.append(os.path.join(temp_path, modify_file_name(video_recording_temp_path, i)))

    while (not os.path.exists(video_recording_temp_path[i])):
        input(f"No video file {i} found in temp folder, please move it there and check if the name is correct")

    if os.path.exists(video_recording_temp_path[i]):
        shutil.move(video_recording_temp_path[i], video_recording_out_path[i])
    else:
        print(f"File {i} not found. Cannot move the file.")

    output_path = []
    for i in range(min(len(camera_devices), len(mic_array_devices))):
        output_path.append(os.path.join(recording_dir, modify_file_name(output_filename, i)))

        avsync.combine_vid_and_audio(audio_recording_out_path[i],
                                     video_recording_out_path[i],
                                     # TODO fix but that only outputs one frame on second file
                                     os.path.join(recording_dir, syncfile_filename),
                                     output_path[i], i, fps, sample_rate, cam_delay)

    return recording_dir


def modify_file_name(file_name, i):
    # Split the file name into name and extension parts
    name, extension = file_name.split('.')
    if i is 0:
        return file_name
    # Create the modified file name with the provided number
    modified_file_name = f"{name}_{i}.{extension}"

    return modified_file_name


# makes analysis based on the default names. only processes first file without _i extention.
def analysis_default(recording_dir):
    calibration.wav2dat(recording_dir)
    calibration.create_paramfile(recording_dir, camera_calibration_path, width, height, sample_rate, num_channels,
                                 mic_array_position[0])
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


def calibrate_cameras(recording_dir):
    for i in range(len(camera_devices)):
        video_path = os.path.join(recording_dir, modify_file_name(video_recording_out_filename, i))
        print("Start camera calibration with: " + modify_file_name(video_recording_out_filename, i))
        cameracalib.analyze_chessboardvid(video_path, 23, saveimg=True, frame_intv=5)
        print("Generating calibration file")
        cameracalib.calibrate_intrinsic(video_path)
        mtx = None
        dist = None
        with h5py.File(os.path.splitext(video_path)[0] + '_cam_intrinsic.h5', mode='r') as f:
            print(f.keys())
            mtx = f['/mtx'][()]
            dist = f['/dist'][()]

        print("CAMEAR MATRIX:")
        print(mtx)

        print("DISTORTION COEFFS:")
        print(dist)
        running = True

        try:
            if os.path.exists(os.path.abspath("./misc/camera_calib/tmp")):
                # Use os.startfile to open the Windows Explorer window at the specified path
                os.startfile(os.path.abspath("./misc/camera_calib/tmp"))
            else:
                print("Path does not exist.")
        except Exception as e:
            print("An error occurred:", e)

        while running:
            user_input = input(
                "Put in the number of picture in Camera Calibration Temp you want to undistort as a test. \nOtherwise just press Enter ")
            if user_input is "":
                running = False
            elif os.path.exists(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + ".jpg")):
                img = cv2.imread(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + ".jpg"))
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # undistort
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

                # crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + '_calibresult.png'), dst)

                # Open the images using PIL
                image_distorded = Image.open(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + ".jpg"))
                image_undistorded = Image.open(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + '_calibresult.png'))
                max_height = max(image_distorded.height,
                                 image_undistorded.height)  # Find the maximum height between the two images
                total_width = image_distorded.width + image_undistorded.width  # Calculate the total width required for both images side by side
                combined_image = Image.new('RGB', (
                total_width, max_height))  # Create a new blank image to combine the two images
                combined_image.paste(image_distorded, (0, 0))  # Paste the first image on the left side
                combined_image.paste(image_undistorded,
                                     (image_distorded.width, 0))  # Paste the second image on the right side
                root = Tk()  # Show the combined image using tkinter
                root.title("Images Side by Side")

                # Convert the PIL image to a PhotoImage object to display in tkinter
                img_tk = ImageTk.PhotoImage(combined_image)

                # Create a label and set the image
                label = Label(root, image=img_tk)
                label.image = img_tk
                label.pack()

                root.mainloop()
            else:
                print("File does not exist")


if __name__ == "__main__":
    # recording.get_microphone_info()
    # recording_dir = record_audio_trigger(data_path)
    # analysis_default(recording_dir)
    calibrate_cameras(r"C:\Users\experimenter\PycharmProjects\Accustic_Camera\data\2023-07-24-14-34-47")
