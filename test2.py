import usvcam_main.usvcam.analysis as analysis
import usvcam_main.usvcam.tool as tool

import numpy as np
import h5py
import copy
import scipy.interpolate
import scipy.signal
import scipy.io
import scipy.io.wavfile
import cv2
import os
import glob
import pickle
import sklearn
from tqdm import tqdm


def my_pick_seg_for_calib(data_dir, n_pos_check=20):
    print('(some instruction here)')

    vid_file = data_dir + '/vid.mp4'
    vr = cv2.VideoCapture(vid_file)

    paramfile = data_dir + '/param.h5'
    with h5py.File(paramfile, mode='r') as f:
        fs = f['/daq_param/fs'][()]
        n_ch = f['/daq_param/n_ch'][()]

    T = np.genfromtxt(data_dir + '/sync.csv', delimiter=',')

    seg = tool.load_usvsegdata_ss(data_dir)
    _, I_ss = np.unique(seg[:, 4], axis=0, return_inverse=True)
    n_ss = np.max(I_ss) + 1

    # snout pos of each call
    P = []
    SEG = []
    L = glob.glob(data_dir + '/*.usvseg_dat.csv')
    wav_name = os.path.splitext(os.path.splitext(os.path.basename(L[0]))[0])[0]

    def update_disp():
        disp_img2 = copy.deepcopy(disp_img)
        if not np.isnan(p_crnt[0]):
            cv2.circle(disp_img2, (int(p_crnt[0]), int(p_crnt[1])), color=(0, 0, 255), thickness=2, radius=5)
        for p in P:
            cv2.drawMarker(disp_img2, (int(p[0]), int(p[1])), color=(0, 255, 255), thickness=2, markerSize=5)

        cv2.imshow(wname, disp_img2)

    def onMouse(event, x, y, flag, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            p_crnt[0] = x
            p_crnt[1] = y
            update_disp()

    wname = 'click sound source'
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, onMouse, [])
    disp_img = np.zeros([int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vr.get(cv2.CAP_PROP_FRAME_WIDTH)), 3], np.uint8)
    p_crnt = np.zeros([2], float)
    p_crnt[:] = np.nan
    for i_ss in range(n_ss):
        seg2 = seg[I_ss == i_ss, :]
        i_frame = tool.time2vidframe(data_dir, (np.min(seg2[:, 0]) + np.max(seg2[:, 0])) / 2, T, fs)
        vr.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, frame = vr.read()

        imgfile = data_dir + '/seg/' + wav_name + '_{:04}.jpg'.format(int(seg2[0, 4]))
        img = cv2.imread(imgfile)

        disp_img = frame
        r = 100
        a = cv2.resize(img, (r, r))
        disp_img[:r, -r:] = a
        update_disp()

        k = cv2.waitKey()
        if k == 27:  # ESC
            break

        if not np.isnan(p_crnt[0]):
            P.append(copy.deepcopy(p_crnt))
            SEG.append(seg2)

        p_crnt[:] = np.nan

    cv2.destroyAllWindows()

    return SEG, P


def my_calc_micpos(data_dir, SEG, P, calibfile=None, h5f_outpath=None):
    paramfile = data_dir + '/param.h5'
    with h5py.File(paramfile, mode='r') as f:
        fs = f['/daq_param/fs'][()]
        n_ch = f['/daq_param/n_ch'][()]
        speedOfSound = f['/misc/speedOfSound'][()]
        pressure_calib = f['/misc/pressure_calib'][()]
        mic0pos = f['/misc/mic0pos'][()]

    fpath_dat = data_dir + '/snd.dat'

    with open(fpath_dat, 'rb') as fp_dat:

        def get_error(dx, P, SEG, data_dir, mic0pos, speedOfSound):

            dx = np.reshape(dx, (n_ch - 1, 3))
            dx = np.vstack([np.array([0, 0, 0]), dx])
            dx = dx + mic0pos

            n_pos_check = len(SEG)
            pwr = np.zeros(n_pos_check)

            for i_seg in range(n_pos_check):
                seg2 = copy.deepcopy(SEG[i_seg])

                pt = np.array([P[i_seg]])

                tau = tool.get_tau(data_dir, None, speedOfSound, points=pt, micpos=dx)

                S = tool.calc_seg_power(fp_dat, seg2, tau, fs, n_ch, pressure_calib, return_average=True)

                pwr[i_seg] = S

            e = -np.sum(pwr ** 2)

            return e

        global Nfeval
        Nfeval = 1

        def callbackF(Xi):
            global Nfeval
            e = get_error(Xi, P, SEG, data_dir, mic0pos, speedOfSound)
            print('iter:{0:4d}, f(x) = '.format(Nfeval) + str(-e))
            Nfeval += 1
            p = np.reshape(Xi, (3, 3))
            p = np.vstack([np.array([0, 0, 0]), p])
            # print(p + mic0pos)

        # run optimization
        if calibfile is None:
            dx0 = np.tile([0, 0, 0], (3, 1))
            # dx0 = np.array([[-0.015,0.015,0],[0.0,0.03,0],[0.015,0.015,0]])
        else:
            with h5py.File(calibfile, mode='r') as f:
                dx0 = f['/result/micpos'][()]
            dx0 = dx0[1:, :] - dx0[0, :]

        lb = np.tile([-0.06, -0.06, -0.01], (3, 1))
        ub = np.tile([0.06, 0.06, 0.01], (3, 1))

        dx0 = dx0.flatten()
        lb = lb.flatten()
        ub = ub.flatten()

        b = scipy.optimize.Bounds(lb, ub)

        print('Running optimization...')
        R = scipy.optimize.minimize(get_error, x0=dx0, args=(P, SEG, data_dir, mic0pos, speedOfSound),
                                    method='L-BFGS-B', bounds=b, callback=callbackF, options={'maxiter': 50})

        dx_pred = R.x
        dx_pred = np.reshape(dx_pred, (3, 3))
        dx_pred = np.vstack([np.array([0, 0, 0]), dx_pred])
        micpos = dx_pred + mic0pos
        print('estimated micpos:')
        print(micpos)

        if h5f_outpath is None:
            h5f_outpath = data_dir + '/micpos.h5'
        with h5py.File(h5f_outpath, mode='w') as f:
            f.create_dataset('/result/micpos', data=micpos)
            for i_seg in range(len(SEG)):
                f.create_dataset('/seg/seg{:06}'.format(i_seg), data=SEG[i_seg])
                f.create_dataset('/snout_pos/seg{:06}'.format(i_seg), data=P[i_seg])

        print('the result is saved in: ' + h5f_outpath)


data_dir = './data/2023-07-12-10-27-46'

"""SEG, P = my_pick_seg_for_calib(data_dir)

data = {"SEG":SEG, "P":P}

with open('calibdata.pickle','wb') as f:
    pickle.dump(data, f)"""

with open('calibdata.pickle', 'rb') as f:
    data = pickle.load(f)

SEG = data["SEG"]
P = data["P"]

# my_calc_micpos(data_dir, SEG, P, h5f_outpath='./micpos.h5')

analysis.create_localization_video(data_dir, './micpos.h5', color_eq=False)

"""[[-0.0508      0.0056      0.0065    ]
 [-0.05796312  0.0056111   0.0063285 ]
 [-0.05135259  0.01316705  0.00633581]
 [-0.0583739   0.01305352  0.00646433]]"""

# import cv2
# import numpy as np
# import pyaudio
# import wave
# import csv
# import time
# import threading
# import avsync
# import os
# import datetime
# import calibration
# import scipy.io.wavfile
# import usvcam.analysis
import comtypes as comtypes
# def cut_wav_channels(input_file, channels_to_keep, output_file):
#     with wave.open(input_file, 'rb') as wav:
#         channels = wav.getnchannels()
#         sample_rate = wav.getframerate()
#         wav_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
#         wav_data = np.reshape(wav_data, (-1, channels))
#         wav_data = wav_data[:, channels_to_keep]
#         with wave.open(output_file, 'wb') as new_wav:
#             new_wav.setnchannels(wav_data.shape[1])
#             new_wav.setsampwidth(wav.getsampwidth())
#             new_wav.setframerate(sample_rate)
#             new_wav.writeframes(wav_data.tobytes())
#     return output_file
# def wav2dat(data_dir):
#     A = scipy.io.wavfile.read(data_dir + '/cut_audio.wav')
#     max_int16 = 32767
#     max_int32 = 2147483647
#     X = A[1]
#
#     with open(data_dir + '/snd.dat', 'wb') as f:
#         f.write(X.tobytes())
#
# data_dir = os.path.join("data", "2023-07-11-16-40-28")
#
# audio_recording_out_filename = "audio.wav"
# cut_wav_channels(os.path.join(data_dir, audio_recording_out_filename), [3 - 1, 5 - 1, 12 - 1, 14 - 1],
#                  os.path.join(data_dir, 'cut_' + audio_recording_out_filename))
#
# wav2dat(data_dir)
# usvcam.analysis.dat2wav(data_dir, 3)
