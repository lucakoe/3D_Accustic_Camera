import os
import sys
import clr
import shutil
import cv2
import numpy as np
from System.Diagnostics import Stopwatch
from System.Drawing import Bitmap
from System.Drawing import Imaging
from System.Runtime.InteropServices import Marshal
from System.Drawing import Rectangle
import System.Windows.Forms as WinForms

# Add the IC Imaging Control .NET Component directory to the Python module search path
sys.path.append(r"C:\Users\experimenter\PycharmProjects\Accustic_Camera\misc\ICImaging Control3.5\redist\dotnet\x64")

# Load the TIS.Imaging.ICImagingControl35.dll assembly
clr.AddReference("TIS.Imaging.ICImagingControl35")

# Import the ICImagingControl class from the TIS.Imaging.ICImagingControl namespace
from TIS.Imaging import ICImagingControl
from System import *
from System.Drawing import *
from System.Windows.Forms import *
from TIS.Imaging import *
import cv2
import subprocess


import clr
import cv2
import os
import subprocess
import shutil
from System import *
from System.Drawing import *
from System.Windows.Forms import *
from TIS.Imaging import *

def StartRecording():
    # Create an instance of the ImagingControl class
    ic = ICImagingControl()

    # Print the name and UID of each device
    if len(ic.Devices) == 0:
        print("No camera devices found.")
    else:
        print("Available camera devices:")
        for device in ic.Devices:
            print("  - Name: " + device.Name)

    ic.ShowDeviceSettingsDialog()

    # Start the live video stream
    ic.LiveStart()

    # Get video format information
    width = ic.ImageWidth
    height = ic.ImageHeight
    fps = ic.DeviceFrameRate

    # Create a VideoWriter object using OpenCV
    output_file = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Event handler for capturing frames
    def OnImageAvailable(sender, e):
        # Get the current frame from the ImagingControl
        frame = ic.ImageActiveBuffer.Bitmap

        # Convert the frame to a NumPy array
        frame_np = bitmap_to_numpy(frame)

        # Convert the frame to BGR format for writing
        image_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Write the frame to the video file using OpenCV
        writer.write(image_bgr)

        # Display the frame in a window (optional)
        cv2.imshow('Frame', image_bgr)
        cv2.waitKey(1)

    # Attach the event handler to the ImageAvailable event
    ic.OverlayUpdate += OnImageAvailable

    # Wait for Enter key press to stop the recording
    input("Press Enter to stop recording...")

    # Stop the live video stream and release the video writer
    ic.LiveStop()
    writer.release()

    # Close OpenCV windows
    cv2.destroyAllWindows()

# Function to convert Bitmap to NumPy array
def bitmap_to_numpy(bitmap):
    # Get the pixel format and dimensions of the bitmap
    pixel_format = bitmap.PixelFormat
    width = bitmap.Width
    height = bitmap.Height

    # Determine the appropriate NumPy data type based on the pixel format
    if pixel_format == PixelFormat.Format24bppRgb:
        dtype = np.uint8
        channels = 3
    elif pixel_format == PixelFormat.Format8bppIndexed:
        dtype = np.uint8
        channels = 1
    else:
        raise ValueError("Unsupported pixel format")

    # Create a NumPy array with the same dimensions and data type as the bitmap
    bitmap_data = bitmap.LockBits(Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, pixel_format)
    array = np.frombuffer(bitmap_data.Scan0, dtype=dtype)
    array = array.reshape((height, width, channels))
    bitmap.UnlockBits(bitmap_data)

    return array

# Start the recording
StartRecording()