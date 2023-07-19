import time

import nidaqmx
from nidaqmx.constants import LineGrouping

start_time=time.time()
fps=1
current_video_frame=0
running=True
while running:
    # Calculate the expected timestamp of the current video frame
    expected_audio_time = current_video_frame/fps

    # Wait until the expected timestamp is reached
    while (time.time() - start_time) < expected_audio_time and running:
        time.sleep(0.001)

    if not running:
        break

    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(
            "Dev1/port1/line0", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
        )

        try:
            task.write([True])
            time.sleep(1/(fps*2))
            task.write([False])
        except nidaqmx.DaqError as e:
            print("Trigger Device Error:\n")
            print(e)



    # Save the timestamps to the CSV file
    current_video_frame += 1