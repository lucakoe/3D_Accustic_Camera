import time

import nidaqmx
from nidaqmx.constants import LineGrouping

fps=25
current_video_frame=0
running=True

with nidaqmx.Task() as task:
    task.do_channels.add_do_chan(
        "Dev1/port1/line0", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
    )

    start_time = time.time_ns() / (10 ** 9)
    while running:
        # Calculate the expected timestamp of the current video frame
        expected_audio_time = current_video_frame/fps
        print("e", expected_audio_time)

        # Wait until the expected timestamp is reached
        while (time.time_ns()/ (10 ** 9) - start_time) < expected_audio_time and running:
            pass
        print(time.time_ns()/ (10 ** 9) - start_time)

        if not running:
            break

        task.write([True])
        #time.sleep(1/(fps*2))
        while (time.time_ns()/ (10 ** 9) - start_time) < expected_audio_time+0.005 and running:
            pass
        task.write([False])



        # Save the timestamps to the CSV file
        current_video_frame += 1