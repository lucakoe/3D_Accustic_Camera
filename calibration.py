from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play

def generateCalibrationTone():
    # Create an empty AudioSegment
    result = AudioSegment.silent(duration=0)  # Loop over 0-14
    tone=AudioSegment.silent(duration=0)  # Loop over 0-14
    for n in range(10, 100):  # Generate a sine tone with frequency 200 * n
        gen = Sine(100 * n)  # AudioSegment with duration 200ms, gain -3
        sine = gen.to_audio_segment(duration=1)  # Fade in / out
        sine = sine.fade_in(10).fade_out(10)  # Append the sine to our result
        tone += sine  # Play the result

    for m in range(180):

        result += tone
        result += AudioSegment.silent(duration=1000)  # Loop over 0-14

    result.export("ascending_tone.wav", format="wav")



if __name__ == '__main__':
    generateCalibrationTone()




