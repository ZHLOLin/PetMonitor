import pyaudio
import time
import audioop
import numpy as np
import email_sender

# Replace this with the index of your USB microphone
usb_microphone_index = 7

# Callback function to process audio input
def audio_callback(in_data, frame_count, time_info, status):
    rms = audioop.rms(in_data, 2)  # Calculate RMS of audio data
    db = 20 * np.log10(rms)        # Convert RMS to dB
    if db >= 50:
        email_sender.send_mail(2)
    time.sleep(0.1)
    # print(f"Volume: {db:.2f} dB")
    return (in_data, pyaudio.paContinue)

# Initialize audio input
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    input_device_index=usb_microphone_index,
                    frames_per_buffer=1024,
                    stream_callback=audio_callback)

# Start audio stream
stream.start_stream()

# Keep the script running and measuring volume
try:
    while True:
        pass
except KeyboardInterrupt:
    # Stop the audio stream when user presses Ctrl+C
    stream.stop_stream()
    stream.close()
    audio.terminate()
