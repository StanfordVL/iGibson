import librosa

from skimage.measure import block_reduce
import wave
import numpy as np
import pybullet as p
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile
import transforms3d as tf3d
import time
import pyaudio
import cv2



class Audiosystem:
    def __init__(self):
        self.SR = 16000
        self.framesPerBuf= int(16000/30)
        self.streaming_input = []
        def pyaudOutputCallback(in_data, frame_count, time_info, status):
            return (bytes(self.current_output), pyaudio.paContinue)
        def pyaudInputCallback(in_data, frame_count, time_info, status):
            # print(in_data)
            self.streaming_input = in_data
            return (None, pyaudio.paContinue)

        self.pyaud = pyaudio.PyAudio()
        info = self.pyaud.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.pyaud.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.pyaud.get_device_info_by_host_api_device_index(0, i).get('name'))
        # self.out_stream = pyaud.open(rate=self.SR, frames_per_buffer=self.framesPerBuf, format=pyaudio.paInt16, channels=2, output=True, stream_callback=pyaudOutputCallback)
        self.in_stream = self.pyaud.open(rate=self.SR, frames_per_buffer=self.framesPerBuf, input_device_index = 1, format=pyaudio.paInt16, channels=1, input=True, stream_callback=pyaudInputCallback)
        self.in_stream.start_stream()
        self.complete_audio = []

    def step(self):
        # source_audio = np.frombuffer(self.streaming_input, dtype=np.int16)
        # print(self.streaming_input)
        self.complete_audio.append(self.streaming_input)
    
    def save_audio(self):
        self.pyaud.terminate()
        # au = np.array(self.complete_audio)#.reshape(-1, 2)
        # print(au.shape)
        waveFile = wave.open("supp_video_results/testing.wav", 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(self.pyaud.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(16000)
        # print(self.complete_audio)
        waveFile.writeframes(b''.join(self.complete_audio))
        waveFile.close()
        # wavfile.write("supp_video_results/testing.wav", 44100, au)

def test():
    import pyaudio
    import wave
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "recordedFile.wav"
    device_index = 1
    audio = pyaudio.PyAudio()

    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    print("-------------------------------------------------------------")

    index = int(input())
    print("recording via index "+str(index))

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    print ("recording started")
    Recordframes = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print ("recording stopped")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()

def main():
    asys = Audiosystem()
    time.sleep(1)
    while asys.in_stream.is_active():
        try:
            # asys.step()
            time.sleep(1/30)
            asys.step()
        except KeyboardInterrupt:
            break
    asys.in_stream.stop_stream()
    asys.in_stream.close()
    asys.save_audio()

if __name__ == "__main__":
    # features()
    # change_music()
    main()
    # test()