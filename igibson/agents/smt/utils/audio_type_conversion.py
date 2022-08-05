from scipy.io import wavfile

# source_file = "/viscam/u/wangzz/avGibson/igibson/audio/440Hz_44100Hz.wav"
source_file = "/viscam/u/wangzz/SoundSpaces/sound-spaces/data/sounds/1s_all/telephone.wav"
output_file = "/viscam/u/wangzz/avGibson/igibson/audio/telephone.wav"

sampling_freq, binaural_rir = wavfile.read(source_file)

if(binaural_rir.dtype.kind == 'f'): # float32 -1.0 ~ 1.0
    output = (binaural_rir * 32768.0).astype('int16')
    print("output", output)
    wavfile.write(output_file, 44100, output)


