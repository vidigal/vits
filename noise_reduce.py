from scipy.io import wavfile
import noisereduce as nr
# load data
rate, data = wavfile.read("output/G_226000.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=0.4,stationary=True)
wavfile.write("output/mywav_reduced_noise.wav", rate, reduced_noise)

