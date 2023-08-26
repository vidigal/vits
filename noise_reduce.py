from scipy.io import wavfile
# import noisereduce as nr
# # load data
# rate, data = wavfile.read("output/G_257000.wav")
# # perform noise reduction
# reduced_noise = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=0.4,stationary=True)
# wavfile.write("output/mywav_reduced_noise.wav", rate, reduced_noise)


import noisereduce as nr

# load data
rate, data = wavfile.read("output/G_318000.wav")
# select section of data that is noise
noisy_part = data[10000:15000]
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, y_noise=noisy_part, sr=rate)
wavfile.write("output/reduced_noise.wav", rate, reduced_noise)