import os
import librosa
import numpy as np
import noisereduce as nr
from scipy.signal import butter, filtfilt

# Importation des fonctions déjà définies dans votre pipeline
def normalize_sampling_rate(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_resampled, target_sr

def calculate_snr(audio):
    signal_power = np.sum(audio**2)
    noise_power = np.sum((audio - np.mean(audio))**2)
    return 10 * np.log10(signal_power / noise_power)

def normalize_amplitude(audio):
    return audio / np.max(np.abs(audio))

def remove_noise(audio, sr):
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

def apply_low_pass_filter(audio, sr, cutoff_freq=4000):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def spectral_modification(audio, sr):
    fft_audio = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(fft_audio), 1 / sr)
    for i, freq in enumerate(frequencies):
        if 500 <= abs(freq) <= 2000:  # Bande critique pour les formants
            fft_audio[i] *= 0.5  # Réduire l'amplitude de ces fréquences
    modified_audio = np.real(np.fft.ifft(fft_audio))
    return modified_audio

def controlled_randomization(audio):
    noise = np.random.normal(0, 0.01, len(audio))
    randomized_audio = audio + noise
    return randomized_audio

def advanced_pseudonymization(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    modified_mfccs = mfccs + np.random.normal(0, 0.1, mfccs.shape)
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(modified_mfccs, sr=sr)
    return reconstructed_audio

# Fonction anonymize
def anonymize(input_audio_path):
    """
    Anonymization algorithm combining multiple techniques.
    
    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.
    
    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32.
    sr : int
        The sample rate of the processed audio.
    """
    # Étape 1 : Normalisation de la fréquence d'échantillonnage
    audio, sr = normalize_sampling_rate(input_audio_path, target_sr=16000)
    
    # Étape 2 : Prétraitement
    audio = normalize_amplitude(audio)  # Normalisation de l'amplitude
    audio = remove_noise(audio, sr)  # Suppression du bruit
    audio = apply_low_pass_filter(audio, sr)  # Filtre passe-bas
    
    # Étape 3 : Anonymisation (Combinaison des techniques)
    audio = spectral_modification(audio, sr)  # Modification spectrale
    audio = controlled_randomization(audio)  # Randomisation contrôlée
    audio = advanced_pseudonymization(audio, sr)  # Pseudonymisation avancée
    
    # Étape 4 : Post-traitement (assurer la compatibilité avec soundfile.write)
    audio = audio.astype(np.float32)
    
    return audio, sr