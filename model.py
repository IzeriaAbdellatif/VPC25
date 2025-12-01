#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`,
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """

    # Read the source audio file
    y, sr = librosa.load(input_audio_path, sr=None)

    # Apply your anonymization algorithm
    # Étape 1 : Modification du Pitch
    y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    # Étape 2 : Transformation des Formants (approximation via Time-Stretching)
    y_time_stretched = librosa.effects.time_stretch(y_pitch_shifted, rate=0.9)

    # Étape 3 : Ajout de Bruit de Fond
    noise = np.random.normal(0, 0.005, y_time_stretched.shape)
    y_noisy = y_time_stretched + noise

    # Étape 4 : Filtrage Fréquentiel (Passe-Haut à 300Hz)
    y_filtered = librosa.effects.preemphasis(y_noisy)

    # Output:
    audio = y_filtered.astype(np.float32)
    sr = sr

    return audio, sr
