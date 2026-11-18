"""
Contains Methods to analyse a wav file containing spoken words for its visemes.
Author: FHR
Based on: https://github.com/marty1885/OpenViseme.git
"""

import pickle

import numpy as np
import librosa
from sklearn.neural_network import MLPClassifier

# DO NOT change. Parameters are fixed by the dataset
# Adapted by from: https://github.com/marty1885/OpenViseme.git
SAMPLE_RATE = 16000
SHIFT_DURATION = 0.008 # 8ms
FRAME_DURATION = 0.016 # 16ms


viseme_model: MLPClassifier | None = None
MODEL_NAME = "viseme_model"
try:
    with open(
        "./oral_exam/audio/" + MODEL_NAME + ".pkl", 
        "rb",
    ) as f:
        viseme_model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: {MODEL_NAME}.pkl could not be found.", flush=True)
    
def wav2frames(signal, sr, shift_duration, frame_duration):
    """Converts wav data into frames as input for the Viseme Model.\n
    Adapted by from: https://github.com/marty1885/OpenViseme.git

    Args:
        signal (_type_): The signal output from loading a wav file with librosa.
        sr (_type_): The sample rate.
        shift_duration (_type_): The shift duration.
        frame_duration (_type_): The length of a Frame.

    Returns:
        view
    """
    shift_length = int(sr * shift_duration)
    frame_length = int(sr * frame_duration)
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / shift_length))
    # Faster way to split signal into frames compared to for loop
    view = np.lib.stride_tricks.as_strided(signal, shape=(num_frames, frame_length), strides=(signal.strides[0]*shift_length, signal.strides[0])).copy()
    assert(len(view) == num_frames)
    return view


def preprocess_utterance(wav_path:str):
    """Loads and Preprocesses a wav file for use with the Viseme-Detection-Model.\n
    Adapted by from: https://github.com/marty1885/OpenViseme.git

    Args:
        wav_path (str): The path to the wav file.

    Returns:
        utterance_frames
    """
    
    signal, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    frames = wav2frames(signal, SAMPLE_RATE, SHIFT_DURATION, FRAME_DURATION)
    
    # wav_data = np.concatenate(frames, axis=0)
    wav_data = frames
    
    # good_frames = np.max(np.abs(wav_data), axis=1) > 1e-6 # Filter out frames with 0 data (quiet frames)
    # wav_data = wav_data[good_frames]
    
    wav = np.array(wav_data)
    
    # Convert into mfcc
    utterance_frames = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=24, n_fft=256, hop_length=128, fmin=120, fmax=4000, center=False)
    utterance_frames = utterance_frames.reshape((utterance_frames.shape[0], -1))
    
    # normalize mfcc
    std = np.mean(np.std(utterance_frames, axis=1))
    mean = np.mean(utterance_frames)
    utterance_frames = (utterance_frames - mean) / std
    
    return utterance_frames

def analyse_visemes(wav_path:str):
    """Analyses a wav file containing spoken words for its visemes.

    Args:
        wav_path (str): The path to the wav file.

    Returns:
        viseme_arr
    """
    
    utterance_frames = preprocess_utterance(wav_path)
    
    analysis = viseme_model.predict(utterance_frames)  
    
    # smoothes out single frames by either replacing them with the value before or after
    for i in range(len(analysis)):
        
        b_smoothen = True
        
        # check whether i should be smoothed
        if (i-1) > 0:
            if analysis[i-1] == analysis[i]:
                b_smoothen = False
        
        if ((i+1) < len(analysis)) and (b_smoothen):
            if analysis[i+1] == analysis[i]:
                b_smoothen = False
        
        # smooth i
        if b_smoothen:
            
            b_replaced = False
            
            if ((i+1) < len(analysis)) and ((i+2) < len(analysis)):
                if (analysis[i+1] == analysis[i+2]):
                    analysis[i] = analysis[i+1]
                    b_replaced = True
            
            if ((i-1) > 0) and ((i-2) > 0) and not b_replaced:
                if (analysis[i+1] == analysis[i+2]):
                    analysis[i] = analysis[i+1]
                    b_replaced = True
            
            if ((i+1) < len(analysis)) and not b_replaced:
                analysis[i] = analysis[i+1]
                b_replaced = True
            
            if ((i-1) > 0) and not b_replaced:
                analysis[i] = analysis[i-1]
                b_replaced = True
        
    # converts the result into an array[dict] to later return it through an api-endpoint
    last_val = -1
    last_dict = {}
    viseme_arr: list[dict] = []
    for val in analysis:
        
        if val == last_val:
            last_dict["duration"] += FRAME_DURATION
        else:
            viseme_dict = {
                "viseme": int(val),
                "duration": FRAME_DURATION
            }
            viseme_arr.append(viseme_dict)
            last_dict = viseme_dict
            last_val = val
    
    return viseme_arr
    