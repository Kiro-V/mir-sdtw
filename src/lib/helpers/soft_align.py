'''
soft_align.py

This module implements the Soft Alignment pre-processing step for training target.
It removes the repetition of the same chord chord in a sequence, reducing the sequence length.
Remove criteria:
    - If the current chord chord is the same as the previous one, it is removed.
    - If the sequence is empty, it returns an empty array.
    - If the current chord chord is different from the previous one, it is kept.
    - Even if the current is different from the previous, but it exists before, it is kept.
    - If pad_blanks is True, it add a blank value between each chord chord.
'''

import numpy as np

def __add_blanks(length, pad_value=1):
    """Add blank values to the chord sequence.

    Args:
        length (int): Length of the chord sequence.
        pad_value (int): Value to use for padding (default is 1).

    Returns:
        numpy.ndarray: Array of padded blanks.
    """
    return np.full(length, pad_value, dtype=int)

def soft_align(chord_matrix, pad_blanks=False, pad_value=1):
    """Soft alignment of chord chords to remove repetitions.

    Args:
        chord_chords (numpy.ndarray): Array of chord chords.
        pad_blanks (bool): Whether to pad the chord transitions with blanks.
        pad_value (int): Value to use for padding if `pad_blanks` is True (default is 1).

    Returns:
        numpy.ndarray: Soft-aligned chord chords.
    """
    if len(chord_matrix) == 0:
        return np.array([])

    # Initialize the list to hold the soft-aligned chord chords
    aligned_chords = []
    previous_chord = None

    for chord in chord_matrix:
        if chord != previous_chord:
            aligned_chords.append(chord)
            previous_chord = chord
            if pad_blanks:
                aligned_chords.append(__add_blanks(chord.shape[0], pad_value=pad_value))
            else:
                continue
        else:
            # If the chord is the same as the previous one, we skip it
            continue

    # Covert the list to a numpy array
    aligned_chords = np.array(aligned_chords, dtype=int)
    
    return aligned_chords