import torch
import numpy as np

"""
Dataset class for handling input and target sequences.

"""

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, target_length, dataset_params):
        self.inputs = inputs                                # Channels x Time x Chromas
        self.targets = targets                              # Time x Chords
        self.target_length = target_length                  # Length of each target sequence
        self.dataset_params = dataset_params                # Parameters for the dataset (e.g., context, stride)

    def __len__(self):
        return self.inputs.size()[0]
    
    def __getitem__(self, idx):
        X = self.inputs.type(torch.FloatTensor)
        y = self.targets.type(torch.FloatTensor)
        return X, y
    
def __padding(target_seq, soft_length, pad_type):
    """
    Pad the target sequence to the specified length.
    Parameters:
        target_seq: The sequence to be padded.
        soft_length: The length after padding.
        pad_type: The type of padding to apply.
            last: Pad with the last element of the sequence until the desired length is reached.
            random: Randomly repeat elements from the sequence until the desired length is reached.
            uniform: uniformly repeat every element in the sequence until the desired length is reached.
                *prioritize high index elements, if length is not divisible by the sequence length*
            none: No padding, return the sequence as is.
    Returns:
        target_seq: Padded sequence
    """
    # Assertion
    if soft_length < len(target_seq):
        raise ValueError(f'Segment length {len(target_seq)} exceeds soft length {soft_length}.')

    if soft_length <= 0:
        raise ValueError(f"Padded length {soft_length} must be a positive integer.")
    
    # Padding logic
    if pad_type == 'last':
        return target_seq + [target_seq[-1]] * (soft_length - len(target_seq))
    
    elif pad_type == 'random':
        while len(target_seq) < soft_length:
            index = np.random.randint(0, len(target_seq))
            temp = target_seq[index]
            target_seq = target_seq[:index] + [temp] + target_seq[index:]
        return target_seq
    
    elif pad_type == 'uniform':
        repeat_count = soft_length // len(target_seq)
        remainder = soft_length % len(target_seq)
        target_seq = target_seq * repeat_count
        if remainder > 0:
            target_seq += target_seq[:remainder]
        return target_seq
    
    elif pad_type is None:
        return target_seq
    
    else:
        raise ValueError(f"Unknown padding type: {pad_type}")
    

def create_dataset(data_dict, ann_dict, song_dict, song_indices, dataset_params, dataset_description='train', mode='soft', pad='last'):
    """
    Create a dataset from the provided data and annotations.

    Parameters:
        data_dict: Dictionary containing the input data for each song.
        ann_dict: Dictionary containing the annotations for each song.
        song_dict: Dictionary containing song metadata.
        song_indices: List of indices for the songs to include in the dataset.
        dataset_params: Dictionary containing parameters for the dataset (e.g., segment_length, soft_length). Note that if soft_length is not specified, it defaults to segment_length.
        dataset_description: Description of the dataset (e.g., 'train', 'test').
        mode: Mode of dataset creation, either 'full' or 'soft'.
        pad: Padding strategy, currently not used but can be extended in the future.

    Returns:
        full_dataset: A concatenated dataset containing all segments created from the input data (One-hot encoded).
    """
    assert pad in ['last', 'random', 'uniform', None], "Pad must be one of 'last', 'random', 'uniform', or None."
    assert mode in ['full', 'soft'], "Mode must be either 'full' or 'soft' for dataset creation."

    all_datasets = []
    segment_length = dataset_params['segment_length']

    if mode == 'soft':
        if pad is not None and 'soft_length' in dataset_params:
            soft_length = dataset_params['soft_length']
        else:
            soft_length = segment_length  # Default to segment length if not specified
    else:
        soft_length = None

    # Extract voiced frame annotations
    for s in song_indices:
        targets = ann_dict[s][0]
        voiced_frames = np.any(targets !=0, axis=0)
        if voiced_frames.sum() == 0:
            print(f'Warning: No voiced frames found for song {song_dict[s][0]}. Skipping this song.')
            continue
    
        s_idx = np.argmax(voiced_frames)
        e_idx = len(voiced_frames) - np.argmax(voiced_frames[::-1])
        label_seq = targets[:, s_idx:e_idx]

        inputs = data_dict[s].T
        inputs_trimmed = inputs[s_idx:e_idx, :]
        T_total = inputs_trimmed.shape[0]
        num_segments = T_total // segment_length

        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            
            if end_idx > T_total:
                break
            input_seg = inputs_trimmed[start_idx:end_idx, :]
            label_seg = label_seq[:, start_idx:end_idx]
            
            # Soft alignment of full segments
            if mode == 'soft':
                target_seg = [label_seg[:,i] for i in range(label_seg.shape[1]) if not np.array_equal(label_seg[:, i], label_seg[:, i-1]) or i == 0]
                target_seg = __padding(target_seg, soft_length, pad_type=pad)

            elif mode == 'full':
                target_seg = [label_seg[:, i] for i in range(label_seg.shape[1])]

            target_seg_len = torch.tensor([len(target_seg)], dtype=torch.int64)
            
            target_seg = np.array(target_seg, dtype=np.float32)
            target_seg = torch.unsqueeze(torch.tensor(target_seg, dtype=torch.float32), 0)

            inputs_tensor = torch.unsqueeze(torch.tensor(input_seg, dtype=torch.float32), 0)

            curr_dataset = dataset(inputs_tensor, target_seg, target_seg_len, dataset_params)
            all_datasets.append(curr_dataset)
                    
        print(f'- {song_dict[s][0]} added to {dataset_description} set. Segments: {num_segments}')
        
    full_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print(f'Total segments created for {dataset_description} dataset: {len(full_dataset)}')
    return full_dataset

"""
    Utility functions for converting between one-hot encoded sequences and lists of indices.
    These functions are meant to be used for easy conversion before creating tensors.

"""

def hot2list(hot_seq):
    """
    Convert a one-hot encoded sequence to a list of indices.
    Parameters:
    - hot_seq: A one-hot encoded sequence.
    Returns:
    - A list of indices corresponding to the non-zero elements in the one-hot encoded sequence.
    """
    if isinstance(hot_seq, torch.Tensor):
        hot_seq = hot_seq.detach().cpu().numpy()
    return [torch.argmax(hot_seq[:, i]).item() for i in range(hot_seq.shape[1]) if torch.any(hot_seq[:, i] != 0)]

def list2hot(seq, num_classes):
    """
    Convert a list of indices to a one-hot encoded sequence.
    Parameters:
    - seq: A list of indices.
    - num_classes: The number of classes for one-hot encoding.
    Returns:
    - A one-hot encoded sequence (2D NP array).
    """
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()
    hot_seq = np.zeros((num_classes, len(seq)), dtype=np.float32)
    for i, idx in enumerate(seq):
        hot_seq[idx, i] = 1.0
    return hot_seq


"""
    Custom collate_fn for loop batching, for dataset with no padding.

"""
def custom_collate_fn(batch):
    """
    Custom collate function for batching without padding.
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [item[2] for item in batch]
    return inputs, targets, lengths
