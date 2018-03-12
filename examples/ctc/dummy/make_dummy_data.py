import numpy as np
import h5py

def store_hdf5(filename, mapping):
    """Function to store data mapping to a hdf5 file

    Args:
        filename (str): The output filename
        mapping (dic): A dictionary containing mapping from name to numpy data
                       The complete mapping will be stored as single datasets
                       in the h5py file.
    """

    print("Storing hdf5 file %s" % filename)
    with h5py.File(filename, 'w') as hf:
        for label, data in mapping.items():
            print("  adding dataset %s with shape %s" % (label, data.shape))
            hf.create_dataset(label, data=data)

    print("  finished")

def generate_data(T_, C_, lab_len_):
    """Function to generate dummy data

    The data is generated non randomly by a defined function.
    The sequence length is exactly T_.
    The target label sequence will be [0 1 2 ... (lab_len_-1)].

    Args:
        T_ (int): The number of timesteps (this value must match the batch_size of the caffe net)
        C_ (int): The number of channgels/labels
        lab_len_(int): The label size that must be smaller or equals T_. This value
            will also be used as the maximum allowed label. The label size in the network
            must therefore be 6 = 5 + 1 (+1 for blank label)

    Returns:
        data (numpy array): A numpy array of shape (T_, 1, C_) containing dummy data
        sequence_indicators (numpy array): A numpy array of shape (T_, 1) indicating the
            sequence
        labels (numpy array): A numpy array of shape (T_, 1) defining the label sequence.
            labels will be -1 for all elements greater than T_ (indicating end of sequence).
    """
    assert(lab_len_ <= T_)

    data = np.zeros((T_, 1, C_), dtype=np.float32)

    # this is an arbitrary function to generate data not randomly
    for t in range(T_):
        for c in range(C_):
            data[t,0,c] = ((c * 0.1 / C_)**2 - 0.25 + t * 0.2 / T_) * (int(T_ / 5)) / T_

    # The sequence length is exactly T_.
    sequence_indicators = np.full((T_, 1), 1, dtype=np.float32)
    sequence_indicators[0] = 0

    # The label lengh is lab_len_
    # The output sequence is [0 1 2 ... lab_len_-1]
    labels = np.full((T_, 1), -1, dtype=np.float32)
    labels[0:lab_len_, 0] = range(lab_len_)

    return data, sequence_indicators, labels


if __name__=="__main__":
    # generate the dummy data
    # not that T_ = 40 must match the batch_size of 40 in the network setup
    # as required by the CTC alorithm to see the full sequence
    # The label length and max label is set to 5. Use 6 = 5 + 1 for the label size in the network
    # to add the blank label
    data, sequence_indicators, labels = generate_data(40, 20, 5)

    # and write it to the h5 file
    store_hdf5("dummy_data.h5", {"data" : data, "seq_ind" : sequence_indicators, "labels" : labels})

