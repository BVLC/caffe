# CTC Examples

## Dummy Example

The 'dummy' example shows the basic usage of the CTC Layer combined with LSTM networks. Use the `make_dummy_data.py` script to create a single batch of data that will be stored in a hdf5 file. Call the solver prototxts to run the examples.

The solver will overfit the network on the data. Therefore the loss will shrink constantly (more or less fast).

### LSTM

The `dummy_lstm_solver.prototxt` implements a standard one directional LSTM model.

### BLSTM

The `dummy_blstm_solver.prototxt` implements a bidirectional LSTM model using the ReverseLayer.

### Plot progress of learning

The `plot_dummy_progress.py` script will show the learning progress, i.e. the input probabilities and their diffs for each label over time.
