This code should help you reimplement the experiments in:

Donahue, J., Hendricks, L. A., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. (2014). Long-term recurrent convolutional networks for visual recognition and description. arXiv preprint arXiv:1411.4389.
Chicago	

Please see http://www.eecs.berkeley.edu/~lisa_anne/LRCN_video for detailed instructions on how to reimplement experiments and download pre-trained models.

# Support Flow LRCN From Quadra_L
- [Quadra_L](http://www.swcontest.net/index.php?a=show&m=news&aid=177)
## Preparing
- install protocol buffer 3.4.0, referring this link http://blog.csdn.net/twilightdream/article/details/72953338
- sudo -H pip install --upgrade protobuf==3.1.0.post1
- sudo apt-get install libhdf5-dev
- sudo apt-get install python-h5py
## Building
- remember WITH_PYTHON_LAYER := 1 !
- make all
- make pycaffe

## Training
- download data and models from http://www.eecs.berkeley.edu/~lisa_anne/LRCN_video,
- create flow frame using python instead of matlab like below
    https://github.com/pathak22/pyflow
    git clone https://github.com/pathak22/pyflow.git
    cd pyflow/
    python setup.py build_ext -i
    python demo.py    # -viz option to visualize output
- change my path like '/home/link/frames' in sequence_input_layer.py to your path and modify other relative path.
- bash run_lstm_flow.sh, start training!






