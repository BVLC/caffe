To run the code on the Reddit ML data dump:
    ./get_char.sh
    python char_model.py

To run on the apollonet.cpp source:
    python char_model.py --data_source ../../../src/caffe/apollonet.cpp

You can run on other files as well. Pass --gpu 0 if you want to run faster.
