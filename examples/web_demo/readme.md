---
title: Web demo
description: Image classification demo running as a Flask web server.
category: example
include_in_docs: true
priority: 10
---

# Web Demo

## Requirements

The demo server requires Python with some dependencies.
To make sure you have the dependencies, please run `pip install -r examples/web_demo/requirements.txt`, and also make sure that you've compiled the Python Caffe interface and that it is on your `PYTHONPATH` (see [installation instructions](http://caffe.berkeleyvision.org/installation.html)).

Make sure that you have obtained the Reference CaffeNet Model and the ImageNet Auxiliary Data:

    ./scripts/download_model_binary.py models/bvlc_reference_caffenet
    ./data/ilsvrc12/get_ilsvrc_aux.sh

NOTE: if you run into trouble, try re-downloading the auxiliary files.

## Run

Running `python examples/web_demo/app.py` will bring up the demo server, accessible at `http://0.0.0.0:5000`.
You can enable debug mode of the web server, or switch to a different port:

    % python examples/web_demo/app.py -h
    Usage: app.py [options]

    Options:
      -h, --help            show this help message and exit
      -d, --debug           enable debug mode
      -p PORT, --port=PORT  which port to serve content on

## How are the "maximally accurate" results generated?

In a nutshell: ImageNet predictions are made at the leaf nodes, but the organization of the project allows leaf nodes to be united via more general parent nodes, with 'entity' at the very top.
To give "maximally accurate" results, we "back off" from maximally specific predictions to maintain a high accuracy.
The `bet_file` that is loaded in the demo provides the graph structure and names of all relevant ImageNet nodes as well as measures of information gain between them.
Please see the "Hedging your bets" paper from [CVPR 2012](http://www.image-net.org/projects/hedging/) for further information.
