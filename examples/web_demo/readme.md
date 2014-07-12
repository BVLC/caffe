---
title: Web demo
description: Image classification demo running as a Flask web server.
category: example
layout: default
include_in_docs: true
---

# Web Demo

## Requirements

The demo server requires Python with some dependencies.
To make sure you have the dependencies, please run `pip install -r examples/web_demo/requirements.txt`, and also make sure that you've compiled the Python Caffe interface and that it is on your `PYTHONPATH` (see [installation instructions](/installation.html)).

Make sure that you have obtained the Caffe Reference ImageNet Model and the ImageNet Auxiliary Data ([instructions](/getting_pretrained_models.html)).
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
