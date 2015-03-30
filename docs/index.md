---
title: Deep Learning Framework
---

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and by community contributors.
[Yangqing Jia](http://daggerfs.com) created the project during his PhD at UC Berkeley.
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

Check out our web image classification [demo](http://demo.caffe.berkeleyvision.org)!

## Why Caffe?

**Expressive architecture** encourages application and innovation.
Models and optimization are defined by configuration without hard-coding.
Switch between CPU and GPU by setting a single flag to train on a GPU machine then deploy to commodity clusters or mobile devices.

**Extensible code** fosters active development.
In Caffe's first year, it has been forked by over 1,000 developers and had many significant changes contributed back.
Thanks to these contributors the framework tracks the state-of-the-art in both code and models.

**Speed** makes Caffe perfect for research experiments and industry deployment.
Caffe can process **over 60M images per day** with a single NVIDIA K40 GPU\*.
That's 1 ms/image for inference and 4 ms/image for learning.
We believe that Caffe is the fastest convnet implementation available.

**Community**: Caffe already powers academic research projects, startup prototypes, and even large-scale industrial applications in vision, speech, and multimedia.
Join our community of brewers on the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) and [Github](https://github.com/BVLC/caffe/).

<p class="footnote" markdown="1">
\* With the ILSVRC2012-winning [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model and caching IO.
Consult performance [details](/performance_hardware.html).
</p>

## Documentation

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)<br>
Tutorial presentation.
- [Tutorial Documentation](/tutorial)<br>
Practical guide and framework reference.
- [arXiv / ACM MM '14 paper](http://arxiv.org/abs/1408.5093)<br>
A 4-page report for the ACM Multimedia Open Source competition (arXiv:1408.5093v1).
- [Installation instructions](/installation.html)<br>
Tested on Ubuntu, Red Hat, OS X.
* [Model Zoo](/model_zoo.html)<br>
BVLC suggests a standard distribution format for Caffe models, and provides trained models.
* [Developing & Contributing](/development.html)<br>
Guidelines for development and contributing to Caffe.
* [API Documentation](/doxygen/annotated.html)<br>
Developer documentation automagically generated from code comments.

### Examples

{% assign examples = site.pages | where:'category','example' | sort: 'priority' %}
{% for page in examples %}
- <div><a href="{{page.url}}">{{page.title}}</a><br>{{page.description}}</div>
{% endfor %}

### Notebook Examples

{% assign notebooks = site.pages | where:'category','notebook' | sort: 'priority' %}
{% for page in notebooks %}
- <div><a href="http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/{{page.original_path}}">{{page.title}}</a><br>{{page.description}}</div>
{% endfor %}

## Citing Caffe

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

If you do publish a paper where Caffe helped your research, we encourage you to update the [publications wiki](https://github.com/BVLC/caffe/wiki/Publications).
Citations are also tracked automatically by [Google Scholar](http://scholar.google.com/scholar?oi=bibs&hl=en&cites=17333247995453974016).

## Contacting Us

Join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) to ask questions and discuss methods and models. This is where we talk about usage, installation, and applications.

Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Contact [caffe-dev](mailto:caffe-dev@googlegroups.com) if you have a confidential proposal for the framework *and the ability to act on it*.
Requests for features, explanations, or personal help will be ignored; post to [caffe-users](https://groups.google.com/forum/#!forum/caffe-users) instead.

The core Caffe developers offer [consulting services](mailto:caffe-coldpress@googlegroups.com) for appropriate projects.

## Acknowledgements

The BVLC Caffe developers would like to thank NVIDIA for GPU donation, A9 and Amazon Web Services for a research grant in support of Caffe development and reproducible research in deep learning, and BVLC PI [Trevor Darrell](http://www.eecs.berkeley.edu/~trevor/) for guidance.

The BVLC members who have contributed to Caffe are (alphabetical by first name):
[Eric Tzeng](https://github.com/erictzeng), [Evan Shelhamer](http://imaginarynumber.net/), [Jeff Donahue](http://jeffdonahue.com/), [Jon Long](https://github.com/longjon), [Ross Girshick](http://www.cs.berkeley.edu/~rbg/), [Sergey Karayev](http://sergeykarayev.com/), [Sergio Guadarrama](http://www.eecs.berkeley.edu/~sguada/), and [Yangqing Jia](http://daggerfs.com/).

The open-source community plays an important and growing role in Caffe's development.
Check out the Github [project pulse](https://github.com/BVLC/caffe/pulse) for recent activity and the [contributors](https://github.com/BVLC/caffe/graphs/contributors) for the full list.

We sincerely appreciate your interest and contributions!
If you'd like to contribute, please read the [developing & contributing](development.html) guide.

Yangqing would like to give a personal thanks to the NVIDIA Academic program for providing GPUs, [Oriol Vinyals](http://www1.icsi.berkeley.edu/~vinyals/) for discussions along the journey, and BVLC PI [Trevor Darrell](http://www.eecs.berkeley.edu/~trevor/) for advice.
