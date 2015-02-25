---
title: Deep Learning Framework
---

# Caffe

Caffe is a deep learning framework developed with cleanliness, readability, and speed in mind.
It was created by [Yangqing Jia](http://daggerfs.com) during his PhD at UC Berkeley, and is in active development by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and by community contributors.
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

Check out our web image classification [demo](http://demo.caffe.berkeleyvision.org)!

## Why use Caffe?

**Clean architecture** enables rapid deployment.
Networks are specified in simple config files, with no hard-coded parameters in the code.
Switching between CPU and GPU is as simple as setting a flag -- so models can be trained on a GPU machine, and then used on commodity clusters.

**Readable & modifiable implementation** fosters active development.
In Caffe's first year, it has been forked by over 600 developers on Github, and many have pushed significant changes.

**Speed** makes Caffe perfect for industry use.
Caffe can process over **40M images per day** with a single NVIDIA K40 or Titan GPU\*.
That's 5 ms/image in training, and 2 ms/image in test.
We believe that Caffe is the fastest CNN implementation available.

**Community**: Caffe already powers academic research projects, startup prototypes, and even large-scale industrial applications in vision, speech, and multimedia.
There is an active discussion and support community on [Github](https://github.com/BVLC/caffe/issues).

<p class="footnote" markdown="1">
\* When files are properly cached, and using the ILSVRC2012-winning [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model.
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
* [API Documentation](/doxygen/)<br>
Developer documentation automagically generated from code comments.

### Examples

{% assign examples = site.pages | where:'category','example' | sort: 'priority' %}
{% for page in examples %}
- <div><a href="{{page.url}}">{{page.title}}</a><br>{{page.description}}</div>
{% endfor %}

### Notebook examples

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

## Acknowledgements

The BVLC Caffe developers would like to thank NVIDIA for GPU donation, A9 and Amazon Web Services for a research grant in support of Caffe development and reproducible research in deep learning, and BVLC PI [Trevor Darrell](http://www.eecs.berkeley.edu/~trevor/) for guidance.

The BVLC members who have contributed to Caffe are (alphabetical by first name):
[Eric Tzeng](https://github.com/erictzeng), [Evan Shelhamer](http://imaginarynumber.net/), [Jeff Donahue](http://jeffdonahue.com/), [Jon Long](https://github.com/longjon), [Ross Girshick](http://www.cs.berkeley.edu/~rbg/), [Sergey Karayev](http://sergeykarayev.com/), [Sergio Guadarrama](http://www.eecs.berkeley.edu/~sguada/), and [Yangqing Jia](http://daggerfs.com/).

The open-source community plays an important and growing role in Caffe's development.
Check out the Github [project pulse](https://github.com/BVLC/caffe/pulse) for recent activity, and the [contributors](https://github.com/BVLC/caffe/graphs/contributors) for a sorted list.

We sincerely appreciate your interest and contributions!
If you'd like to contribute, please read the [developing & contributing](development.html) guide.

Yangqing would like to give a personal thanks to the NVIDIA Academic program for providing GPUs, [Oriol Vinyals](http://www1.icsi.berkeley.edu/~vinyals/) for discussions along the journey, and BVLC PI [Trevor Darrell](http://www.eecs.berkeley.edu/~trevor/) for advice.

## Contacting us

All questions about usage, installation, code, and applications should be searched for and asked on the [caffe-users mailing list](https://groups.google.com/forum/#!forum/caffe-users).

All development discussion should be carried out at [GitHub Issues](https://github.com/BVLC/caffe/issues).

If you have a proposal that may not be suited for public discussion *and an ability to act on it*, please email us [directly](mailto:caffe-dev@googlegroups.com).
Requests for features, explanations, or personal help will be ignored; post such matters publicly as issues.

The core Caffe developers may be able to provide [consulting services](mailto:caffe-coldpress@googlegroups.com) for appropriate projects.
