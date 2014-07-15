---
layout: default
---
# Caffe

Caffe is a deep learning framework developed with cleanliness, readability, and speed in mind.
It was created by [Yangqing Jia](http://daggerfs.com), and is in active development by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and by community contributors.
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

Check out our web image classification [demo](http://demo.caffe.berkeleyvision.org)!

## Why use Caffe?

**Clean architecture** enables rapid deployment.
Networks are specified in simple config files, with no hard-coded parameters in the code.
Switching between CPU and GPU is as simple as setting a flag -- so models can be trained on a GPU machine, and then used on commodity clusters.

**Readable & modifiable implementation** fosters active development.
In Caffe's first six months, it has been forked by over 300 developers on Github, and many have pushed significant changes.

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

- [Introductory slides](http://dl.caffe.berkeleyvision.org/caffe-presentation.pdf)<br />
Slides about the Caffe architecture, *updated 03/14*.
- [ACM MM paper](http://ucb-icsi-vision-group.github.io/caffe-paper/caffe.pdf)<br />
A 4-page report for the ACM Multimedia Open Source competition.
- [Installation instructions](/installation.html)<br />
Tested on Ubuntu, Red Hat, OS X.
* [Pre-trained models](/getting_pretrained_models.html)<br />
BVLC provides ready-to-use models for non-commercial use.
* [Developing & Contributing](/development.html)<br />
Guidelines for development and contributing to Caffe.

### Examples

{% for page in site.pages %}
{% if page.category == 'example' %}
- <div><a href="{{page.url}}">{{page.title}}</a><br />{{page.description}}</div>
{% endif %}
{% endfor %}

### Notebook examples

{% for page in site.pages %}
{% if page.category == 'notebook' %}
- <div><a href="http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/{{page.original_path}}">{{page.title}}</a><br />{{page.description}}</div>
{% endif %}
{% endfor %}

## Citing Caffe

Please cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
       Author = {Yangqing Jia},
       Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
       Year  = {2013},
       Howpublished = {\url{http://caffe.berkeleyvision.org/}
    }

If you do publish a paper where Caffe helped your research, we encourage you to update the [publications wiki](https://github.com/BVLC/caffe/wiki/Publications).
Citations are also tracked automatically by [Google Scholar](http://scholar.google.com/scholar?oi=bibs&hl=en&cites=17333247995453974016).

## Acknowledgements

Yangqing would like to thank the NVIDIA Academic program for providing GPUs, [Oriol Vinyals](http://www1.icsi.berkeley.edu/~vinyals/) for discussions along the journey, and BVLC PI [Trevor Darrell](http://www.eecs.berkeley.edu/~trevor/) for guidance.

A core set of BVLC members have contributed much new functionality and many fixes since the original release (alphabetical by first name):
[Eric Tzeng](https://github.com/erictzeng), [Evan Shelhamer](http://imaginarynumber.net/), [Jeff Donahue](http://jeffdonahue.com/), [Jon Long](https://github.com/longjon), [Ross Girshick](http://www.cs.berkeley.edu/~rbg/), [Sergey Karayev](http://sergeykarayev.com/), [Sergio Guadarrama](http://www.eecs.berkeley.edu/~sguada/).

Additionally, the open-source community plays a large and growing role in Caffe's development.
Check out the Github [project pulse](https://github.com/BVLC/caffe/pulse) for recent activity, and the [contributors](https://github.com/BVLC/caffe/graphs/contributors) for a sorted list.

We sincerely appreciate your interest and contributions!
If you'd like to contribute, please read the [developing & contributing](development.html) guide.

## Contacting us

All questions about installation, code, future development, and applications should be searched for and asked at [GitHub Issues](https://github.com/BVLC/caffe/issues).

If you have a proposal that may not be suited for public discussion *and an ability to act on it*, please email us [directly](mailto:caffe-dev@googlegroups.com).
Requests for features, explanations, or personal help will be ignored; post such matters publicly as issues.

Some developers may be able to provide [consulting services](mailto:caffe-coldpress@googlegroups.com) for appropriate projects.
