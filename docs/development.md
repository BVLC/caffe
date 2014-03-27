---
layout: default
title: Caffe
---

Developing & Contributing
=========================

Caffe is developed with active participation of the community by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu/).
We welcome all contributions!

The [contributing workflow](https://github.com/BVLC/caffe#development) is explained in the README. These guidelines cover development practices in Caffe. This is a work-in-progress.

**Development Flow**

- `master` is golden.
- `dev` is for new development: it is the branching point for features and the base of pull requests.
  * The history of `dev` is not rewritten.
  * Contributions are shepherded from `dev` to `master` by BVLC by merge.
- To err is human. Accidents are fixed by reverts.
- Releases are marked with tags on merge from `dev` to `master`.

**Issues & Pull Request Protocol**

0. Make issues for [bugs](https://github.com/BVLC/caffe/issues?labels=bug&page=1&state=open), tentative proposals, and [questions](https://github.com/BVLC/caffe/issues?labels=question&page=1&state=open).
1. Make PRs to signal development:
  a. Make PRs *as soon as development begins*. Create a feature branch, make your initial commit, push, and PR to let everyone know you are working on it and let discussion guide development instead of review development after-the-fact.
  b. When a proposal from the first step earns enough interest to warrant development, make a PR, and reference and close the old issue to direct the conversation to the PR.
2. When a PR is ready, comment to request a maintainer be assigned to review and merge to `dev`.

A PR is only ready for review when the code is committed, documented, linted, and tested!

**Documentation**: the documentation is bundled with Caffe in `docs/`. This includes the site you are reading now. Contributions should be documented both inline in code and through usage examples. New documentation is published by BVLC with each release and between releases as-needed.

We'd appreciate your contribution to the documentation effort!

**Testing**: run `make runtest` to check the project tests. New code requires new tests. Pull requests that fail tests will not be accepted.

The `googletest` framework we use provides many additional options, which you can access by running the test binaries directly. One of the more useful options is `--gtest_filter`, which allows you to filter tests by name:

    # run all tests with CPU in the name
    build/src/caffe/test/test_all.testbin --gtest_filter='*CPU*'

    # run all tests without GPU in the name (note the leading minus sign)
    build/src/caffe/test/test_all.testbin --gtest_filter=-'*GPU*'

To get a list of all options `googletest` provides, simply pass the `--help` flag:

    build/src/caffe/test/test_all.testbin --help

**Style**

- Follow [Google C++ style](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml) and [Google python style](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html) + [PEP 8](http://legacy.python.org/dev/peps/pep-0008/).
- Wrap lines at 80 chars.
- Remember that “a foolish consistency is the hobgoblin of little minds,” so use your best judgement to write the clearest code for your particular case.

**Lint**: run `make lint` to check C++ code.

**Copyright**: assign copyright jointly to BVLC and contributors like so:

    // Copyright 2014 BVLC and contributors.

The exact details of contributions are recorded by versioning and cited in our [acknowledgements](http://caffe.berkeleyvision.org/#acknowledgements). This method is impartial and always up-to-date.
