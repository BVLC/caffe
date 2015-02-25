---
title: Developing and Contributing
---
# Development

Caffe is developed with active participation of the community.<br>
The [BVLC](http://bvlc.eecs.berkeley.edu/) maintainers welcome all contributions!

The exact details of contributions are recorded by versioning and cited in our [acknowledgements](http://caffe.berkeleyvision.org/#acknowledgements).
This method is impartial and always up-to-date.

## License

Caffe is licensed under the terms in [LICENSE](https://github.com/BVLC/caffe/blob/master/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Copyright

Caffe uses a shared copyright model: each contributor holds copyright over their contributions to Caffe. The project versioning records all such contribution and copyright details.

If a contributor wants to further mark their specific copyright on a particular contribution, they should indicate their copyright solely in the commit message of the change when it is committed. Do not include copyright notices in files for this purpose.

### Documentation

This website, written with [Jekyll](http://jekyllrb.com/), functions as the official Caffe documentation -- simply run `scripts/build_docs.sh` and view the website at `http://0.0.0.0:4000`.

We prefer tutorials and examples to be documented close to where they live, in `readme.md` files.
The `build_docs.sh` script gathers all `examples/**/readme.md` and `examples/*.ipynb` files, and makes a table of contents.
To be included in the docs, the readme files must be annotated with [YAML front-matter](http://jekyllrb.com/docs/frontmatter/), including the flag `include_in_docs: true`.
Similarly for IPython notebooks: simply include `"include_in_docs": true` in the `"metadata"` JSON field.

Other docs, such as installation guides, are written in the `docs` directory and manually linked to from the `index.md` page.

We strive to provide provide lots of usage examples, and to document all code in docstrings.
We absolutely appreciate any contribution to this effort!

### The release cycle

- The `dev` branch receives all new development, including community contributions.
We aim to keep it in a functional state, but large changes do occur, and things do get broken every now and then.
Use only if you want the "bleeding edge".
- BVLC maintainers will periodically update the `master` branch with changes from `dev`, giving it a release tag ([releases so far](https://github.com/BVLC/caffe/releases)).
Use this if you want more stability.

### Issues & Pull Request Protocol

Use Github Issues to report [bugs], propose features, and ask development [questions].
Large-scale development work is guided by [milestones], which are sets of Issues selected for concurrent release (integration from `dev` to `master`).

Please note that since the core developers are largely researchers, we may work on a feature in isolation for some time before releasing it to the community, so as to claim honest academic contribution.
We do release things as soon as a reasonable technical report may be written, and we still aim to inform the community of ongoing development through Github Issues.

When you are ready to start developing your feature or fixing a bug, follow this protocol:

- Develop in [feature branches] with descriptive names.
    - For new development branch off `dev`.
    - For documentation and fixes for `master` branch off `master`.
- Bring your work up-to-date by [rebasing] onto the latest `dev` / `master`.
(Polish your changes by [interactive rebase], if you'd like.)
- [Pull request] your contribution to `BVLC/caffe`'s `dev` / `master` branch for discussion and review.
  - Make PRs *as soon as development begins*, to let discussion guide development.
  - A PR is only ready for merge review when it is a fast-forward merge, and all code is documented, linted, and tested -- that means your PR must include tests!
- When the PR satisfies the above properties, use comments to request maintainer review.

Below is a poetic presentation of the protocol in code form.

#### [Shelhamer's](https://github.com/shelhamer) “life of a branch in four acts”

Make the `feature` branch off of the latest `bvlc/dev`
```
git checkout dev
git pull upstream dev
git checkout -b feature
# do your work, make commits
```

Prepare to merge by rebasing your branch on the latest `bvlc/dev`
```
# make sure dev is fresh
git checkout dev
git pull upstream dev
# rebase your branch on the tip of dev
git checkout feature
git rebase dev
```

Push your branch to pull request it into `dev`
```
git push origin feature
# ...make pull request to dev...
```

Now make a pull request! You can do this from the command line (`git pull-request -b dev`) if you install [hub](https://github.com/github/hub).

The pull request of `feature` into `dev` will be a clean merge. Applause.

[bugs]: https://github.com/BVLC/caffe/issues?labels=bug&page=1&state=open
[questions]: https://github.com/BVLC/caffe/issues?labels=question&page=1&state=open
[milestones]: https://github.com/BVLC/caffe/issues?milestone=1
[Pull request]: https://help.github.com/articles/using-pull-requests
[interactive rebase]: https://help.github.com/articles/interactive-rebase
[rebasing]: http://git-scm.com/book/en/Git-Branching-Rebasing
[feature branches]: https://www.atlassian.com/git/workflows#!workflow-feature-branch

### Testing

Run `make runtest` to check the project tests. New code requires new tests. Pull requests that fail tests will not be accepted.

The `googletest` framework we use provides many additional options, which you can access by running the test binaries directly. One of the more useful options is `--gtest_filter`, which allows you to filter tests by name:

    # run all tests with CPU in the name
    build/test/test_all.testbin --gtest_filter='*CPU*'

    # run all tests without GPU in the name (note the leading minus sign)
    build/test/test_all.testbin --gtest_filter=-'*GPU*'

To get a list of all options `googletest` provides, simply pass the `--help` flag:

    build/test/test_all.testbin --help

### Style

- Follow [Google C++ style](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml) and [Google python style](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html) + [PEP 8](http://legacy.python.org/dev/peps/pep-0008/).
- Wrap lines at 80 chars.
- Remember that “a foolish consistency is the hobgoblin of little minds,” so use your best judgement to write the clearest code for your particular case.
- **Run `make lint` to check C++ code.**
