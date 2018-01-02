# Contributing

## Issues

Specific Caffe design and development issues, bugs, and feature requests are maintained by GitHub Issues.

_Please do not post usage, installation, or modeling questions, or other requests for help to Issues._
Use the [caffe-users list](https://groups.google.com/forum/#!forum/caffe-users) instead. This helps developers maintain a clear, uncluttered, and efficient view of the state of Caffe.

When reporting a bug, it's most helpful to provide the following information, where applicable:

* What steps reproduce the bug?
* Can you reproduce the bug using the latest [master](https://github.com/BVLC/caffe/tree/master), compiled with the `DEBUG` make option?
* What hardware and operating system/distribution are you running?
* If the bug is a crash, provide the backtrace (usually printed by Caffe; always obtainable with `gdb`).

Try to give your issue a title that is succinct and specific. The devs will rename issues as needed to keep track of them.

## Pull Requests

Caffe welcomes all contributions.

See the [contributing guide](http://caffe.berkeleyvision.org/development.html) for details.

Briefly: read commit by commit, a PR should tell a clean, compelling story of _one_ improvement to Caffe. In particular:

* A PR should do one clear thing that obviously improves Caffe, and nothing more. Making many smaller PRs is better than making one large PR; review effort is superlinear in the amount of code involved.
* Similarly, each commit should be a small, atomic change representing one step in development. PRs should be made of many commits where appropriate.
* Please do rewrite PR history to be clean rather than chronological. Within-PR bugfixes, style cleanups, reversions, etc. should be squashed and should not appear in merged PR history.
* Anything nonobvious from the code should be explained in comments, commit messages, or the PR description, as appropriate.
