# Contributing

## Issues

Specific Caffe design and development issues, bugs, and feature requests are maintained by GitHub Issues.

*Please do not post installation, build, usage, or modeling questions, or other requests for help to Issues.*
Use the [caffe-users list](https://groups.google.com/forum/#!forum/caffe-users) instead.
This helps developers maintain a clear, uncluttered, and efficient view of the state of Caffe.

When reporting an issue, it's most helpful to provide the following information, where applicable:
* What steps reproduce the problem?
* Can you reproduce it using the latest [master](https://github.com/BVLC/caffe/tree/master), compiled with the `DEBUG` make option?
* What hardware and software are you running? In particular:
	* GPU make and model, if relevant,
	* operating system/distribution,
	* compiler; please also post which version (for example, if you're using GCC, run `gcc --version` to check which one),
	* CUDA version, if applicable (run `nvcc --version` to check),
	* cuDNN version, if applicable (version number is stored in `cudnn.h`, look for lines containing `CUDNN_MAJOR`, `CUDNN_MINOR` and `CUDNN_PATCHLEVEL`),
	* BLAS library,
	* Python version, if relevant,
	* MATLAB version, if relevant.
* What steps have you already tried to solve the issue? What didn't work and how?
* If the bug is a crash, provide the backtrace (usually printed by Caffe; always obtainable with `gdb`).
* If you are reporting a build error that seems to be due to a bug in Caffe, please attach your build configuration (either Makefile.config or CMakeCache.txt) and the output of the make (or cmake) command.

If only a small portion of the code/log is relevant to your issue, you may paste it directly into the post, preferably using Markdown syntax for code block: triple backtick ( \` ) to open/close a block.
In other cases (multiple files, or long files), please **attach** them to the post - this greatly increases readability.

Try to give your issue a title that is succinct and specific. The devs will rename issues as needed to keep track of them.

## Pull Requests

Caffe welcomes all contributions.

See the [contributing guide](http://caffe.berkeleyvision.org/development.html) for details.

Briefly: read commit by commit, a PR should tell a clean, compelling story of _one_ improvement to Caffe. In particular:

* A PR should do one clear thing that obviously improves Caffe, and nothing more. Making many smaller PRs is better than making one large PR; review effort is superlinear in the amount of code involved.
* Similarly, each commit should be a small, atomic change representing one step in development. PRs should be made of many commits where appropriate.
* Please do rewrite PR history to be clean rather than chronological. Within-PR bugfixes, style cleanups, reversions, etc. should be squashed and should not appear in merged PR history.
* Anything nonobvious from the code should be explained in comments, commit messages, or the PR description, as appropriate.
