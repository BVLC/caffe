Good pull requests - patches, improvements, new features - are a fantastic
help.

If you've spotted any small, obvious errors and want to help out by patching it,
that will be much appreciated.

If your contribution involves a significant amount of work or substantial
changes to any part of the project, please open a "contribution enquiry" issue
first to check that the work is wanted or matches the goals of the project.

All pull requests should remain focused in scope and avoid containing unrelated
commits.

Please follow this process; it's the best way to get your work included in the
project:

1. [Fork](http://help.github.com/fork-a-repo/) the project.

2. Clone your fork (`git clone
   git@github.com:<your-username>/<repo-name>.git`).

3. Add an `upstream` remote (`git remote add upstream
   git://github.com/<upsteam-owner>/<repo-name>.git`).

4. Get the latest changes from upstream (e.g. `git pull upstream
   <dev-branch>`).

5. Create a new topic branch to contain your feature, change, or fix (`git
   checkout -b <topic-branch-name>`).

6. Create the needed tests to ensure that your contribution is not broken in the future.
   If you are creating a small fix or patch to an existing feature, just a simple test
   will do, if it is a brand new feature, make sure to create a new test suite.

7. Make sure that your changes adhere to the current coding conventions used
   throughout the project - indentation, accurate comments, etc.

8. Commit your changes in logical chunks; use git's [interactive
   rebase](https://help.github.com/articles/interactive-rebase) feature to tidy
   up your commits before making them public. Please adhere to these [git commit
   message guidelines](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
   or your pull request is unlikely be merged into the main project.

9. Locally merge (or rebase) the upstream branch into your topic branch.

10. Push your topic branch up to your fork (`git push origin
   <topic-branch-name>`).

11. [Open a Pull Request](http://help.github.com/send-pull-requests/) with a
    clear title and description.

If you have any other questions about contributing, please feel free to contact
me.