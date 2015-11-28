# Getting Involved

There are many ways to contribute to the project, you can fix issues,
improve documentation or work on any of the features on the
[wish list](https://github.com/ember-cli/ember-cli/wiki/Wish-List).

# Questions

This is the issue tracker for `ember-cli`. The community uses this site
to collect and track bugs and discussions of new features. If you are
having difficulties using `ember-cli` or have a question about usage
please ask a question on Stack Overflow: http://stackoverflow.com/questions/ask?tags=ember-cli

# Issues

Think you've found a bug or have a new feature to suggest? Let us know!

## Reporting a Bug

1. Update to the most recent master release if possible. We may have already
   fixed your bug.

2. Search for similar issues. It's possible somebody has encountered this bug
   already.

3. Provide a demo that specifically shows the problem. This demo should be fully
   operational with the exception of the bug you want to demonstrate. The more
   pared down, the better. Issues with demos are prioritized.

4. Your issue will be verified. The provided demo will be tested for
   correctness. The ember-cli team will work with you until your issue can be
   verified.

5. Keep up to date with feedback from the ember-cli team on your ticket. Your
   ticket may be closed if it becomes stale.

6. If possible, submit a Pull Request with a failing test. Better yet, take
   a stab at fixing the bug yourself if you can!

The more information you provide, the easier it is for us to validate that
there is a bug and the faster we'll be able to take action.

## Requesting a Feature

1. Search Issues for similar feature requests. It's possible somebody has
   already asked for this feature or provided a pull request that we're still
   discussing.

2. Provide a clear and detailed explanation of the feature you want and why it's
   important to add. Keep in mind that we want features that will be useful to
   the majority of our users and not just a small subset. If you're just
   targeting a minority of users, consider writing an addon library for
   `ember-cli`.

3. If the feature is complex, consider writing some initial documentation for
   it. If we do end up accepting the feature it will need to be documented and
   this will also help us to understand it better ourselves.

4. Attempt a Pull Request. If you are willing to help the project out, you can
   submit a Pull Request. We always have more work to do than time to do it. If
   you can write some code then that will speed the process along.

# Pull Requests

We love pull requests. Here's a quick guide:

1. Fork the repo.

2. Ensure you have the development requirements:

   * node (0.12 recommended) or io.js (1.x) -- *do not install node using sudo*
   * npm (2.x)
   * phantomjs

3. Run the tests. We only take pull requests with passing tests, and it's great
   to know that you have a clean slate: `npm install && npm run test-all`.

4. Add a test for your change. Only refactoring and documentation changes
   require no new tests. If you are adding functionality or fixing a bug, we
   need a test!

5. Make the test pass.

6. Commit your changes. If your pull request fixes an issue specify it in the
   commit message. Here's an example: `git commit - m "Close #52  Fix
   generators"`

7. Push to your fork and submit a pull request. In the pull-request title,
   please prefix it with one of our tags: BUGFIX, FEATURE, ENHANCEMENT or
   INTERNAL

   * FEATURE and ENHANCEMENT tags are for things that users are interested in.
     Avoid super technical talk. Craft a concise description of the change.
     - FEATURE tag is a standalone new addition, an example of this would be
       adding a new command
     - ENHANCEMENT tag is an improvement on an existing feature
   * BUGFIX tag is a link to a bug + a link to a patch.
   * INTERNAL tag is an internal log of changes.

   In the description, please provide us with some explanation of why you made
   the changes you made. For new features make sure to explain a standard use
   case to us.

   If a change requires a user to change their configuration, `bower.json`,
   `package.json` or `Brocfile.js` also add a BREAKING tag within the brackets
   before any other tags (example [BREAKING BUGFIX]).

We try to be quick about responding to tickets but sometimes we get a bit
backlogged. If the response is slow, try to find someone on IRC(#ember-cli) to
give the ticket a review.

Some things that will increase the chance that your pull request is accepted,
taken straight from the Ruby on Rails guide:

* Use Node idioms and helpers.
* Include tests that fail without your code, and pass with it.
* Update the documentation, the surrounding one, examples elsewhere, guides,
  whatever is affected by your contribution.

#### Syntax

* Two spaces, no tabs.
* No trailing whitespace. Blank lines should not have any space.
* Follow the conventions you see used in the source already.

#### Inline Documentation Guidelines

All inline documentation is written using YUIDoc. Follow these rules when
updating or writing new documentation:

1. All code blocks must be fenced.
2. All code blocks must have a language declared.
3. All code blocks must be valid code for syntax highlighting.
4. All examples in code blocks must be aligned.
5. Use two spaces between the code and the example: `foo(); // result`.
6. All references to code words must be enclosed in backticks.
7. Prefer a single space between sentences.
8. Wrap long markdown blocks > 80 characters.
9. Don't include blank lines after `@param` definitions.

#### Website

The codebase for the website [ember-cli.com](http://ember-cli.com) is located
at: https://github.com/ember-cli/ember-cli/tree/gh-pages

#### Code Words

* `thisPropertyName`
* `Global.Class.attribute`
* `thisFunction()`
* `Global.CONSTANT_NAME`
* `true`, `false`, `null`, `undefined` (when referring to programming values)

And in case we didn't emphasize it enough: **we love tests!**

NOTE: Partially copied from https://raw.githubusercontent.com/emberjs/ember.js/master/CONTRIBUTING.md

# Docs

Have you got enough knowledge in a specific feature and want to help with docs?
Ember-cli documentation lives at the branch
[gh-pages](https://github.com/ember-cli/ember-cli/tree/gh-pages).

Feel free to contribute and help us to keep an updated, clear and complete
documentation.
