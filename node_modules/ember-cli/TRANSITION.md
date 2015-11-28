# Brocfile Transition

## Why

Broccoli historically assumed its brocfile was like a main file. Originally, we loaded the required config on-demand when the brocfile was invoked, unfortunately as the addon story evolved, we required this data before the brocfile was even invoked (or for tasks where no brocfile invocation was needed). This created a problem, and the quickest solution lead us to the current state of affairs, where we end up duplicating much internal state to satisfy both parts of the system.

This change occurred in an effort to unify both sides, rather then duplicating state, we can now pass that state in as an argument. Since the behavior of this file is a superset of a typical Brocfile, we opted to also change the name.

With this change, we are in a good position to fix more bugs, remove some much needed tech-debt, and continue to improve the internals without causing uneeded pain in userland.

Happy Hacking!

## How

Transitioning your Brocfile is fairly straight forward. Simply take the contents of your Brocfile and place it in the body of the function in the new `ember-cli-build.js` file.  Instead of using `module.exports` to return the tree simply have the function return the tree.  Ensure you pass the `defaults` to the EmberApp constructor along with any options you were passing to `EmberApp` in the Brocfile.  Internally these two objects will be merged from right to left.

Two steps:

1. Remove `Brocfile.js` and add contents into `ember-cli-build.js`
2. Pass `defaults` into the `EmberApp` constructor

## Before (Brocfile.js)

```
var EmberApp = require('ember-cli/lib/broccoli/ember-app');
var app = new EmberApp();
app.import("file1.js");
app.import("file2.js");
module.exports = app.toTree();
```

## After (ember-cli-build.js)
```
var EmberApp = require('ember-cli/lib/broccoli/ember-app');

module.exports = function(defaults) {
    var app = new EmberApp(defaults, {
        // Any other options
    });

    app.import("file1.js");
    app.import("file2.js");

    return app.toTree();
};
```
