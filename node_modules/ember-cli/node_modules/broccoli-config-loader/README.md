# broccoli-config-loader [![Build Status](https://travis-ci.org/ember-cli/broccoli-config-loader.svg)](https://travis-ci.org/ember-cli/broccoli-config-loader)

This plugin writes the environment files for an [ember-cli][] [Project][].

```js
new ConfigLoader('./config', {
  // annotate the output. See broccoli-plugin
  annotations: true,
  // write environments/development.json
  env: 'development',
  // if true, also write environments/test.json
  tests: true,
  // an ember-cli project
  project: new Project(...)
});
```

## Running tests

`npm test`.

[ember-cli]: http://github.com/ember-cli/ember-cli
[project]: https://github.com/ember-cli/ember-cli/blob/master/lib/models/project.js
