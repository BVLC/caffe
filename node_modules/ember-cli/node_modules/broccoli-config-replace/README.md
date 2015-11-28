# broccoli-config-replace [![Build Status](https://travis-ci.org/ember-cli/broccoli-config-replace.svg)](https://travis-ci.org/ember-cli/broccoli-config-replace)

Simple templating using a config.json and regex patterns.

```js
new ConfigReplace(appNode, configNode, {
  // annotate the output. See broccoli-plugin
  annotations: true,

  // A list of files to parse:
  files: [
    'index.html',
    'tests/index.html'
  ],

  configPath: 'development.json',
  outputPath: 'dist/',
  patterns: [{
    match: /\{\{EMBER_ENV\}\}/g,
    replacement: function(config) { return config.EMBER_ENV; }
  }, {
    match: /\{\{APPLICATION_NAME\}\}/g,
    replacement: 'My Application'
  }]
});
```

If `replacement` is a function, it's passed the config object. Otherwise,
do a simple string replacement.

## Running tests

`npm test`
