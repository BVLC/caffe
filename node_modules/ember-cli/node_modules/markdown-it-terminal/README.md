markdown-it-terminal
===
[![Build Status](https://travis-ci.org/trabus/markdown-it-terminal.svg)](https://travis-ci.org/trabus/markdown-it-terminal)

This is a plugin to provide ansi terminal output for markdown-it. It is heavily inspired by [marked-terminal](https://github.com/mikaelbr/marked-terminal), a terminal renderer for the marked library.

__This library is not officially supported by markdown-it.__

## Install

`npm install markdown-it markdown-it-terminal`

## Usage

`markdown-it` provides a method for extending it with plugins.
```js
var markdown = require('markdown-it');
var terminal = require('markdown-it-terminal');

markdown.use(terminal);
```

You can override the default options if you choose.
```js
var styles   = require('ansi-styles');
var markdown = require('markdown-it');
var terminal = require('markdown-it-terminal');

var options = {
  styleOptions: {
    code: styles.green
  }
}
markdown.use(terminal, options);
// inline code now prints in green instead of the default yellow
```

## Options
`markdown-it-terminal` takes several options, most of which are to override existing defaults.
```js
var options = {
  styleOptions:{},
  highlight: require('cardinal').highlight,
  unescape: true
}
```

### styleOptions
Styles are defined per token, and make use of the `ansi-styles` library, which provides a number of open and close values for ansi codes.

In the most basic implementation, you can simply provide a supported style like so:
```js
var styles   = require('ansi-styles');

var options = {
  styleOptions: {
    code: styles.green
  }
}
```
`markdown-it-terminal` exposes a utility method to build compound styles, using an array of style names (must be supported by `ansi-styles`).

```js
var styles   = require('ansi-styles');
var terminal = require('markdown-it-terminal');

var options = {
  styleOptions: {
    code: terminal.compoundStyle(['green','underline'])
  }
}
```

The following tokens can be overridden through styleOptions:
* code
* blockquote
* html
* heading
* firstHeading
* hr
* listitem
* table
* paragraph
* strong
* em
* codespan
* del
* link
* href

### highlight
Highlight function to parse code blocks. Should be a function that takes a string and outputs a formatted string.

### unescape
Unescape content, `true` by default.

## Highlighting
`markdown-it-terminal` uses the [cardinal](https://github.com/thlorenz/cardinal) library 
for code highlight support by default.

## Windows Support
Because ansi is not supported on cmd.exe, `markdown-it-terminal` only works on Windows shells with ansi support.