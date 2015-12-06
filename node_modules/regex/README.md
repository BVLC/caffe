# Regex #

Native RegExp compatible regular expressions for JavaScript. Patterns may be composed of subexpressions and multiple expressions may be combined into a single expression.

## Installation ##

Regex may be installed on [node.js](http://nodejs.org/ "node.js") via the [node package manager](https://npmjs.org/ "npm") using the command `npm install regex`.

You may also install it on [RingoJS](http://ringojs.org/ "Home - RingoJS") using the command `ringo-admin install aaditmshah/regex`.

You may install it as a [component](https://github.com/component/component "component/component") for web apps using the command `component install aaditmshah/regex`.

## Usage ##

The `Regex` constructor is compatible with the native `RegExp` constructor. You may pass it a `regexp` object or a source string:

```javascript
var Regex = require("regex");
var regex = new Regex(/(a|b)*abb/);
```

## Methods ##

Like the native `RegExp` constructor instances of `Regex` have the following methods:

### test ###

The `test` method is used to simply test whether a string matches the regex pattern:

```javascript
regex.test("abb");   // true
regex.test("aabb");  // true
regex.test("babb");  // true
regex.test("aaabb"); // true
regex.test("ababb"); // true
regex.test("abba");  // false
regex.test("cabb");  // false
```

## Operations ##

The `Regex` constructor currently supports the following regex operations:

1. Concatenation
2. Alternation
3. Grouping
4. Closure

## Changelog ##

The following changes were made in this version of Regex:

### v0.1.0 ###

* Supports basic regex operations - concatenation, alternation, grouping and closure. No support for pattern composition or combining subexpressions yet.
