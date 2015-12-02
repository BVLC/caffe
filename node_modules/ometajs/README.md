```
   ____  __  ___     __            _______
  / __ \/  |/  /__  / /_____ _    / / ___/
 / / / / /|_/ / _ \/ __/ __ `/_  / /\__ \
/ /_/ / /  / /  __/ /_/ /_/ / /_/ /___/ /
\____/_/  /_/\___/\__/\__,_/\____//____/
```

## OMetaJS [![Build Status](https://secure.travis-ci.org/veged/ometa-js.png)](http://travis-ci.org/veged/ometa-js)

OMetaJS is a JavaScript implementation of OMeta, an object-oriented language
for pattern matching.

This is a *node.js* module for developing and using such pattern matching
grammars.

## Installation

### Installing npm (node package manager)

``` bash
$ curl http://npmjs.org/install.sh | sh
```

### Installing ometajs

``` bash
$ [sudo] npm install ometajs -g
```

**Note:** If you are using ometajs _programmatically_ you should not install
it globally.

``` bash
$ cd /path/to/your/project
$ npm install ometajs
```

## Usage

### Command line

```bash
$ ometajs2js --help

Usage:
  ometajs2js [OPTIONS] [ARGS]


Options:
  -h, --help : Help
  -v, --version : Version
  -i INPUT, --input=INPUT : Input file (default: stdin)
  -o OUTPUT, --output=OUTPUT : Output file (default: stdout)
  --root=ROOT : Path to root module (default: ometajs)
```

`ometajs2js` will take input `*.ometajs` file and produce a [CommonJS][0]-
compatible javascript file.

You may also `require('*.ometajs')` files directly without compilation.
(OMetaJS is patching `require.extensions` as [CoffeeScript][1] does).

### Usage as CommonJS module

```javascript
var ometajs = require('ometajs');

var ast = ometajs.grammars.BSJSParser.matchAll('var x = 1', 'topLevel'),
    code = ometajs.grammars.BSJSTranslator.matchAll([ast], 'trans');
```

### Example grammar

```javascript
ometa Simple {
  top = [#simple] -> 'ok'
}
```

[More information][5] about OMetaJS syntax.

### Use cases

Quickly prototype and buildyour own parser/language. Process/traverse complex
[AST][2].

Some projects that are using OMetaJS:

 -   [XJST](http://github.com/veged/xjst)
 -   [ShmakoWiki](http://github.com/veged/shmakowiki/)
 -   [OmetaHighlighter](http://github.com/veged/ometa-highlighter)

### More information

To study OMetaJS or ask questions about its core you can reach out to the
[original repository][3] author [Alessandro Warth][4] or me.

Here is the [documented code][5].

#### Contributors

* [Alessandro Warth][4]
* [Takashi Yamamiya](https://github.com/propella)
* [Sergey Berezhnoy](https://github.com/veged)
* [Nikita Vasilyev](https://github.com/NV)
* [Fedor Indutny](https://github.com/indutny)
* [Jason Davies](https://github.com/jasondavies)

[0]: http://www.commonjs.org/
[1]: http://coffeescript.org/
[2]: http://en.wikipedia.org/wiki/Abstract_syntax_tree
[3]: http://www.tinlizzie.org/ometa/
[4]: http://github.com/alexwarth
[5]: http://veged.github.com/ometa-js/
