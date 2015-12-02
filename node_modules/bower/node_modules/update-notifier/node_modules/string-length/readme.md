# string-length [![Build Status](https://travis-ci.org/sindresorhus/string-length.svg?branch=master)](https://travis-ci.org/sindresorhus/string-length)

> Get the real length of a string - by correctly counting astral symbols and ignoring [ansi escape codes](https://github.com/sindresorhus/strip-ansi)

`String#length` errornously counts [astral symbols](http://www.tlg.uci.edu/~opoudjis/unicode/unicode_astral.html) as two characters.


## Install

```
$ npm install --save string-length
```


## Usage

```js
'🐴'.length;
//=> 2

stringLength('🐴');
//=> 1

stringLength('\u001b[1municorn\u001b[22m');
//=> 7
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
