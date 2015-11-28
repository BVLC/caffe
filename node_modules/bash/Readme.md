# bash

Utilities for using bash from node.js.

## API

### bash.escape(parameter)

Escapes the given `parameter` for bash. This is done by escaping all non
alpha-numeric / dash characters with a backslash.

Example:

```javascript
> bash.escape('hello world');
'Hello\\ World'
```

### bash.args(options, prefix, suffix)

Takes a list of `options` and turns them into an arguments string common to
most *nix programs.

Objects are turned into arguments:

```javascript
> bash.args({a: 1, b: 2}, '--', '=');
'--a=1 --b=2'
```

Values are escaped:

```javascript
> bash.args({foo: 'hi you'}, '--', '=');
'--foo=hi\\ you'
```

Array values turn into multiple arguments:

```javascript
> bash.args({a: [1, 2]}, '--', '=');
'--a=1 --a=2'
```

`null` / `true` values turn into flags:

```javascript
> bash.args({a: true, b: null}, '--', '=');
'--a --b'
```

Alternate suffix / prefix settings:

```javascript
> bash.args({a: 1, b: 2}, '-', ' ');
'-a 1 -b 2'
```

`options` can be an array as well:

```javascript
> bash.args([{a: 1}, {a: 2, b: 3}] '-', ' ');
'-a 1 -a 2 -b 3'
```

## License

This library is released under the MIT license.
