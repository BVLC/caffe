# cross-spawn-async [![Build Status](https://travis-ci.org/IndigoUnited/node-cross-spawn-async.svg?branch=master)](https://travis-ci.org/IndigoUnited/node-cross-spawn-async) [![Build status](https://ci.appveyor.com/api/projects/status/9rglfmcmxuu5lbcq/branch/master?svg=true)](https://ci.appveyor.com/project/satazor/node-cross-spawn-async/branch/master)

A cross platform solution to node's spawn.


## Installation

`$ npm install cross-spawn-async`


## Why

Node has issues when using spawn on Windows:

- It ignores [PATHEXT](https://github.com/joyent/node/issues/2318)
- It does not support [shebangs](http://pt.wikipedia.org/wiki/Shebang)
- It does not allow you to run `del` or `dir`
- It does not properly escape arguments with spaces or special characters

All these issues are handled correctly by `cross-spawn-async`.
There are some known modules, such as [win-spawn](https://github.com/ForbesLindesay/win-spawn), that try to solve this but they are either broken or provide faulty escaping of shell arguments.


## Usage

Exactly the same way as node's [`spawn`](https://nodejs.org/api/child_process.html#child_process_child_process_spawn_command_args_options), so it's a drop in replacement.

```javascript
var spawn = require('cross-spawn-async');

var process = spawn('npm', ['list', '-g', '-depth' '0'], { stdio: 'inherit' });
```


## Tests

`$ npm test`


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
