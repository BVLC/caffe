# forever-monitor [![Build Status](https://secure.travis-ci.org/foreverjs/forever-monitor.png)](http://travis-ci.org/foreverjs/forever-monitor)

The core monitoring functionality of forever without the CLI

## Usage
You can also use forever from inside your own node.js code.

``` js
  var forever = require('forever-monitor');

  var child = new (forever.Monitor)('your-filename.js', {
    max: 3,
    silent: true,
    args: []
  });

  child.on('exit', function () {
    console.log('your-filename.js has exited after 3 restarts');
  });

  child.start();
```

### Spawning a non-node process
You can spawn non-node processes too. Either set the `command` key in the
`options` hash or pass in an `Array` in place of the `file` argument like this:

``` js
  var forever = require('forever-monitor');
  var child = forever.start([ 'perl', '-le', 'print "moo"' ], {
    max : 1,
    silent : true
  });
```

### Options available when using Forever in node.js
There are several options that you should be aware of when using forever. Most of this configuration is optional.

``` js
  {
    //
    // Basic configuration options
    //
    'silent': false,            // Silences the output from stdout and stderr in the parent process
    'uid': 'your-UID',          // Custom uid for this forever process. (default: autogen)
    'pidFile': 'path/to/a.pid', // Path to put pid information for the process(es) started
    'max': 10,                  // Sets the maximum number of times a given script should run
    'killTree': true,           // Kills the entire child process tree on `exit`

    //
    // These options control how quickly forever restarts a child process
    // as well as when to kill a "spinning" process
    //
    'minUptime': 2000,     // Minimum time a child process has to be up. Forever will 'exit' otherwise.
    'spinSleepTime': 1000, // Interval between restarts if a child is spinning (i.e. alive < minUptime).

    //
    // Command to spawn as well as options and other vars
    // (env, cwd, etc) to pass along
    //
    'command': 'perl',         // Binary to run (default: 'node')
    'args':    ['foo','bar'],  // Additional arguments to pass to the script,
    'sourceDir': 'script/path',// Directory that the source script is in

    //
    // Options for restarting on watched files.
    //
    'watch': true,               // Value indicating if we should watch files.
    'watchIgnoreDotFiles': null, // Whether to ignore file starting with a '.'
    'watchIgnorePatterns': null, // Ignore patterns to use when watching files.
    'watchDirectory': null,      // Top-level directory to watch from.

    //
    // All or nothing options passed along to `child_process.spawn`.
    //
    'spawnWith': {
      customFds: [-1, -1, -1], // that forever spawns.
      setsid: false,
      uid: 0, // Custom UID
      gid: 0  // Custom GID
    },

    //
    // More specific options to pass along to `child_process.spawn` which
    // will override anything passed to the `spawnWith` option
    //
    'env': { 'ADDITIONAL': 'CHILD ENV VARS' },
    'cwd': '/path/to/child/working/directory',

    //
    // Log files and associated logging options for this instance
    //
    'logFile': 'path/to/file', // Path to log output from forever process (when daemonized)
    'outFile': 'path/to/file', // Path to log output from child stdout
    'errFile': 'path/to/file', // Path to log output from child stderr

    //
    // ### function parseCommand (command, args)
    // #### @command {String} Command string to parse
    // #### @args    {Array}  Additional default arguments
    //
    // Returns the `command` and the `args` parsed from
    // any command. Use this to modify the default parsing
    // done by 'forever-monitor' around spaces.
    //
    'parser': function (command, args) {
      return {
        command: command,
        args:    args
      };
    }
  }
```

### Events available when using an instance of Forever in node.js
Each forever object is an instance of the node.js core EventEmitter. There are several core events that you can listen for:

* **error**   _[err]:_             Raised when an error occurs
* **start**   _[process, data]:_   Raised when the target script is first started.
* **stop**    _[process]:_         Raised when the target script is stopped by the user
* **restart** _[forever]:_         Raised each time the target script is restarted
* **exit**    _[forever]:_         Raised when the target script actually exits (permanently).
* **stdout**  _[data]:_            Raised when data is received from the child process' stdout
* **stderr**  _[data]:_            Raised when data is received from the child process' stderr

### Typical console output

When running the forever CLI tool, it produces debug outputs about which files have changed / how processes exited / etc. To get a similar behaviour with `forever-monitor`, add the following event listeners:

```js
var child = new (forever.Monitor)('your-filename.js');

child.on('watch:restart', function(info) {
    console.error('Restaring script because ' + info.file + ' changed');
});

child.on('restart', function() {
    console.error('Forever restarting script for ' + child.times + ' time');
});

child.on('exit:code', function(code) {
    console.error('Forever detected script exited with code ' + code);
});
```

## Installation

``` bash
  $ npm install forever-monitor
```

## Run Tests

``` bash
  $ npm test
```

#### License: MIT
#### Author: [Charlie Robbins](http://github.com/indexzero)
#### Contributors: [Fedor Indutny](http://github.com/indutny), [James Halliday](http://substack.net/), [Charlie McConnell](http://github.com/avianflu), [Maciej Malecki](http://github.com/mmalecki)
