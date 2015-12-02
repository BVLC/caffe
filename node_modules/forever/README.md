# forever

[![Version npm](https://img.shields.io/npm/v/forever.svg?style=flat-square)](https://www.npmjs.com/package/forever)[![npm Downloads](https://img.shields.io/npm/dm/forever.svg?style=flat-square)](https://www.npmjs.com/package/forever)[![Build Status](https://img.shields.io/travis/foreverjs/forever/master.svg?style=flat-square)](https://travis-ci.org/foreverjs/forever)[![Dependencies](https://img.shields.io/david/foreverjs/forever.svg?style=flat-square)](https://david-dm.org/foreverjs/forever)[![Inline docs](http://inch-ci.org/github/foreverjs/forever.svg?branch=master)](http://inch-ci.org/github/foreverjs/forever)

[![NPM](https://nodei.co/npm/forever.png?downloads=true&downloadRank=true)](https://nodei.co/npm/forever/)


A simple CLI tool for ensuring that a given script runs continuously (i.e. forever).

## Installation

``` bash
  $ [sudo] npm install forever -g
```

**Note:** If you are using forever _programmatically_ you should install [forever-monitor][0].

``` bash
  $ cd /path/to/your/project
  $ [sudo] npm install forever-monitor
```

## Usage
There are two ways to use forever: through the command line or by using forever in your code. **Note:** If you are using forever _programatically_ you should install [forever-monitor][0].

### Command Line Usage
You can use forever to run scripts continuously (whether it is written in node.js or not).

**Example**
```
forever start app.js
```

**Options**
```
  $ forever --help
  usage: forever [action] [options] SCRIPT [script-options]

  Monitors the script specified in the current process or as a daemon

  actions:
    start               Start SCRIPT as a daemon
    stop                Stop the daemon SCRIPT by Id|Uid|Pid|Index|Script
    stopall             Stop all running forever scripts
    restart             Restart the daemon SCRIPT
    restartall          Restart all running forever scripts
    list                List all running forever scripts
    config              Lists all forever user configuration
    set <key> <val>     Sets the specified forever config <key>
    clear <key>         Clears the specified forever config <key>
    logs                Lists log files for all forever processes
    logs <script|index> Tails the logs for <script|index>
    columns add <col>   Adds the specified column to the output in `forever list`
    columns rm <col>    Removed the specified column from the output in `forever list`
    columns set <cols>  Set all columns for the output in `forever list`
    cleanlogs           [CAREFUL] Deletes all historical forever log files

  options:
    -m  MAX          Only run the specified script MAX times
    -l  LOGFILE      Logs the forever output to LOGFILE
    -o  OUTFILE      Logs stdout from child script to OUTFILE
    -e  ERRFILE      Logs stderr from child script to ERRFILE
    -p  PATH         Base path for all forever related files (pid files, etc.)
    -c  COMMAND      COMMAND to execute (defaults to node)
    -a, --append     Append logs
    -f, --fifo       Stream logs to stdout
    -n, --number     Number of log lines to print
    --pidFile        The pid file
    --uid            Process uid, useful as a namespace for processes (must wrap in a string)
                     e.g. forever start --uid "production" app.js
                         forever stop production
    --sourceDir      The source directory for which SCRIPT is relative to
    --workingDir     The working directory in which SCRIPT will execute
    --minUptime      Minimum uptime (millis) for a script to not be considered "spinning"
    --spinSleepTime  Time to wait (millis) between launches of a spinning script.
    --colors         --no-colors will disable output coloring
    --plain          Disable command line colors
    -d, --debug      Forces forever to log debug output
    -v, --verbose    Turns on the verbose messages from Forever
    -s, --silent     Run the child script silencing stdout and stderr
    -w, --watch      Watch for file changes
    --watchDirectory Top-level directory to watch from
    --watchIgnore    To ignore pattern when watch is enabled (multiple option is allowed)
    --killSignal     Support exit signal customization (default is SIGKILL),
                     used for restarting script gracefully e.g. --killSignal=SIGTERM
    -h, --help       You're staring at it

  [Long Running Process]
    The forever process will continue to run outputting log messages to the console.
    ex. forever -o out.log -e err.log my-script.js

  [Daemon]
    The forever process will run as a daemon which will make the target process start
    in the background. This is extremely useful for remote starting simple node.js scripts
    without using nohup. It is recommended to run start with -o -l, & -e.
    ex. forever start -l forever.log -o out.log -e err.log my-daemon.js
        forever stop my-daemon.js
```

There are [several examples][1] designed to test the fault tolerance of forever. Here's a simple usage example:

``` bash
  $ forever -m 5 examples/error-on-timer.js
```

### JSON Configuration Files

In addition to passing forever the path to a script (along with accompanying options, described above), you may also pass forever the path to a JSON file containing these options. For example, consider an application with the following file structure:

```
.
├── forever
│   └── development.json
└── index.js

// forever/development.json
{
	// Comments are supported
    "uid": "app",
    "append": true,
    "watch": true,
    "script": "index.js",
    "sourceDir": "/home/myuser/app"
}
```

This application could be started with forever, as shown below:

``` bash
$ forever start ./forever/development.json
```

Absolute paths to such configuration files are also supported:

``` bash
$ forever start /home/myuser/app/forever/development.json
```

**Note:** Forever parses JSON configuration files using [shush](https://github.com/krakenjs/shush), allowing the use of in-line comments within such files.

#### Multi-App Configuration Files

JSON configuration files can also be used to define the startup options for *multiple* applications, as shown below.

```
[
  {
    // App1
    "uid": "app1",
    "append": true,
    "watch": true,
    "script": "index.js",
    "sourceDir": "/home/myuser/app1"
  },
  {
    // App2
    "uid": "app2",
    "append": true,
    "watch": true,
    "script": "index.js",
    "sourceDir": "/home/myuser/app2",
    "args": ["--port", "8081"]
  }
]
```

### Using In Your Code
The forever module exposes some useful methods to use in your code. Each method returns an instance of an EventEmitter which emits when complete. See the [forever cli commands][2] for sample usage.

**Remark:** As of `forever@0.6.0` processes will not automatically be available in `forever.list()`. In order to get your processes into `forever.list()` or `forever list` you must instantiate the `forever` socket server:

``` js
  forever.startServer(child);
```

This method takes multiple `forever.Monitor` instances which are defined in the `forever-monitor` dependency.

#### forever.load (config)
_Synchronously_ sets the specified configuration (config) for the forever module. There are two important options:

Option    | Description                                       | Default
-------   | ------------------------------------------------- | ---------
root      | Directory to put all default forever log files    | `forever.root`
pidPath   | Directory to put all forever *.pid files          | `[root]/pids`
sockPath  | Directory for sockets for IPC between workers     | `[root]/sock`
loglength | Number of logs to return in `forever tail`        | 100
columns   | Array of columns to display when `format` is true | `forever.config.get('columns')`
debug     | Boolean value indicating to run in debug mode     | false
stream    | Boolean value indicating if logs will be streamed | false

#### forever.start (file, options)
Starts a script with forever. The `options` object is what is expected by the `Monitor` of `forever-monitor`.

#### forever.startDaemon (file, options)
Starts a script with forever as a daemon. WARNING: Will daemonize the current process. The `options` object is what is expected by the `Monitor` of `forever-monitor`.

#### forever.stop (index)
Stops the forever daemon script at the specified index. These indices are the same as those returned by forever.list(). This method returns an EventEmitter that raises the 'stop' event when complete.

#### forever.stopAll (format)
Stops all forever scripts currently running. This method returns an EventEmitter that raises the 'stopAll' event when complete.

The `format` parameter is a boolean value indicating whether the returned values should be formatted according to the configured columns which can set with `forever columns` or programmatically `forever.config.set('columns')`.

#### forever.list (format, callback)
Returns a list of metadata objects about each process that is being run using forever. This method will return the list of metadata as such. Only processes which have invoked `forever.startServer()` will be available from `forever.list()`

The `format` parameter is a boolean value indicating whether the returned values should be formatted according to the configured columns which can set with `forever columns` or programmatically `forever.config.set('columns')`.

#### forever.tail (target, options, callback)
Responds with the logs from the target script(s) from `tail`. There are two options:

* `length` (numeric): is is used as the `-n` parameter to `tail`.
* `stream` (boolean): is is used as the `-f` parameter to `tail`.

#### forever.cleanUp ()
Cleans up any extraneous forever *.pid files that are on the target system. This method returns an EventEmitter that raises the 'cleanUp' event when complete.

#### forever.cleanLogsSync (processes)
Removes all log files from the root forever directory that do not belong to current running forever processes. Processes are the value returned from `Monitor.data` in `forever-monitor`.

#### forever.startServer (monitor0, monitor1, ..., monitorN)
Starts the `forever` HTTP server for communication with the forever CLI. **NOTE:** This will change your `process.title`. This method takes multiple `forever.Monitor` instances which are defined in the `forever-monitor` dependency.

### Logging and output file locations

By default `forever` places all of the files it needs into `/$HOME/.forever`. If you would like to change that location just set the `FOREVER_ROOT` environment variable when you are running forever:

```
FOREVER_ROOT=/etc/forever forever start index.js
```

Make sure that the user running the process has the appropriate privileges to read & write to this directory.

## Run Tests

``` bash
  $ npm test
```

#### License: MIT
#### Author: [Charlie Robbins](https://github.com/indexzero)
#### Contributors: [Fedor Indutny](https://github.com/indutny), [James Halliday](http://substack.net/), [Charlie McConnell](https://github.com/avianflu), [Maciej Malecki](https://github.com/mmalecki), [John Lancaster](http://jlank.com)

[0]: https://github.com/foreverjs/forever-monitor
[1]: https://github.com/foreverjs/forever-monitor/tree/master/examples
[2]: https://github.com/foreverjs/forever/blob/master/lib/forever/cli.js
