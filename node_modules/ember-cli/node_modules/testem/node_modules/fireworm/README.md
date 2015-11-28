Fireworm
========

<img src="https://api.travis-ci.org/airportyh/fireworm.png">

Fireworm is a crawling file watcher.

## Install

    npm install fireworm

## Usage

``` js
var fireworm = require('fireworm')

// make a new file watcher
var fw = fireworm('start_dir')

// Add the files you want to watch for changes on (can be glob)
fw.add('lib/**/*.js')
fw.add('tests/**/*.js')

// ignore some patterns
fw.ignore('tests/dontcare/*.js')

// register for the `change` event
fw.on('change', function(filename){
    console.log(filename + ' just changed!')
})
```

## How is this different from other file watchers?

Fireworm works by crawling and re-crawling the relevant directories when necessary. Because of this, it can detect newly created files, new files in newly created directories, re-created files, and even new files within re-created directories too - as long as the file matches the paths you are watching.

As of version 0.5.2, fireworm uses `fs.watch` to watch directories only.

## API

### Constructor

`fireworm(dir, [options])` - returns a file watcher object for files within `dir`.

Options:

* `ignoreInitial:boolean` (default `false`) - do not fire `add` events when encountering files or directory for the initial time it is crawled.
* `skipDirEntryPatterns:array` (default `['node_modules', '.*']`) - when crawling directories, skip the directory entries which match these glob patterns.

### File Watcher Object

### Methods

* `add(filepath:string|array)` - add these file patterns (glob) to the watch list. The parameter(s) can be a variable number of strings or arrays of strings.
* `ignore(filepath:string|array)` - add these file patterns (glob) to the ignore list. The parameter(s) can be a variable number of strings or arrays of strings.
* `clear()` - clear all previously added match patterns or ignore patterns.

### Events

File watcher objects are [EventEmitters](http://nodejs.org/api/events.html#events_class_events_eventemitter), and can emit these events:

* `add` - fired when a matching file has been added to the FS. Parameter: `path` - the path to the file
* `change` - fired when a matching file has been modified. Parameter: `path` - the path to the file
* `remove` - fired when a matching file has been removed from the FS. Parameter: `path` - the path to the file

## License

(The MIT License)

Copyright (c) 2013 Toby Ho &lt;airportyh@gmail.com&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
