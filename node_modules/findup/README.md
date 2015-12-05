[![build status](https://secure.travis-ci.org/Filirom1/findup.png)](http://travis-ci.org/Filirom1/findup)
Find-up
=======

### Install

    npm install -g findup

### Usage

Find up a file in ancestor's dir


    .
    ├── config.json
    └── f
        └── e
            └── d
                └── c
                    ├── b
                    │   └── a
                    └── config.json

#### Async

findup(dir, fileName, callback)
findup(dir, iterator, callback) with `iterator(dir, cb)` where cb only accept `true` or `false`

    var findup = require('findup');


    findup(__dirname + '/f/e/d/c/b/a', 'config.json', function(err, dir){
      // if(e) e === new Error('not found')
      // dir === '/f/e/d/c'
    });

or

    findup(__dirname + '/f/e/d/c/b/a', function(dir, cb){
      require('path').exists(dir + '/config.json', cb);
    }, function(err, dir){
      // if(e) e === new Error('not found')
      // dir === '/f/e/d/c'
    });


#### EventEmitter

findup(dir, fileName)

    var findup = require('findup');
    var fup = findup(__dirname + '/f/e/d/c/b/a', 'config.json');

findup(dir, iterator) with `iterator(dir, cb)` where cb only accept `true` or `false`

    var findup = require('findup');
    var fup = findup(__dirname + '/f/e/d/c/b/a', function(dir, cb){
      require('path').exists(dir + '/config.json', cb);
    });

findup return an EventEmitter. 3 events are emitted: `found`, `error`, `end`

`found` event is emitted each time a file is found.

You can stop the traversing by calling `stop` manually.

    fup.on('found', function(dir){
      // dir === '/f/e/d/c'
      fup.stop();
    });

`error` event is emitted when error happens

    fup.on('error', function(e){
      // if(e) e === new Error('not found')
    });

`end` event is emitted at the end of the traversing or after `stop()` is
called.

    fup.on('end', function(){
      // happy end
    });

#### Sync

findup(dir, fileName)
findup(dir, iteratorSync) with `iteratorSync` return `true` or `false`

    var findup = require('findup');

    try{
      var dir = findup.sync(__dirname + '/f/e/d/c/b/a', 'config.json'); // dir === '/f/e/d/c'
    }catch(e){
      // if(e) e === new Error('not found')
    }

#### CLI

    npm install -g findup

    $ cd test/fixture/f/e/d/c/b/a/
    $ findup package.json
    /root/findup/package.json

Usage

    $ findup -h

    Usage: findup [FILE]

        --name, -n       The name of the file to found
        --dir, -d        The directoy where we will start walking up    $PWD
        --help, -h       show usage                                     false
        --verbose, -v    print log                                      false

### LICENSE MIT

### Read the tests :)
