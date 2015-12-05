Did It Work?
============

***Did it Work?***: A simple process launcher that determines whether the process succeeded or failed.

Install
-------

    npm install did_it_work

Usage
-----

    var process = require('did_it_work');

    process('my_awesome_program')
      .goodIfMatches(/Ready/, 1000)
      .badIfMatches(/Error/)
      .good(function(){
        console.log('The program worked (because it spat out "Ready" to stdout within 1000ms).')
      })
      .bad(function(){
        console.log('The program didn\'t work (because it spat out "Error" to stdout, or the program exited with non-zero code, or it didn\'t spit out "Ready" within 1000ms)')
      })
      .complete(function(){
        console.log('In any case, the program exited')
      })

Use `spawn` instead of `exec`
-----------------------------

If you pass one string argument to the function, it will use `child_process.exec` to create the process. If, on the other hand, you need to use `child_process.spawn`, pass two arguments instead, the first being the executable and the second being an array of arguments. Example

    process('echo', ['hello', 'world'])
      .complete(function(stdout){
        console.log('The program returned ' + stdout)
      })

RTFT
----

To see more, read the tests in `tests.js`.