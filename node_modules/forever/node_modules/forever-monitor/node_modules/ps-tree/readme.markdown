#ps-tree

sometimes you cannot kill child processes like you would expect,  
this a feature of UNIX.

>in UNIX,  a process may terminate by using the exit call, and it's parent process may wait for that event by using the wait system call. the wait system call returns the process identifier of a terminated child, so that the parent tell which of the possibly many children has terminated. If the parent terminates, however, all it's children have assigned as their new parent the init process. Thus, the children still have a parent to collect their status and execution statistics.  
> (from "operating system concepts")

solution: use `ps-tree` to get all processes that a child_process may have started, so that they may all be terminated.

``` js
  var cp = require('child_process'),
      psTree = require('ps-tree')
    
  var child = cp.exec("node -e 'while (true);'",function () {...})

  child.kill() //this will not actually kill the child it will kill the `sh` process.

```

wtf? it's because exec actually works like this:

``` js
function exec (cmd, cb) {
  spawn('sh', ['-c', cmd])
  ...
}

```

sh starts parses the command string and starts processes, and waits for them to terminate.  
but exec returns a process object with the pid of the sh process.  
but since it is in `wait` mode killing it does not kill the children.

used ps tree like this:

``` js

  var cp = require('child_process'),
      psTree = require('ps-tree')
    
  var child = cp.exec("node -e 'while (true);'",function () {...})

  psTree(child.pid, function (err, children) {
    cp.spawn('kill', ['-9'].concat(children.map(function (p) {return p.PID})))
  })

```