var util     =   require('util');
var spawn    =   require('child_process').spawn;
var child    =   spawn('python',['-u','-i']);
var cmdQueue =   new Array();


child.stdout.on('data', handleStdout);
child.stderr.on('data', handleStderr);
child.on('exit', handleExit);


function handleStdout(data) {
  var datastr = data.toString('utf8');
  var finished = false;
  if (datastr.match(/Command Start\n/)) {
    datastr = datastr.replace(/Command Start\n/,'');
  } 
  if (datastr.match(/Command End\n/)) {
    datastr = datastr.replace(/Command End\n/,'');
    finished = true;
  }
  if (cmdQueue.length > 0) {
    cmdQueue[0].data+=datastr;
  } 
  if (finished) {
    cmd = cmdQueue.shift();
    if (cmd && cmd.command) {
      if (undefined != typeof cmd.callback) {
        cmd.callback(null, cmd.data);
        processQueue();
      }
    }
  } 
};


function handleStderr(data) {
  processQueue();
};

function processQueue() {
  if (cmdQueue.length > 0 && cmdQueue[0].state === 'pending') {
    cmdQueue[0].state = 'processing';
    child.stdin.write(cmdQueue[0].command, encoding='utf8');
  }
};


function handleExit(code) {
  console.log('child process exited with code ' + code);
  process.exit();
};


this.shell = function (command, callback) {
  command = 'print "Command Start"; ' + command + '\nprint "Command End"';
  if (command.charAt[command.length-1]!='\n') command += '\n';
  cmdQueue.push({'command':command, 'callback':callback, 'data': '', state: 'pending'});
  processQueue();
};
