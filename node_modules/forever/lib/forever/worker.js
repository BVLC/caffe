var events = require('events'),
    fs = require('fs'),
    path = require('path'),
    nssocket = require('nssocket'),
    utile = require('utile'),
    forever = require(path.resolve(__dirname, '..', 'forever'));

var Worker = exports.Worker = function (options) {
  events.EventEmitter.call(this);
  options = options || {};

  this.monitor  = options.monitor;
  this.sockPath = options.sockPath || forever.config.get('sockPath');
  this.exitOnStop = options.exitOnStop === true;

  this._socket = null;
};

utile.inherits(Worker, events.EventEmitter);

Worker.prototype.start = function (callback) {
  var self = this,
      err;

  if (this._socket) {
    err = new Error("Can't start already started worker");
    if (callback) {
      return callback(err);
    }

    throw err;
  }

  //
  // Defines a simple `nssocket` protocol for communication
  // with a parent process.
  //
  function workerProtocol(socket) {
    socket.on('error', function() {
      socket.destroy();
    });

    socket.data(['ping'], function () {
      socket.send(['pong']);
    });

    socket.data(['data'], function () {
      socket.send(['data'], self.monitor.data);
    });

    socket.data(['spawn'], function (data) {
      if (!data.script) {
        return socket.send(['spawn', 'error'], { error: new Error('No script given') });
      }

      if (self.monitor) {
        return socket.send(['spawn', 'error'], { error: new Error("Already running") });
      }

      var monitor = new (forever.Monitor)(data.script, data.args);
      monitor.start();

      monitor.on('start', function () {
        socket.send(['spawn', 'start'], monitor.data);
      });
    });

    socket.data(['stop'], function () {
      function onStop(err) {
        var args = [];
        if (err && err instanceof Error) {
          args.push(['stop', 'error'], { message: err.message, stack: err.stack });
          self.monitor.removeListener('stop', onStop);
        }
        else {
          args.push(['stop', 'ok']);
          self.monitor.removeListener('error', onStop);
        }

        socket.send.apply(socket, args);
        if (self.exitOnStop) {
          process.exit();
        }
      }

      self.monitor.once('stop', onStop);
      self.monitor.once('error', onStop);

      if (process.platform === 'win32') {
        //
        // On Windows, delete the 'symbolic' sock file. This
        // file is used for exploration during `forever list`
        // as a mapping to the `\\.pipe\\*` "files" that can't
        // be enumerated because ... Windows.
        //
        fs.unlink(self._sockFile);
      }

      self.monitor.stop();
    });

    socket.data(['restart'], function () {
      self.monitor.once('restart', function () {
        socket.send(['restart', 'ok']);
      });

      self.monitor.restart();
    });
  }

  function findAndStart() {
    self._socket = nssocket.createServer(workerProtocol);
    self._socket.on('listening', function () {
      //
      // `listening` listener doesn't take error as the first parameter
      //
      self.emit('start');
      if (callback) {
        callback(null, self._sockFile);
      }
    });

    self._socket.on('error', function (err) {
      if (err.code === 'EADDRINUSE') {
        return findAndStart();
      }
      else if (callback) {
        callback(err);
      }
    });

    //
    // Create a unique socket file based on the current microtime.
    //
    var sock = self._sockFile = path.join(self.sockPath, [
      'worker',
      new Date().getTime() + utile.randomString(3),
      'sock'
    ].join('.'));

    if (process.platform === 'win32') {
      //
      // Create 'symbolic' file on the system, so it can be later
      // found via "forever list" since the `\\.pipe\\*` "files" can't
      // be enumerated because ... Windows.
      //
      fs.openSync(sock, 'w');

      //
      // It needs the prefix, otherwise EACCESS error happens on Windows
      // (no .sock extension, only named pipes with .pipe prefixes)
      //
      sock = '\\\\.\\pipe\\' + sock;
    }

    self._socket.listen(sock);
  }

  //
  // Attempt to start the server the first time
  //
  findAndStart();
  return this;
};

