var fs   = require('fs'),
  Path   = require('path'),
  util   = require('util'),
  colors = require('colors'),
  EE     = require('events').EventEmitter,
  fsExists = fs.exists ? fs.exists : Path.exists,
  fsExistsSync = fs.existsSync ? fs.existsSync : Path.existsSync;

module.exports = function(dir, iterator, options, callback){
  return FindUp(dir, iterator, options, callback);
};

function FindUp(dir, iterator, options, callback){
  if (!(this instanceof FindUp)) {
    return new FindUp(dir, iterator, options, callback);
  }
  if(typeof options === 'function'){
    callback = options;
    options = {};
  }
  options = options || {};

  EE.call(this);
  this.found = false;
  this.stopPlease = false;
  var self = this;

  if(typeof iterator === 'string'){
    var file = iterator;
    iterator = function(dir, cb){
      return fsExists(Path.join(dir, file), cb);
    };
  }

  if(callback) {
    this.on('found', function(dir){
      if(options.verbose) console.log(('found '+ dir ).green);
      callback(null, dir);
      self.stop();
    });

    this.on('end', function(){
      if(options.verbose) console.log('end'.grey);
      if(!self.found) callback(new Error('not found'));
    });

    this.on('error', function(err){
      if(options.verbose) console.log('error'.red, err);
      callback(err);
    });
  }

  this._find(dir, iterator, options, callback);
}
util.inherits(FindUp, EE);

FindUp.prototype._find = function(dir, iterator, options, callback){
  var self = this;

  iterator(dir, function(exists){
    if(options.verbose) console.log(('traverse '+ dir).grey);
    if(exists) {
      self.found = true;
      self.emit('found', dir);
    }

    var parentDir = Path.join(dir, '..');
    if (self.stopPlease) return self.emit('end');
    if (dir === parentDir) return self.emit('end');
    if(dir.indexOf('../../') !== -1 ) return self.emit('error', new Error(dir + ' is not correct.'));
    self._find(parentDir, iterator, options, callback);
  });
};

FindUp.prototype.stop = function(){
  this.stopPlease = true;
};

module.exports.FindUp = FindUp;

module.exports.sync = function(dir, iteratorSync){
  if(typeof iteratorSync === 'string'){
    var file = iteratorSync;
    iteratorSync = function(dir){
      return fsExistsSync(Path.join(dir, file));
    };
  }
  var initialDir = dir;
  while(dir !== Path.join(dir, '..')){
    if(dir.indexOf('../../') !== -1 ) throw new Error(initialDir + ' is not correct.');
    if(iteratorSync(dir)) return dir;
    dir = Path.join(dir, '..');
  }
  throw new Error('not found');
};
