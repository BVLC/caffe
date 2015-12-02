/*
 * file.js: Simple file storage engine for nconf files
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var fs = require('fs'),
    path = require('path'),
    util = require('util'),
    formats = require('../formats'),
    Memory = require('./memory').Memory,
    exists = fs.exists || path.exists,
    existsSync = fs.existsSync || path.existsSync;

//
// ### function File (options)
// #### @options {Object} Options for this instance
// Constructor function for the File nconf store, a simple abstraction
// around the Memory store that can persist configuration to disk.
//
var File = exports.File = function (options) {
  if (!options || !options.file) {
    throw new Error ('Missing required option `file`');
  }

  Memory.call(this, options);

  this.type   = 'file';
  this.file   = options.file;
  this.dir    = options.dir    || process.cwd();
  this.format = options.format || formats.json;
  this.json_spacing = options.json_spacing || 2;

  if (options.search) {
    this.search(this.dir);
  }
};

// Inherit from the Memory store
util.inherits(File, Memory);

//
// ### function save (value, callback)
// #### @value {Object} _Ignored_ Left here for consistency
// #### @callback {function} Continuation to respond to when complete.
// Saves the current configuration object to disk at `this.file`
// using the format specified by `this.format`.
//
File.prototype.save = function (value, callback) {
  if (!callback) {
    callback = value;
    value = null;
  }

  fs.writeFile(this.file, this.format.stringify(this.store, null, this.json_spacing), function (err) {
    return err ? callback(err) : callback();
  });
};

//
// ### function saveSync (value, callback)
// #### @value {Object} _Ignored_ Left here for consistency
// #### @callback {function} **Optional** Continuation to respond to when complete.
// Saves the current configuration object to disk at `this.file`
// using the format specified by `this.format` synchronously.
//
File.prototype.saveSync = function (value) {
  try {
    fs.writeFileSync(this.file, this.format.stringify(this.store, null, this.json_spacing));
  }
  catch (ex) {
    throw(ex);
  }
  return this.store;
};

//
// ### function load (callback)
// #### @callback {function} Continuation to respond to when complete.
// Responds with an Object representing all keys associated in this instance.
//
File.prototype.load = function (callback) {
  var self = this;

  exists(self.file, function (exists) {
    if (!exists) {
      return callback(null, {});
    }

    //
    // Else, the path exists, read it from disk
    //
    fs.readFile(self.file, function (err, data) {
      if (err) {
        return callback(err);
      }

      try {
        //deals with string that include BOM
        var stringData = data.toString();

        if (stringData.charAt(0) === '\uFEFF') stringData = stringData.substr(1);
        self.store = self.format.parse(stringData);

      }
      catch (ex) {
        return callback(new Error("Error parsing your JSON configuration file: [" + self.file + '].'));
      }

      callback(null, self.store);
    });
  });
};

//
// ### function loadSync (callback)
// Attempts to load the data stored in `this.file` synchronously
// and responds appropriately.
//
File.prototype.loadSync = function () {
  var data, self = this;

  if (!existsSync(self.file)) {
    self.store = {};
    data = {};
  }
  else {
    //
    // Else, the path exists, read it from disk
    //
    try {
      //deals with file that include BOM
      var fileData = fs.readFileSync(this.file, 'utf8');
      if (fileData.charAt(0) === '\uFEFF') fileData = fileData.substr(1);

      data = this.format.parse(fileData);
      this.store = data;
    }
    catch (ex) {
      throw new Error("Error parsing your JSON configuration file: [" + self.file + '].');
    }
  }

  return data;
};

//
// ### function search (base)
// #### @base {string} Base directory (or file) to begin searching for the target file.
// Attempts to find `this.file` by iteratively searching up the
// directory structure
//
File.prototype.search = function (base) {
  var looking = true,
      fullpath,
      previous,
      stats;

  base = base || process.cwd();

  if (this.file[0] === '/') {
    //
    // If filename for this instance is a fully qualified path
    // (i.e. it starts with a `'/'`) then check if it exists
    //
    try {
      stats = fs.statSync(fs.realpathSync(this.file));
      if (stats.isFile()) {
        fullpath = this.file;
        looking = false;
      }
    }
    catch (ex) {
      //
      // Ignore errors
      //
    }
  }

  if (looking && base) {
    //
    // Attempt to stat the realpath located at `base`
    // if the directory does not exist then return false.
    //
    try {
      var stat = fs.statSync(fs.realpathSync(base));
      looking = stat.isDirectory();
    }
    catch (ex) {
      return false;
    }
  }

  while (looking) {
    //
    // Iteratively look up the directory structure from `base`
    //
    try {
      stats = fs.statSync(fs.realpathSync(fullpath = path.join(base, this.file)));
      looking = stats.isDirectory();
    }
    catch (ex) {
      previous = base;
      base = path.dirname(base);

      if (previous === base) {
        //
        // If we've reached the top of the directory structure then simply use
        // the default file path.
        //
        try {
          stats = fs.statSync(fs.realpathSync(fullpath = path.join(this.dir, this.file)));
          if (stats.isDirectory()) {
            fullpath = undefined;
          }
        }
        catch (ex) {
          //
          // Ignore errors
          //
        }

        looking = false;
      }
    }
  }

  //
  // Set the file for this instance to the fullpath
  // that we have found during the search. In the event that
  // the search was unsuccessful use the original value for `this.file`.
  //
  this.file = fullpath || this.file;

  return fullpath;
};
