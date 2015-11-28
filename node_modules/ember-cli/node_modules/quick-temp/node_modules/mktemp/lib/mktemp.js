var fs = require('fs'),
    randomstring = require('./randomstring');


/**
 * create unique name file.
 *
 * @param {String} template template string for filename.
 * @param {Function} callback callback function.
 */
function createFile(template, callback) {
  var filename = randomstring.generate(template);

  fs.open(filename, 'ax+', 384 /*=0600*/, function(err, fd) {
    if (err) {
      if (err.code === 'EEXIST') {
        // FIXME: infinite loop
        setImmediate(function(tmpl, cb) {
          createFile(tmpl, cb);
        }, template, callback);

        return;
      }

      // filename set to null if throws error
      filename = null;
    }

    if (fd) {
      fs.close(fd, function(err) {
        callback(err, filename);
      });
    } else {
      callback(err, filename);
    }
  });
}


/**
 * sync version createFile.
 *
 * @param {String} template template string for filename.
 * @throws {Error} error of fs.openSync or fs.closeSync.
 * @return {String} created filename.
 */
function createFileSync(template) {
  var isExist, filename, fd;

  // FIXME: infinite loop
  do {
    isExist = false;
    filename = randomstring.generate(template);
    try {
      fd = fs.openSync(filename, 'ax+', 384 /*=0600*/);
    } catch (e) {
      if (e.code === 'EEXIST') {
        isExist = true;
      } else {
        throw e;
      }
    } finally {
      fd && fs.closeSync(fd);
    }
  } while (isExist);

  return filename;
}


/**
 * create unique name dir.
 *
 * @param {String} template template string for dirname.
 * @param {Function} callback callback function.
 */
function createDir(template, callback) {
  var dirname = randomstring.generate(template);

  fs.mkdir(dirname, 448 /*=0700*/, function(err) {
    if (err) {
      if (err.code === 'EEXIST') {
        // FIXME: infinite loop
        setImmediate(function(tmpl, cb) {
          createDir(tmpl, cb);
        }, template, callback);

        return;
      }

      // dirname set to null if throws error
      dirname = null;
    }

    callback(err, dirname);
  });
}


/**
 * sync version createDir.
 *
 * @param {String} template template string for dirname.
 * @return {String} created dirname.
 */
function createDirSync(template) {
  var isExist, dirname;

  // FIXME: infinite loop
  do {
    isExist = false;
    dirname = randomstring.generate(template);
    try {
      fs.mkdirSync(dirname, 448 /*=0700*/);
    } catch (e) {
      if (e.code === 'EEXIST') {
        isExist = true;
      } else {
        throw e;
      }
    }
  } while (isExist);

  return dirname;
}


/**
 * export.
 */
module.exports = {
  createFile: createFile,
  createFileSync: createFileSync,
  createDir: createDir,
  createDirSync: createDirSync
};
