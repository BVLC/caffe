/*
 * index.js: Top-level include for the features module.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

exports.ensure = function (app, callback) {
  return callback();
}

exports.all = [
  {
    name: 'Entry Point',
    test: function (target, name) {
      return typeof target.start === 'function' ||
        typeof target.createServer === 'function';
    },
    allExports: ['start', 'createServer', 'init', 'getRoutes']
  },
  {
    name: 'Resource',
    test: function (target, name) {
      var methods = ['create', 'get', 'update', 'destroy'],
          resource = target[capitalize(name)];

      if (typeof resource !== 'function') {
        return false;
      }

      for (var i = 0; i < methods.length; i++) {
        if (typeof resource[method] !== 'function') {
          return false;
        }
      }
    },
    allExports: ['addRoutes', 'init']
  },
  {
    name: 'Configurator',
    exports: ['config'],
  },
  {
    name: 'Serve Files',
    exports: 'serve'
  }
];