/*
 * inspect.js: Plugin responsible for attaching inspection behavior using `cliff` and `eyes`.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

//
// ### Name this plugin
//
exports.name = 'inspect';

//
// ### function init (done)
// #### @done {function} Continuation to respond to when complete.
// Attaches inspection behavior through `cliff` and `eyes`.
//
exports.init = function (done) {
  var namespace = 'default',
      app = this;

  if (app.options['inspect'] && app.options['inspect'].namespace) {
    namespace = app.options['inspect'].namespace;
  }

  app.inspect = require('cliff');
  app.inspect.logger = app.log.get('namespace');
  done();
};

//
// ### function detact()
// Removes inspection behavior exposed by this plugin.
//
exports.detach = function () {
  if (this.inspect) {
    delete this.inspect;
  }
};