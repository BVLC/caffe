var events = require('events'),
    qs = require('querystring'),
    util = require('util'),
    BaseRouter = require('../router').Router,
    responses = require('./responses');

//
// ### Expose all HTTP methods and responses
//
exports.methods   = require('./methods');
Object.keys(responses).forEach(function (name) {
  exports[name] = responses[name];
});

//
// ### function Router (routes)
// #### @routes {Object} **Optional** Routing table for this instance.
// Constuctor function for the HTTP Router object responsible for building
// and dispatching from a given routing table.
//
var Router = exports.Router = function (routes) {
  //
  // ### Extend the `Router` prototype with all of the RFC methods.
  //
  this.params   = {};
  this.routes   = {};
  this.methods  = ['on', 'after', 'before'];
  this.scope    = [];
  this._methods = {};
  this.recurse = 'forward';
  this._attach = [];

  this.extend(exports.methods.concat(['before', 'after']));
  this.configure();
  this.mount(routes || {});
};

//
// Inherit from `BaseRouter`.
//
util.inherits(Router, BaseRouter);

//
// ### function configure (options)
// #### @options {Object} **Optional** Options to configure this instance with
// Configures this instance with the specified `options`.
//
Router.prototype.configure = function (options) {
  options = options || {};

  // useful when using connect's bodyParser
  this.stream = options.stream || false;

  return BaseRouter.prototype.configure.call(this, options);
};

//
// ### function on (method, path, route)
// #### @method {string} **Optional** Method to use
// #### @path {string} Path to set this route on.
// #### @route {Array|function} Handler for the specified method and path.
// Adds a new `route` to this instance for the specified `method`
// and `path`.
//
Router.prototype.on = function (method, path) {
  var args = Array.prototype.slice.call(arguments, 2),
      route = args.pop(),
      options = args.pop(),
      accept;

  if (options) {
    if (options.stream) {
      route.stream = true;
    }

    if (options.accept) {
      this._hasAccepts = true;
      accept = options.accept;
      route.accept = (Array.isArray(accept) ? accept : [accept]).map(function (a) {
        return typeof a === 'string' ? RegExp(a) : a;
      });
    }
  }

  if (typeof path !== 'string' && !path.source) {
    path = '';
  }

  BaseRouter.prototype.on.call(this, method, path, route);
};

//
// ### function attach (func)
// ### @func {function} Function to execute on `this` before applying to router function
// Ask the router to attach objects or manipulate `this` object on which the
// function passed to the http router will get applied
Router.prototype.attach = function (func) {
  this._attach.push(func);
};

//
// ### function dispatch (method, path)
// #### @req {http.ServerRequest} Incoming request to dispatch.
// #### @res {http.ServerResponse} Outgoing response to dispatch.
// #### @callback {function} **Optional** Continuation to respond to for async scenarios.
// Finds a set of functions on the traversal towards
// `method` and `path` in the core routing table then
// invokes them based on settings in this instance.
//
Router.prototype.dispatch = function (req, res, callback) {
  //
  // Dispatch `HEAD` requests to `GET`
  //
  var method = req.method === 'HEAD' ? 'get' : req.method.toLowerCase(),
      thisArg = { req: req, res: res },
      self = this,
      contentType,
      runlist,
      stream,
      error,
      fns,
      url;

  //
  // Trap bad URLs from `decodeUri`
  //
  try { url = decodeURI(req.url.split('?', 1)[0]); }
  catch (ex) { url = null }

  if (url && this._hasAccepts) {
    contentType = req.headers['content-type'];
    fns = this.traverse(method, url, this.routes, '', function (route) {
      return !route.accept || route.accept.some(function (a) {
        return a.test(contentType);
      });
    });
  }
  else if (url) {
    fns = this.traverse(method, url, this.routes, '');
  }

  if (this._attach) {
    for (var i in this._attach) {
      this._attach[i].call(thisArg);
    }
  }

  if (!fns || fns.length === 0) {
    error = new exports.NotFound('Could not find path: ' + req.url);
    if (typeof this.notfound === 'function') {
      this.notfound.call(thisArg, callback);
    }
    else if (callback) {
      callback.call(thisArg, error, req, res);
    }
    return false;
  }

  if (this.recurse === 'forward') {
    fns = fns.reverse();
  }

  runlist = this.runlist(fns);
  stream  = this.stream || runlist.some(function (fn) { return fn.stream === true; });

  function parseAndInvoke() {
    error = self.parse(req);
    if (error) {
      if (callback) {
        callback.call(thisArg, error, req, res);
      }
      return false;
    }

    self.invoke(runlist, thisArg, callback);
  }

  if (!stream) {
    //
    // If there is no streaming required on any of the functions on the
    // way to `path`, then attempt to parse the fully buffered request stream
    // once it has emitted the `end` event.
    //
    if (req.readable) {
      //
      // If the `http.ServerRequest` is still readable, then await
      // the end event and then continue
      //
      req.once('end', parseAndInvoke);
      // Streams2 requires us to start the stream if we're not explicitly
      // reading from it.
      req.resume();
    }
    else {
      //
      // Otherwise, just parse the body now.
      //
      parseAndInvoke();
    }
  }
  else {
    this.invoke(runlist, thisArg, callback);
  }

  return true;
};

//
// ### @parsers {Object}
// Lookup table of parsers to use when attempting to
// parse incoming responses.
//
Router.prototype.parsers = {
  'application/x-www-form-urlencoded': qs.parse,
  'application/json': JSON.parse
};

//
// ### function parse (req)
// #### @req {http.ServerResponse|BufferedStream} Incoming HTTP request to parse
// Attempts to parse `req.body` using the value found at `req.headers['content-type']`.
//
Router.prototype.parse = function (req) {
  function mime(req) {
    var str = req.headers['content-type'] || '';
    return str.split(';')[0];
  }

  var parser = this.parsers[mime(req)],
      body;

  if (parser) {
    req.body = req.body || '';

    if (req.chunks) {
      req.chunks.forEach(function (chunk) {
        req.body += chunk;
      });
    }

    if ('string' === typeof req.body) {
      try {
        req.body = req.body && req.body.length
          ? parser(req.body)
          : {};
      }
      catch (err) {
        return new exports.BadRequest('Malformed data');
      }
    }
  }
};

