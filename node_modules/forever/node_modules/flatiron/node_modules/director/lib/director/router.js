/*
 * router.js: Base functionality for the router.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var QUERY_SEPARATOR = /\?.*/;

//
// Helper function to turn flatten an array.
//
function _flatten (arr) {
  var flat = [];

  for (var i = 0, n = arr.length; i < n; i++) {
    flat = flat.concat(arr[i]);
  }

  return flat;
}

//
// Helper function for wrapping Array.every
// in the browser.
//
function _every (arr, iterator) {
  for (var i = 0; i < arr.length; i += 1) {
    if (iterator(arr[i], i, arr) === false) {
      return;
    }
  }
}

//
// Helper function for performing an asynchronous every
// in series in the browser and the server.
//
function _asyncEverySeries (arr, iterator, callback) {
  if (!arr.length) {
    return callback();
  }

  var completed = 0;
  (function iterate() {
    iterator(arr[completed], function (err) {
      if (err || err === false) {
        callback(err);
        callback = function () {};
      }
      else {
        completed += 1;
        if (completed === arr.length) {
          callback();
        }
        else {
          iterate();
        }
      }
    });
  })();
}

//
// Helper function for expanding "named" matches
// (e.g. `:dog`, etc.) against the given set
// of params:
//
//    {
//      ':dog': function (str) {
//        return str.replace(/:dog/, 'TARGET');
//      }
//      ...
//    }
//
function paramifyString(str, params, mod) {
  mod = str;
  for (var param in params) {
    if (params.hasOwnProperty(param)) {
      mod = params[param](str);
      if (mod !== str) { break; }
    }
  }

  return mod === str
    ? '([._a-zA-Z0-9-%()]+)'
    : mod;
}

//
// Helper function for expanding wildcards (*) and
// "named" matches (:whatever)
//
function regifyString(str, params) {
  var matches,
      last = 0,
      out = '';

  while (matches = str.substr(last).match(/[^\w\d\- %@&]*\*[^\w\d\- %@&]*/)) {
    last = matches.index + matches[0].length;
    matches[0] = matches[0].replace(/^\*/, '([_\.\(\)!\\ %@&a-zA-Z0-9-]+)');
    out += str.substr(0, matches.index) + matches[0];
  }

  str = out += str.substr(last);

   var captures = str.match(/:([^\/]+)/ig),
       capture,
       length;

   if (captures) {
     length = captures.length;
     for (var i = 0; i < length; i++) {
       capture = captures[i];
       if ( capture.slice(0, 2) === "::" ) {
           // This parameter was escaped and should be left in the url as a literal
           // Remove the escaping : from the beginning
           str = capture.slice( 1 );
       } else {
           str = str.replace(capture, paramifyString(capture, params));
  }
     }
   }

  return str;
}

//
// ### Fix unterminated RegExp groups in routes.
//
function terminator(routes, delimiter, start, stop) {
  var last = 0,
      left = 0,
      right = 0,
      start = (start || '(').toString(),
      stop = (stop || ')').toString(),
      i;

  for (i = 0; i < routes.length; i++) {
    var chunk = routes[i];

    if ((chunk.indexOf(start, last) > chunk.indexOf(stop, last)) ||
        (~chunk.indexOf(start, last) && !~chunk.indexOf(stop, last)) ||
        (!~chunk.indexOf(start, last) && ~chunk.indexOf(stop, last))) {

      left = chunk.indexOf(start, last);
      right = chunk.indexOf(stop, last);

      if ((~left && !~right) || (!~left && ~right)) {
        var tmp = routes.slice(0, (i || 1) + 1).join(delimiter);
        routes = [tmp].concat(routes.slice((i || 1) + 1));
      }

      last = (right > left ? right : left) + 1;
      i = 0;
    }
    else {
      last = 0;
    }
  }

  return routes;
}



//
// ### function Router (routes)
// #### @routes {Object} **Optional** Routing table for this instance.
// Constuctor function for the Router object responsible for building
// and dispatching from a given routing table.
//
var Router = exports.Router = function (routes) {
  this.params   = {};
  this.routes   = {};
  this.methods  = ['on', 'after', 'before'];
  this.scope    = [];
  this._methods = {};

  this.configure();
  this.mount(routes || {});
};

//
// ### function configure (options)
// #### @options {Object} **Optional** Options to configure this instance with
// Configures this instance with the specified `options`.
//
Router.prototype.configure = function (options) {
  options = options || {};

  for (var i = 0; i < this.methods.length; i++) {
    this._methods[this.methods[i]] = true;
  }

  this.recurse   = options.recurse   || this.recurse || false;
  this.async     = options.async     || false;
  this.delimiter = options.delimiter || '\/';
  this.strict    = typeof options.strict === 'undefined' ? true : options.strict;
  this.notfound  = options.notfound;
  this.resource  = options.resource;

  // Client only, but browser.js does not include a super implementation
  this.history     = (options.html5history && this.historySupport) || false;
  this.run_in_init = (this.history === true && options.run_handler_in_init !== false);
  this.convert_hash_in_init = (this.history === true && options.convert_hash_in_init !== false);

  //
  // TODO: Global once
  //
  this.every = {
    after: options.after || null,
    before: options.before || null,
    on: options.on || null
  };

  return this;
};

//
// ### function param (token, regex)
// #### @token {string} Token which to replace (e.g. `:dog`, 'cat')
// #### @matcher {string|RegExp} Target to replace the token with.
// Setups up a `params` function which replaces any instance of `token`,
// inside of a given `str` with `matcher`. This is very useful if you
// have a common regular expression throughout your code base which
// you wish to be more DRY.
//
Router.prototype.param = function (token, matcher) {
  if (token[0] !== ':') {
    token = ':' + token;
  }

  var compiled = new RegExp(token, 'g');
  this.params[token] = function (str) {
    return str.replace(compiled, matcher.source || matcher);
  };
  return this;
};

//
// ### function on (method, path, route)
// #### @method {string} **Optional** Method to use
// #### @path {Array|string} Path to set this route on.
// #### @route {Array|function} Handler for the specified method and path.
// Adds a new `route` to this instance for the specified `method`
// and `path`.
//
Router.prototype.on = Router.prototype.route = function (method, path, route) {
  var self = this;

  if (!route && typeof path == 'function') {
    //
    // If only two arguments are supplied then assume this
    // `route` was meant to be a generic `on`.
    //
    route = path;
    path = method;
    method = 'on';
  }

  if (Array.isArray(path)) {
    return path.forEach(function(p) {
      self.on(method, p, route);
    });
  }

  if (path.source) {
    path = path.source.replace(/\\\//ig, '/');
  }

  if (Array.isArray(method)) {
    return method.forEach(function (m) {
      self.on(m.toLowerCase(), path, route);
    });
  }

  //
  // ### Split the route up by the delimiter.
  //
  path = path.split(new RegExp(this.delimiter));

  //
  // ### Fix unterminated groups. Fixes #59
  //
  path = terminator(path, this.delimiter);

  this.insert(method, this.scope.concat(path), route);
};

//
// ### function path (path, routesFn)
// #### @path {string|RegExp} Nested scope in which to path
// #### @routesFn {function} Function to evaluate in the new scope
// Evalutes the `routesFn` in the given path scope.
//
Router.prototype.path = function (path, routesFn) {
  var self = this,
      length = this.scope.length;

  if (path.source) {
    path = path.source.replace(/\\\//ig, '/');
  }

  //
  // ### Split the route up by the delimiter.
  //
  path = path.split(new RegExp(this.delimiter));

  //
  // ### Fix unterminated groups.
  //
  path = terminator(path, this.delimiter);
  this.scope = this.scope.concat(path);

  routesFn.call(this, this);
  this.scope.splice(length, path.length);
};

//
// ### function dispatch (method, path[, callback])
// #### @method {string} Method to dispatch
// #### @path {string} Path to dispatch
// #### @callback {function} **Optional** Continuation to respond to for async scenarios.
// Finds a set of functions on the traversal towards
// `method` and `path` in the core routing table then
// invokes them based on settings in this instance.
//
Router.prototype.dispatch = function (method, path, callback) {
  var self = this,
      fns = this.traverse(method, path.replace(QUERY_SEPARATOR, ''), this.routes, ''),
      invoked = this._invoked,
      after;

  this._invoked = true;
  if (!fns || fns.length === 0) {
    this.last = [];
    if (typeof this.notfound === 'function') {
      this.invoke([this.notfound], { method: method, path: path }, callback);
    }

    return false;
  }

  if (this.recurse === 'forward') {
    fns = fns.reverse();
  }

  function updateAndInvoke() {
    self.last = fns.after;
    self.invoke(self.runlist(fns), self, callback);
  }

  //
  // Builds the list of functions to invoke from this call
  // to dispatch conforming to the following order:
  //
  // 1. Global after (if any)
  // 2. After functions from the last call to dispatch
  // 3. Global before (if any)
  // 4. Global on (if any)
  // 5. Matched functions from routing table (`['before', 'on'], ['before', 'on`], ...]`)
  //
  after = this.every && this.every.after
    ? [this.every.after].concat(this.last)
    : [this.last];

  if (after && after.length > 0 && invoked) {
    if (this.async) {
      this.invoke(after, this, updateAndInvoke);
    }
    else {
      this.invoke(after, this);
      updateAndInvoke();
    }

    return true;
  }

  updateAndInvoke();
  return true;
};

//
// ### function runlist (fns)
// #### @fns {Array} List of functions to include in the runlist
// Builds the list of functions to invoke from this call
// to dispatch conforming to the following order:
//
// 1. Global before (if any)
// 2. Global on (if any)
// 3. Matched functions from routing table (`['before', 'on'], ['before', 'on`], ...]`)
//
Router.prototype.runlist = function (fns) {
  var runlist = this.every && this.every.before
    ? [this.every.before].concat(_flatten(fns))
    : _flatten(fns);

  if (this.every && this.every.on) {
    runlist.push(this.every.on);
  }

  runlist.captures = fns.captures;
  runlist.source = fns.source;
  return runlist;
};

//
// ### function invoke (fns, thisArg)
// #### @fns {Array} Set of functions to invoke in order.
// #### @thisArg {Object} `thisArg` for each function.
// #### @callback {function} **Optional** Continuation to pass control to for async `fns`.
// Invokes the `fns` synchronously or asynchronously depending on the
// value of `this.async`. Each function must **not** return (or respond)
// with false, or evaluation will short circuit.
//
Router.prototype.invoke = function (fns, thisArg, callback) {
  var self = this;

  var apply;
  if (this.async) {
    apply = function(fn, next){
      if (Array.isArray(fn)) {
        return _asyncEverySeries(fn, apply, next);
      }
      else if (typeof fn == 'function') {
        fn.apply(thisArg, (fns.captures || []).concat(next));
      }
    };
    _asyncEverySeries(fns, apply, function () {
      //
      // Ignore the response here. Let the routed take care
      // of themselves and eagerly return true.
      //

      if (callback) {
        callback.apply(thisArg, arguments);
      }
    });
  }
  else {
    apply = function(fn){
      if (Array.isArray(fn)) {
        return _every(fn, apply);
      }
      else if (typeof fn === 'function') {
        return fn.apply(thisArg, fns.captures || []);
      }
      else if (typeof fn === 'string' && self.resource) {
        self.resource[fn].apply(thisArg, fns.captures || []);
      }
    }
    _every(fns, apply);
  }
};

//
// ### function traverse (method, path, routes, regexp)
// #### @method {string} Method to find in the `routes` table.
// #### @path {string} Path to find in the `routes` table.
// #### @routes {Object} Partial routing table to match against
// #### @regexp {string} Partial regexp representing the path to `routes`.
// #### @filter {function} Filter function for filtering routes (expensive).
// Core routing logic for `director.Router`: traverses the
// specified `path` within `this.routes` looking for `method`
// returning any `fns` that are found.
//
Router.prototype.traverse = function (method, path, routes, regexp, filter) {
  var fns = [],
      current,
      exact,
      match,
      next,
      that;

  function filterRoutes(routes) {
    if (!filter) {
      return routes;
    }

    function deepCopy(source) {
      var result = [];
      for (var i = 0; i < source.length; i++) {
        result[i] = Array.isArray(source[i]) ? deepCopy(source[i]) : source[i];
      }
      return result;
    }

    function applyFilter(fns) {
      for (var i = fns.length - 1; i >= 0; i--) {
        if (Array.isArray(fns[i])) {
          applyFilter(fns[i]);
          if (fns[i].length === 0) {
            fns.splice(i, 1);
          }
        }
        else {
          if (!filter(fns[i])) {
            fns.splice(i, 1);
          }
        }
      }
    }

    var newRoutes = deepCopy(routes);
    newRoutes.matched = routes.matched;
    newRoutes.captures = routes.captures;
    newRoutes.after = routes.after.filter(filter);

    applyFilter(newRoutes);

    return newRoutes;
  }

  //
  // Base Case #1:
  // If we are dispatching from the root
  // then only check if the method exists.
  //
  if (path === this.delimiter && routes[method]) {
    next = [[routes.before, routes[method]].filter(Boolean)];
    next.after = [routes.after].filter(Boolean);
    next.matched = true;
    next.captures = [];
    return filterRoutes(next);
  }

  for (var r in routes) {
    //
    // We dont have an exact match, lets explore the tree
    // in a depth-first, recursive, in-order manner where
    // order is defined as:
    //
    //    ['before', 'on', '<method>', 'after']
    //
    // Remember to ignore keys (i.e. values of `r`) which
    // are actual methods (e.g. `on`, `before`, etc), but
    // which are not actual nested route (i.e. JSON literals).
    //
    if (routes.hasOwnProperty(r) && (!this._methods[r] ||
      this._methods[r] && typeof routes[r] === 'object' && !Array.isArray(routes[r]))) {
      //
      // Attempt to make an exact match for the current route
      // which is built from the `regexp` that has been built
      // through recursive iteration.
      //
      current = exact = regexp + this.delimiter + r;

      if (!this.strict) {
        exact += '[' + this.delimiter + ']?';
      }

      match = path.match(new RegExp('^' + exact));

      if (!match) {
        //
        // If there isn't a `match` then continue. Here, the
        // `match` is a partial match. e.g.
        //
        //    '/foo/bar/buzz'.match(/^\/foo/)   // ['/foo']
        //    '/no-match/route'.match(/^\/foo/) // null
        //
        continue;
      }

      if (match[0] && match[0] == path && routes[r][method]) {
        //
        // ### Base case 2:
        // If we had a `match` and the capture is the path itself,
        // then we have completed our recursion.
        //
        next = [[routes[r].before, routes[r][method]].filter(Boolean)];
        next.after = [routes[r].after].filter(Boolean);
        next.matched = true;
        next.captures = match.slice(1);

        if (this.recurse && routes === this.routes) {
          next.push([routes.before, routes.on].filter(Boolean));
          next.after = next.after.concat([routes.after].filter(Boolean));
        }

        return filterRoutes(next);
      }

      //
      // ### Recursive case:
      // If we had a match, but it is not yet an exact match then
      // attempt to continue matching against the next portion of the
      // routing table.
      //
      next = this.traverse(method, path, routes[r], current);

      //
      // `next.matched` will be true if the depth-first search of the routing
      // table from this position was successful.
      //
      if (next.matched) {
        //
        // Build the in-place tree structure representing the function
        // in the correct order.
        //
        if (next.length > 0) {
          fns = fns.concat(next);
        }

        if (this.recurse) {
          fns.push([routes[r].before, routes[r].on].filter(Boolean));
          next.after = next.after.concat([routes[r].after].filter(Boolean));

          if (routes === this.routes) {
            fns.push([routes['before'], routes['on']].filter(Boolean));
            next.after = next.after.concat([routes['after']].filter(Boolean));
          }
        }

        fns.matched = true;
        fns.captures = next.captures;
        fns.after = next.after;

        //
        // ### Base case 2:
        // Continue passing the partial tree structure back up the stack.
        // The caller for `dispatch()` will decide what to do with the functions.
        //
        return filterRoutes(fns);
      }
    }
  }

  return false;
};

//
// ### function insert (method, path, route, context)
// #### @method {string} Method to insert the specific `route`.
// #### @path {Array} Parsed path to insert the `route` at.
// #### @route {Array|function} Route handlers to insert.
// #### @parent {Object} **Optional** Parent "routes" to insert into.
// Inserts the `route` for the `method` into the routing table for
// this instance at the specified `path` within the `context` provided.
// If no context is provided then `this.routes` will be used.
//
Router.prototype.insert = function (method, path, route, parent) {
  var methodType,
      parentType,
      isArray,
      nested,
      part;

  path = path.filter(function (p) {
    return p && p.length > 0;
  });

  parent = parent || this.routes;
  part = path.shift();
  if (/\:|\*/.test(part) && !/\\d|\\w/.test(part)) {
    part = regifyString(part, this.params);
  }

  if (path.length > 0) {
    //
    // If this is not the last part left in the `path`
    // (e.g. `['cities', 'new-york']`) then recurse into that
    // child
    //
    parent[part] = parent[part] || {};
    return this.insert(method, path, route, parent[part]);
  }

  //
  // If there is no part and the path has been exhausted
  // and the parent is the root of the routing table,
  // then we are inserting into the root and should
  // only dive one level deep in the Routing Table.
  //
  if (!part && !path.length && parent === this.routes) {
    methodType = typeof parent[method];

    switch (methodType) {
      case 'function':
        parent[method] = [parent[method], route];
        return;
      case 'object':
        parent[method].push(route);
        return;
      case 'undefined':
        parent[method] = route;
        return;
    }

    return;
  }

  //
  // Otherwise, we are at the end of our insertion so we should
  // insert the `route` based on the `method` after getting the
  // `parent` of the last `part`.
  //
  parentType = typeof parent[part];
  isArray = Array.isArray(parent[part]);

  if (parent[part] && !isArray && parentType == 'object') {
    methodType = typeof parent[part][method];

    switch (methodType) {
      case 'function':
        parent[part][method] = [parent[part][method], route];
        return;
      case 'object':
        parent[part][method].push(route);
        return;
      case 'undefined':
        parent[part][method] = route;
        return;
    }
  }
  else if (parentType == 'undefined') {
    nested = {};
    nested[method] = route;
    parent[part] = nested;
    return;
  }

  throw new Error('Invalid route context: ' + parentType);
};


//
// ### function extend (methods)
// #### @methods {Array} List of method names to extend this instance with
// Extends this instance with simple helper methods to `this.on`
// for each of the specified `methods`
//
Router.prototype.extend = function(methods) {
  var self = this,
      len = methods.length,
      i;

  function extend(method) {
    self._methods[method] = true;
    self[method] = function () {
      var extra = arguments.length === 1
        ? [method, '']
        : [method];

      self.on.apply(self, extra.concat(Array.prototype.slice.call(arguments)));
    };
  }

  for (i = 0; i < len; i++) {
    extend(methods[i]);
  }
};

//
// ### function mount (routes, context)
// #### @routes {Object} Routes to mount onto this instance
// Mounts the sanitized `routes` onto the root context for this instance.
//
// e.g.
//
//    new Router().mount({ '/foo': { '/bar': function foobar() {} } })
//
// yields
//
//    { 'foo': 'bar': function foobar() {} } }
//
Router.prototype.mount = function(routes, path) {
  if (!routes || typeof routes !== "object" || Array.isArray(routes)) {
    return;
  }

  var self = this;
  path = path || [];
  if (!Array.isArray(path)) {
    path = path.split(self.delimiter);
  }

  function insertOrMount(route, local) {
    var rename = route,
        parts = route.split(self.delimiter),
        routeType = typeof routes[route],
        isRoute = parts[0] === "" || !self._methods[parts[0]],
        event = isRoute ? "on" : rename;

    if (isRoute) {
      rename = rename.slice((rename.match(new RegExp('^' + self.delimiter)) || [''])[0].length);
      parts.shift();
    }

    if (isRoute && routeType === 'object' && !Array.isArray(routes[route])) {
      local = local.concat(parts);
      self.mount(routes[route], local);
      return;
    }

    if (isRoute) {
      local = local.concat(rename.split(self.delimiter));
      local = terminator(local, self.delimiter);
    }

    self.insert(event, local, routes[route]);
  }

  for (var route in routes) {
    if (routes.hasOwnProperty(route)) {
      insertOrMount(route, path.slice(0));
    }
  }
};

