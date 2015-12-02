(function () {
  "use strict";

  var g_http = require('http')
    ;

  function create(config) {
    config = config || {};

    function handler(reqProto, resProto, http) {
      if (!resProto) {
        resProto = g_http.ServerResponse.prototype;
      }

      resProto.error = function (msg, code, opts) {
        this.errors = this.errors || [];
        if ('object' !== typeof opts) {
          opts = {};
        }

        opts.message = msg;
        opts.code = code;
        this.errors.push(opts);
      };

      // TODO maybe jsonp
      console.warn('[ jason WARNING ] backwards INCOMPATIBLE change in API')
      console.warn('[ jason WARNING ] your CODE is BROKEN')
      console.log(' ');
      console.warn('[ jason WARNING ] res.json(obj) will now output `{ result: obj, error: false, ... }` rather than mixing in the error and session stuff');
      resProto.json = function (data, opts) {
        var json
          , response = {}
          , space
          ;

        opts = opts || {};
        space = (config.debug || opts.debug) ? '  ': null
        response.timestamp = Date.now();
        response.errors = this.errors || data.errors || (data.error ? [data.error] : []);
        response.error = response.error || (response.errors.length ? true : false);
        response.result = data;

        this.statusCode = this.statusCode || opts.statusCode || (response.error ? 400 : 200);

        try {
          json = JSON.stringify(response, null, space);
        } catch(e) {
          this.statusCode = 500;
          json = JSON.stringify({ error: true, errors: [e] }, null, space);
        }

        this.charset = this.charset || 'utf-8';
        this.setHeader('Content-Type', 'application/json');
        this.end(json);
      };

    }
    return handler;
  }

  module.exports = create;
}());
