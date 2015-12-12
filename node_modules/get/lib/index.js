// node.js libraries
var http = require('http'),
    https = require('https'),
    util = require('util'),
    fs = require('fs'),
    events = require('events'),
    Buffer = require('buffer').Buffer,
    url = require('url'),
    path = require('path');

// Local node-get libraries
var encodings = require('./encodings');

var default_headers = {
    'Accept-Encoding': 'none',
    'Connection': 'close',
    'User-Agent': 'curl'
};

// Get a Get object. Takes an argument, that is either
// a plain string representing the URI, or an object:
//
//     {
//       uri: "string of uri", // required
//       headers: {} // optional, default in source
//       max_redirs: 5 // optional, default 10
//       no_proxy: true // prevent automatic proxy usage when HTTP_PROXY is set.
//     }
function Get(options) {
    // Allow calling without new keyword.
    if (!(this instanceof Get)) {
        return new Get(options);
    }

    if (typeof options == 'string') {
        this.uri = options;
        this.headers = default_headers;
        if (process.env.HTTP_PROXY) {
            this.proxy = url.parse(process.env.HTTP_PROXY);
            this.headers.Host = url.parse(this.uri).host;
            if (this.proxy.auth) {
                this.headers['proxy-authorization'] =
                    'Basic ' + new Buffer(this.proxy.auth).toString('base64');
            }
        } else {
            this.proxy = {};
        }
    } else {
        if (!options.uri) {
            throw Error('uri option required in get constructor');
        }
        this.uri = options.uri;
        this.max_redirs = options.max_redirs || 10;
        this.max_length = options.max_length || 0;
        this.encoding = options.encoding;
        this.headers = options.headers || default_headers;
        this.timeout = 'timeout' in options ? options.timeout : 10000;
        if (!this.no_proxy && process.env.HTTP_PROXY) {
            this.proxy = url.parse(process.env.HTTP_PROXY);
            this.headers.Host = url.parse(this.uri).host;
            if (this.proxy.auth) {
                this.headers['proxy-authorization'] =
                    'Basic ' + new Buffer(this.proxy.auth).toString('base64');
            }
        } else {
            this.proxy = {};
        }
        this.agent = options.agent || undefined;
    }
}

util.inherits(Get, events.EventEmitter);

// Create a HTTP request. Just sanity-checks URLs and
// chooses an HTTP or HTTPS request.
//
// - @return {http.ClientRequest}
Get.prototype.request = function(callback) {
    // TODO: handle non http/https protocols
    this.uri_o = url.parse(this.uri);

    // Validate the URI at this step so that invalid
    // redirects are also caught.
    if (!(this.uri_o.protocol &&
        (this.uri_o.protocol == 'http:' || this.uri_o.protocol == 'https:') &&
        this.uri_o.hostname)) {
        return callback.call(this, null, new Error('Invalid URL: ' + url.format(this.uri)));
    }

    // TODO: should pronode-getxies support HTTPS?
    if (this.uri_o.protocol == 'https:') {
        return https.request({
            agent: this.agent,
            host: this.uri_o.hostname,
            port: 443,
            headers: this.headers,
            path: this.proxy.hostname ?
                 this.uri :
                 ((this.uri_o.pathname || '') +
                 (this.uri_o.search || '') +
                 (this.uri_o.hash || '')) || '/'
        }, callback);
    } else {
        return http.request({
            agent: this.agent,
            port: this.proxy.port || this.uri_o.port || 80,
            host: this.proxy.hostname || this.uri_o.hostname,
            headers: this.headers,
            path: this.proxy.hostname ?
                 this.uri :
                 ((this.uri_o.pathname || '') +
                 (this.uri_o.search || '') +
                 (this.uri_o.hash || '')) || '/'
        }, callback);
    }
};  


// Innermost API function of Get
//
// - @param {Function} callback
// - @param {Number} times number of times re-called.
Get.prototype.perform = function(callback, times) {
    if (times > this.max_redirs) {
        return callback(new Error('Redirect limit of ' +
            this.max_redirs +
            ' reached'));
    }

    times = times || 1;

    var clientrequest = this.request(function handleClientRequest(response, err) {
        if (err) return callback.call(this, err, null);
        response.resume();
        if (response.statusCode >= 300 &&
            response.statusCode < 400 &&
            response.headers.location) {
            // Redirection
            // -----------
            // Servers can send a full redirect location
            // or a short form, like a hyperlink. Handle both.
            if (url.parse(response.headers.location).protocol) {
                this.uri = response.headers.location;
            } else {
                this.uri = url.resolve(this.uri, response.headers.location);
            }
            this.perform(callback, times + 1);
            return;
        } else if (response.statusCode >= 400) {
            // failure
            var err = new Error('Server returned HTTP ' + response.statusCode);
            err.status = response.statusCode;
            return callback.call(this, err, response);
        } else {
            // success
            return callback.call(this, null, response);
        }
    }.bind(this));

    // The client can fail if the url is invalid
    if (clientrequest) {
        // Ensure the callback is only called once in error cases.
        // Timeouts can trigger both error and timeout callbacks.
        var error = 0;

        // Handle DNS-level errors, like ECONNREFUSED
        clientrequest.on('error', function(err) {
            if (++error > 1) return;
            return callback.call(this, err);
        }.bind(this));

        // Enforce a timeout of 10 seconds.
        // Add a no-op version of setTimeout for node <= 0.4.x.
        if (this.timeout > 0) {
            clientrequest.setTimeout = clientrequest.setTimeout || function() {};
            clientrequest.setTimeout(this.timeout, function() {
                clientrequest.connection.end();
                if (++error > 1) return;
                var err = new Error('Timed out after ' + this.timeout + 'ms');
                err.status = 504; // HTTP status code for "Gateway Timeout".
                return callback.call(this, err);
            }.bind(this));
        }

        // TODO: fix when/if gzip is supported.
        // If a proxy is defined, ask for the full requested URL,
        // otherwise construct the URL without a hostname and protocol.
        clientrequest.end();
    }
};

Get.prototype.guessResponseExtension = function(response) {
    if (response.headers['content-disposition']) {
        var match = response.headers['content-disposition'].match(/filename=\"([^"]+)\"/);
        if (match) {
            var ext = path.extname(match[1]);
            if (ext) {
                return ext;
            }
        }
    }
    return false;
};

// Stream a file to disk
// ---------------------
// - @param {String} filename.
// - @param {Function} callback.
Get.prototype.toDisk = function(filename, callback) {
    // TODO: catch all errors
    this.perform(function(err, response) {
        if (err) return callback(err);

        // Don't set an encoding. Using an encoding messes up binary files.
        // Pump contents from the response stream into a new writestream.
        var file = fs.createWriteStream(filename);
        file.on('error', callback);
        file.on('close', function() {
            return callback(null, filename, response, this);
        }.bind(this));
        response.pipe(file);
    });
};


// Get the contents of a URL as a string
//
// - @param {Function} callback.
Get.prototype.asString = function(callback) {
    var max_length = this.max_length;
    var payload = 0;

    function error(err) {
        if (!callback) return;
        callback(err);
        callback = null;
    }

    // TODO: catch all errors
    this.perform(function pipeResponseToString(err, response) {
        if (err) return callback(err);

        var mime = (response.headers['content-type'] || '').toLowerCase();
        if (mime !== 'application/json') switch (mime.split('/')[0]) {
            case 'binary':
            case 'application':
            case 'image':
            case 'video':
                return callback(new Error("Can't download binary file as string"));
            default:
                // TODO: respect Content-Transfer-Encoding header
                response.setEncoding(this.guessEncoding(this.uri));
        }

        function returnString() {
            if (!callback) return;

            var err = checkContentLength(response.headers, payload);
            if (err) return error(err);

            callback(null, out.join(''), response.headers);
            callback = null;
        }

        // Fill an array with chunks of data,
        // and then join it into a string before calling `callback`
        var out = [];
        response.on('data', function(chunk) {
            if (!callback) return;
            payload += chunk.length;
            var err = checkMaxLength(max_length, payload);
            if (err) {
                response.socket.end();
                error(err);
            } else {
                out.push(chunk);
            }
        });
        response.on('error', error);
        response.on('end', returnString);
        response.on('close', returnString);
    });
};

// Get the contents of a URL as a buffer
//
// - @param {Function} callback.
Get.prototype.asBuffer = function(callback) {
    var max_length = this.max_length;
    var payload = 0;

    function error(err) {
        if (!callback) return;
        callback(err);
        callback = null;
    }

    this.perform(function(err, response) {
        if (err) return error(err);

        function returnBuffer() {
            if (!callback) return;

            var err = checkContentLength(response.headers, payload);
            if (err) return error(err);

            for (var length = 0, i = 0; i < out.length; i++) {
                length += out[i].length;
            }
            var result = new Buffer(length);
            for (var pos = 0, j = 0; j < out.length; j++) {
                out[j].copy(result, pos);
                pos += out[j].length;
            }
            callback(null, result, response.headers);
            callback = null;
        }

        // Fill an array with chunks of data,
        // and then join it into a buffer before calling `callback`
        var out = [];
        response.on('data', function(chunk) {
            if (!callback) return;
            payload += chunk.length;
            var err = checkMaxLength(max_length, payload);
            if (err) {
                response.socket.end();
                error(err);
            } else {
                out.push(chunk);
            }
        });
        response.on('error', error);
        response.on('end', returnBuffer);
        response.on('close', returnBuffer);
    });
};

Get.prototype.guessEncoding = function(location) {
    // The 'most reliable' measure is probably the end of files, so
    // start off with extname.
    if (this.encoding) return this.encoding;
    var ext = path.extname(location).toLowerCase();
    if (encodings.ext[ext]) return encodings.ext[ext];
};

function checkMaxLength(max, length) {
    if (!max) return;
    if (length <= max) return;
    return new Error('File exceeds maximum allowed length of ' + max + ' bytes');
}

function checkContentLength(headers, length) {
    if (!headers['content-length']) return;
    var contentLength = parseInt(headers['content-length'], 10);
    if (isNaN(contentLength)) return;
    if (length === contentLength) return;
    return new Error('Body ('+length+' bytes) does not match content-length (' + contentLength + ' bytes)');
}

module.exports = Get;
module.exports.checkMaxLength = checkMaxLength;
module.exports.checkContentLength = checkContentLength;
