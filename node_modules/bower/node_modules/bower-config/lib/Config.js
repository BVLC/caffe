var lang = require('mout/lang');
var object = require('mout/object');
var rc = require('./util/rc');
var expand = require('./util/expand');
var EnvProxy = require('./util/proxy');
var path = require('path');
var fs = require('fs');

function Config(cwd) {
    this._cwd = cwd;
    this._proxy = new EnvProxy();
    this._config = {};
}

Config.prototype.load = function (overwrites) {
    this._config = rc('bower', this._cwd);

    this._config = object.merge(
      expand(this._config || {}),
      expand(overwrites || {})
    );

    this._config = normalise(this._config);

    this._proxy.set(this._config);

    return this;
};

Config.prototype.restore = function () {
  this._proxy.restore();
};

function readCertFile(path) {
    path = path || '';

    var sep = '-----END CERTIFICATE-----';

    var certificates;

    if (path.indexOf(sep) === -1) {
        certificates = fs.readFileSync(path, { encoding: 'utf8' });
    } else {
        certificates = path;
    }

    return certificates.
      split(sep).
      filter(function(s) { return !s.match(/^\s*$/); }).
      map(function(s) { return s + sep; });
}

function loadCAs(caConfig) {
    // If a ca file path has been specified, expand that here to the file's
    // contents. As a user can specify these individually, we must load them
    // one by one.
    for (var p in caConfig) {
        if (caConfig.hasOwnProperty(p)) {
            var prop = caConfig[p];
            if (Array.isArray(prop)) {
                caConfig[p] = prop.map(function(s) {
                    return readCertFile(s);
                });
            } else if (prop) {
                caConfig[p] = readCertFile(prop);
            }
        }
    }
}

Config.prototype.toObject = function () {
    return lang.deepClone(this._config);
};

Config.create = function (cwd) {
    return new Config(cwd);
};

Config.read = function (cwd, overrides) {
    var config = Config.create(cwd);
    return config.load(overrides).toObject();
};

function normalise(config) {
    config = expand(config);

    // Some backwards compatible things..
    if (config.shorthandResolver) {
      config.shorthandResolver = config.shorthandResolver
        .replace(/\{\{\{/g, '{{')
        .replace(/\}\}\}/g, '}}');
    }

    // Ensure that every registry endpoint does not end with /
    config.registry.search = config.registry.search.map(function (url) {
        return url.replace(/\/+$/, '');
    });
    config.registry.register = config.registry.register.replace(/\/+$/, '');
    config.registry.publish = config.registry.publish.replace(/\/+$/, '');
    config.tmp = path.resolve(config.tmp);

    loadCAs(config.ca);

    return config;
}

Config.DEFAULT_REGISTRY = require('./util/defaults').registry;

module.exports = Config;
