var object = require('mout/object');
var lang = require('mout/lang');
var string = require('mout/string');

function camelCase(config) {
    var camelCased = {};

    // Camel case
    object.forOwn(config, function (value, key) {
        // Ignore null values
        if (value == null) {
            return;
        }

        key = string.camelCase(key.replace(/_/g, '-'));
        camelCased[key] = lang.isPlainObject(value) ? camelCase(value) : value;
    });

    return camelCased;
}

function expand(config) {
    config = camelCase(config);

    if (typeof config.registry === 'string') {
        config.registry = {
            default: config.registry,
            search: [config.registry],
            register: config.registry,
            publish: config.registry
        };
    } else if (typeof config.registry === 'object') {
        config.registry.default = config.registry.default || 'https://bower.herokuapp.com';

        config.registry = {
            default: config.registry.default,
            search: config.registry.search || config.registry.default,
            register: config.registry.register || config.registry.default,
            publish: config.registry.publish || config.registry.default
        };

        if (config.registry.search && !Array.isArray(config.registry.search)) {
            config.registry.search = [config.registry.search];
        }
    }

    // CA
    if (typeof config.ca === 'string') {
        config.ca = {
            default: config.ca,
            search: [config.ca],
            register: config.ca,
            publish: config.ca
        };
    } else if (typeof config.ca === 'object') {
        if (config.ca.search && !Array.isArray(config.ca.search)) {
            config.ca.search = [config.ca.search];
        }

        if (config.ca.default) {
            config.ca.search = config.ca.search || config.ca.default;
            config.ca.register = config.ca.register || config.ca.default;
            config.ca.publish = config.ca.publish || config.ca.default;
        }
    }

    return config;
}

module.exports = expand;
