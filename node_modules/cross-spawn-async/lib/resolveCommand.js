'use strict';

var path  = require('path');
var which = require('which');
var LRU   = require('lru-cache');

var resolveCache = LRU({ max: 50, maxAge: 30 * 1000 });  // Cache just for 30sec

function resolveCommand(command, noExtension) {
    var resolved;

    noExtension = !!noExtension;
    resolved = resolveCache.get(command + '!' + noExtension);

    // Check if its resolved in the cache
    if (resolveCache.has(command)) {
        return resolveCache.get(command);
    }

    try {
        resolved = !noExtension ?
            which.sync(command) :
            which.sync(command, { pathExt: path.delimiter + (process.env.PATHEXT || '')  });
    } catch (e) {}

    resolveCache.set(command + '!' + noExtension, resolved);

    return resolved;
}

module.exports = resolveCommand;
