'use strict';

function mixIn(target, source) {
    var key;

    // No need to check for hasOwnProperty.. this is used
    // just in plain objects
    for (key in source) {
        target[key] = source[key];
    }

    return target;
}

module.exports = mixIn;
