var inherits = require('inherits');

var NestedError = function (message, nested) {
    this.nested = nested;

    Error.captureStackTrace(this, this.constructor);

    var oldStackDescriptor = Object.getOwnPropertyDescriptor(this, 'stack');

    if (typeof message !== 'undefined') {
        Object.defineProperty(this, 'message', {
            value: message,
            writable: true,
            enumerable: false,
            configurable: true
        });
    }

    Object.defineProperties(this, {
        stack: {
            get: function () {
                var stack = oldStackDescriptor.get.call(this);
                if (this.nested) {
                    stack += '\nCaused By: ' + this.nested.stack;
                }
                return stack;
            }
        }

    });
};

inherits(NestedError, Error);
NestedError.prototype.name = 'NestedError';


module.exports = NestedError;
