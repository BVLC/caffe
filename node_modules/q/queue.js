
var Q = require("./q");

module.exports = Queue;
function Queue() {
    var ends = Q.defer();
    var closed = Q.defer();
    return {
        put: function (value) {
            var next = Q.defer();
            ends.resolve({
                head: value,
                tail: next.promise
            });
            ends.resolve = next.resolve;
        },
        get: function () {
            var result = ends.promise.get("head");
            ends.promise = ends.promise.get("tail");
            return result.fail(function (reason) {
                closed.resolve();
                throw reason;
            });
        },
        closed: closed.promise,
        close: function (reason) {
            reason = reason || new Error("Can't get value from closed queue");
            var end = {head: Q.reject(reason)};
            end.tail = end;
            ends.resolve(end);
            return closed.promise;
        }
    };
}

