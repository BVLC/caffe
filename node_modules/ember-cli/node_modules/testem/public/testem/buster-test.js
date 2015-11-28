(function () { var when; function define(cb) { when = cb(); } /** @license MIT License (c) copyright B Cavalier & J Hann */

/**
 * when
 * A lightweight CommonJS Promises/A and when() implementation
 *
 * when is part of the cujo.js family of libraries (http://cujojs.com/)
 *
 * Licensed under the MIT License at:
 * http://www.opensource.org/licenses/mit-license.php
 *
 * @version 1.3.0
 */

(function(define) {
define(function() {
    var freeze, reduceArray, slice, undef;

    //
    // Public API
    //

    when.defer     = defer;
    when.reject    = reject;
    when.isPromise = isPromise;

    when.all       = all;
    when.some      = some;
    when.any       = any;

    when.map       = map;
    when.reduce    = reduce;

    when.chain     = chain;

    /** Object.freeze */
    freeze = Object.freeze || function(o) { return o; };

    /**
     * Trusted Promise constructor.  A Promise created from this constructor is
     * a trusted when.js promise.  Any other duck-typed promise is considered
     * untrusted.
     *
     * @constructor
     */
    function Promise() {}

    Promise.prototype = freeze({
        always: function(alwaysback, progback) {
            return this.then(alwaysback, alwaysback, progback);
        },

        otherwise: function(errback) {
            return this.then(undef, errback);
        }
    });

    /**
     * Create an already-resolved promise for the supplied value
     * @private
     *
     * @param value anything
     * @return {Promise}
     */
    function resolved(value) {

        var p = new Promise();

        p.then = function(callback) {
            try {
                return promise(callback ? callback(value) : value);
            } catch(e) {
                return rejected(e);
            }
        };

        return freeze(p);
    }

    /**
     * Create an already-rejected {@link Promise} with the supplied
     * rejection reason.
     * @private
     *
     * @param reason rejection reason
     * @return {Promise}
     */
    function rejected(reason) {

        var p = new Promise();

        p.then = function(callback, errback) {
            try {
                return errback ? promise(errback(reason)) : rejected(reason);
            } catch(e) {
                return rejected(e);
            }
        };

        return freeze(p);
    }

    /**
     * Returns a rejected promise for the supplied promiseOrValue. If
     * promiseOrValue is a value, it will be the rejection value of the
     * returned promise.  If promiseOrValue is a promise, its
     * completion value will be the rejected value of the returned promise
     *
     * @param promiseOrValue {*} the rejected value of the returned {@link Promise}
     *
     * @return {Promise} rejected {@link Promise}
     */
    function reject(promiseOrValue) {
        return when(promiseOrValue, function(value) {
            return rejected(value);
        });
    }

    /**
     * Creates a new, CommonJS compliant, Deferred with fully isolated
     * resolver and promise parts, either or both of which may be given out
     * safely to consumers.
     * The Deferred itself has the full API: resolve, reject, progress, and
     * then. The resolver has resolve, reject, and progress.  The promise
     * only has then.
     *
     * @memberOf when
     * @function
     *
     * @returns {Deferred}
     */
    function defer() {
        var deferred, promise, listeners, progressHandlers, _then, _progress, complete;

        listeners = [];
        progressHandlers = [];

        /**
         * Pre-resolution then() that adds the supplied callback, errback, and progback
         * functions to the registered listeners
         *
         * @private
         *
         * @param [callback] {Function} resolution handler
         * @param [errback] {Function} rejection handler
         * @param [progback] {Function} progress handler
         *
         * @throws {Error} if any argument is not null, undefined, or a Function
         */
        _then = function unresolvedThen(callback, errback, progback) {
            var deferred = defer();

            listeners.push(function(promise) {
                promise.then(callback, errback)
                    .then(deferred.resolve, deferred.reject, deferred.progress);
            });

            progback && progressHandlers.push(progback);

            return deferred.promise;
        };

        /**
         * Registers a handler for this {@link Deferred}'s {@link Promise}.  Even though all arguments
         * are optional, each argument that *is* supplied must be null, undefined, or a Function.
         * Any other value will cause an Error to be thrown.
         *
         * @memberOf Promise
         *
         * @param [callback] {Function} resolution handler
         * @param [errback] {Function} rejection handler
         * @param [progback] {Function} progress handler
         *
         * @throws {Error} if any argument is not null, undefined, or a Function
         */
        function then(callback, errback, progback) {
            return _then(callback, errback, progback);
        }

        /**
         * Resolves this {@link Deferred}'s {@link Promise} with val as the
         * resolution value.
         *
         * @memberOf Resolver
         *
         * @param val anything
         */
        function resolve(val) {
            complete(resolved(val));
        }

        /**
         * Rejects this {@link Deferred}'s {@link Promise} with err as the
         * reason.
         *
         * @memberOf Resolver
         *
         * @param err anything
         */
        function reject(err) {
            complete(rejected(err));
        }

        /**
         * @private
         * @param update
         */
        _progress = function(update) {
            var progress, i = 0;
            while (progress = progressHandlers[i++]) progress(update);
        };

        /**
         * Emits a progress update to all progress observers registered with
         * this {@link Deferred}'s {@link Promise}
         *
         * @memberOf Resolver
         *
         * @param update anything
         */
        function progress(update) {
            _progress(update);
        }

        /**
         * Transition from pre-resolution state to post-resolution state, notifying
         * all listeners of the resolution or rejection
         *
         * @private
         *
         * @param completed {Promise} the completed value of this deferred
         */
        complete = function(completed) {
            var listener, i = 0;

            // Replace _then with one that directly notifies with the result.
            _then = completed.then;

            // Replace complete so that this Deferred can only be completed
            // once. Also Replace _progress, so that subsequent attempts to issue
            // progress throw.
            complete = _progress = function alreadyCompleted() {
                // TODO: Consider silently returning here so that parties who
                // have a reference to the resolver cannot tell that the promise
                // has been resolved using try/catch
                throw new Error("already completed");
            };

            // Free progressHandlers array since we'll never issue progress events
            // for this promise again now that it's completed
            progressHandlers = undef;

            // Notify listeners
            // Traverse all listeners registered directly with this Deferred

            while (listener = listeners[i++]) {
                listener(completed);
            }

            listeners = [];
        };

        /**
         * The full Deferred object, with both {@link Promise} and {@link Resolver}
         * parts
         * @class Deferred
         * @name Deferred
         */
        deferred = {};

        // Promise and Resolver parts
        // Freeze Promise and Resolver APIs

        promise = new Promise();
        promise.then = deferred.then = then;

        /**
         * The {@link Promise} for this {@link Deferred}
         * @memberOf Deferred
         * @name promise
         * @type {Promise}
         */
        deferred.promise = freeze(promise);

        /**
         * The {@link Resolver} for this {@link Deferred}
         * @memberOf Deferred
         * @name resolver
         * @class Resolver
         */
        deferred.resolver = freeze({
            resolve:  (deferred.resolve  = resolve),
            reject:   (deferred.reject   = reject),
            progress: (deferred.progress = progress)
        });

        return deferred;
    }

    /**
     * Determines if promiseOrValue is a promise or not.  Uses the feature
     * test from http://wiki.commonjs.org/wiki/Promises/A to determine if
     * promiseOrValue is a promise.
     *
     * @param promiseOrValue anything
     *
     * @returns {Boolean} true if promiseOrValue is a {@link Promise}
     */
    function isPromise(promiseOrValue) {
        return promiseOrValue && typeof promiseOrValue.then === 'function';
    }

    /**
     * Register an observer for a promise or immediate value.
     *
     * @function
     * @name when
     * @namespace
     *
     * @param promiseOrValue anything
     * @param {Function} [callback] callback to be called when promiseOrValue is
     *   successfully resolved.  If promiseOrValue is an immediate value, callback
     *   will be invoked immediately.
     * @param {Function} [errback] callback to be called when promiseOrValue is
     *   rejected.
     * @param {Function} [progressHandler] callback to be called when progress updates
     *   are issued for promiseOrValue.
     *
     * @returns {Promise} a new {@link Promise} that will complete with the return
     *   value of callback or errback or the completion value of promiseOrValue if
     *   callback and/or errback is not supplied.
     */
    function when(promiseOrValue, callback, errback, progressHandler) {
        // Get a promise for the input promiseOrValue
        // See promise()
        var trustedPromise = promise(promiseOrValue);

        // Register promise handlers
        return trustedPromise.then(callback, errback, progressHandler);
    }

    /**
     * Returns promiseOrValue if promiseOrValue is a {@link Promise}, a new Promise if
     * promiseOrValue is a foreign promise, or a new, already-resolved {@link Promise}
     * whose resolution value is promiseOrValue if promiseOrValue is an immediate value.
     *
     * Note that this function is not safe to export since it will return its
     * input when promiseOrValue is a {@link Promise}
     *
     * @private
     *
     * @param promiseOrValue anything
     *
     * @returns Guaranteed to return a trusted Promise.  If promiseOrValue is a when.js {@link Promise}
     *   returns promiseOrValue, otherwise, returns a new, already-resolved, when.js {@link Promise}
     *   whose resolution value is:
     *   * the resolution value of promiseOrValue if it's a foreign promise, or
     *   * promiseOrValue if it's a value
     */
    function promise(promiseOrValue) {
        var promise, deferred;

        if(promiseOrValue instanceof Promise) {
            // It's a when.js promise, so we trust it
            promise = promiseOrValue;

        } else {
            // It's not a when.js promise.  Check to see if it's a foreign promise
            // or a value.

            deferred = defer();
            if(isPromise(promiseOrValue)) {
                // It's a compliant promise, but we don't know where it came from,
                // so we don't trust its implementation entirely.  Introduce a trusted
                // middleman when.js promise

                // IMPORTANT: This is the only place when.js should ever call .then() on
                // an untrusted promise.
                promiseOrValue.then(deferred.resolve, deferred.reject, deferred.progress);
                promise = deferred.promise;

            } else {
                // It's a value, not a promise.  Create an already-resolved promise
                // for it.
                deferred.resolve(promiseOrValue);
                promise = deferred.promise;
            }
        }

        return promise;
    }

    /**
     * Return a promise that will resolve when howMany of the supplied promisesOrValues
     * have resolved. The resolution value of the returned promise will be an array of
     * length howMany containing the resolutions values of the triggering promisesOrValues.
     *
     * @memberOf when
     *
     * @param promisesOrValues {Array} array of anything, may contain a mix
     *      of {@link Promise}s and values
     * @param howMany
     * @param [callback]
     * @param [errback]
     * @param [progressHandler]
     *
     * @returns {Promise}
     */
    function some(promisesOrValues, howMany, callback, errback, progressHandler) {

        checkCallbacks(2, arguments);

        return when(promisesOrValues, function(promisesOrValues) {

            var toResolve, results, ret, deferred, resolver, rejecter, handleProgress, len, i;

            len = promisesOrValues.length >>> 0;

            toResolve = Math.max(0, Math.min(howMany, len));
            results = [];
            deferred = defer();
            ret = when(deferred, callback, errback, progressHandler);

            // Wrapper so that resolver can be replaced
            function resolve(val) {
                resolver(val);
            }

            // Wrapper so that rejecter can be replaced
            function reject(err) {
                rejecter(err);
            }

            // Wrapper so that progress can be replaced
            function progress(update) {
                handleProgress(update);
            }

            function complete() {
                resolver = rejecter = handleProgress = noop;
            }

            // No items in the input, resolve immediately
            if (!toResolve) {
                deferred.resolve(results);

            } else {
                // Resolver for promises.  Captures the value and resolves
                // the returned promise when toResolve reaches zero.
                // Overwrites resolver var with a noop once promise has
                // be resolved to cover case where n < promises.length
                resolver = function(val) {
                    // This orders the values based on promise resolution order
                    // Another strategy would be to use the original position of
                    // the corresponding promise.
                    results.push(val);

                    if (!--toResolve) {
                        complete();
                        deferred.resolve(results);
                    }
                };

                // Rejecter for promises.  Rejects returned promise
                // immediately, and overwrites rejecter var with a noop
                // once promise to cover case where n < promises.length.
                // TODO: Consider rejecting only when N (or promises.length - N?)
                // promises have been rejected instead of only one?
                rejecter = function(err) {
                    complete();
                    deferred.reject(err);
                };

                handleProgress = deferred.progress;

                // TODO: Replace while with forEach
                for(i = 0; i < len; ++i) {
                    if(i in promisesOrValues) {
                        when(promisesOrValues[i], resolve, reject, progress);
                    }
                }
            }

            return ret;
        });
    }

    /**
     * Return a promise that will resolve only once all the supplied promisesOrValues
     * have resolved. The resolution value of the returned promise will be an array
     * containing the resolution values of each of the promisesOrValues.
     *
     * @memberOf when
     *
     * @param promisesOrValues {Array|Promise} array of anything, may contain a mix
     *      of {@link Promise}s and values
     * @param [callback] {Function}
     * @param [errback] {Function}
     * @param [progressHandler] {Function}
     *
     * @returns {Promise}
     */
    function all(promisesOrValues, callback, errback, progressHandler) {

        checkCallbacks(1, arguments);

        return when(promisesOrValues, function(promisesOrValues) {
            return _reduce(promisesOrValues, reduceIntoArray, []);
        }).then(callback, errback, progressHandler);
    }

    function reduceIntoArray(current, val, i) {
        current[i] = val;
        return current;
    }

    /**
     * Return a promise that will resolve when any one of the supplied promisesOrValues
     * has resolved. The resolution value of the returned promise will be the resolution
     * value of the triggering promiseOrValue.
     *
     * @memberOf when
     *
     * @param promisesOrValues {Array|Promise} array of anything, may contain a mix
     *      of {@link Promise}s and values
     * @param [callback] {Function}
     * @param [errback] {Function}
     * @param [progressHandler] {Function}
     *
     * @returns {Promise}
     */
    function any(promisesOrValues, callback, errback, progressHandler) {

        function unwrapSingleResult(val) {
            return callback ? callback(val[0]) : val[0];
        }

        return some(promisesOrValues, 1, unwrapSingleResult, errback, progressHandler);
    }

    /**
     * Traditional map function, similar to `Array.prototype.map()`, but allows
     * input to contain {@link Promise}s and/or values, and mapFunc may return
     * either a value or a {@link Promise}
     *
     * @memberOf when
     *
     * @param promise {Array|Promise} array of anything, may contain a mix
     *      of {@link Promise}s and values
     * @param mapFunc {Function} mapping function mapFunc(value) which may return
     *      either a {@link Promise} or value
     *
     * @returns {Promise} a {@link Promise} that will resolve to an array containing
     *      the mapped output values.
     */
    function map(promise, mapFunc) {
        return when(promise, function(array) {
            return _map(array, mapFunc);
        });
    }

    /**
     * Private map helper to map an array of promises
     * @private
     *
     * @param promisesOrValues {Array}
     * @param mapFunc {Function}
     * @return {Promise}
     */
    function _map(promisesOrValues, mapFunc) {

        var results, len, i;

        // Since we know the resulting length, we can preallocate the results
        // array to avoid array expansions.
        len = promisesOrValues.length >>> 0;
        results = new Array(len);

        // Since mapFunc may be async, get all invocations of it into flight
        // asap, and then use reduce() to collect all the results
        for(i = 0; i < len; i++) {
            if(i in promisesOrValues)
                results[i] = when(promisesOrValues[i], mapFunc);
        }

        // Could use all() here, but that would result in another array
        // being allocated, i.e. map() would end up allocating 2 arrays
        // of size len instead of just 1.  Since all() uses reduce()
        // anyway, avoid the additional allocation by calling reduce
        // directly.
        return _reduce(results, reduceIntoArray, results);
    }

    /**
     * Traditional reduce function, similar to `Array.prototype.reduce()`, but
     * input may contain {@link Promise}s and/or values, and reduceFunc
     * may return either a value or a {@link Promise}, *and* initialValue may
     * be a {@link Promise} for the starting value.
     *
     * @memberOf when
     *
     * @param promise {Array|Promise} array of anything, may contain a mix
     *      of {@link Promise}s and values.  May also be a {@link Promise} for
     *      an array.
     * @param reduceFunc {Function} reduce function reduce(currentValue, nextValue, index, total),
     *      where total is the total number of items being reduced, and will be the same
     *      in each call to reduceFunc.
     * @param initialValue starting value, or a {@link Promise} for the starting value
     *
     * @returns {Promise} that will resolve to the final reduced value
     */
    function reduce(promise, reduceFunc, initialValue) {
        var args = slice.call(arguments, 1);
        return when(promise, function(array) {
            return _reduce.apply(undef, [array].concat(args));
        });
    }

    /**
     * Private reduce to reduce an array of promises
     * @private
     *
     * @param promisesOrValues {Array}
     * @param reduceFunc {Function}
     * @param initialValue {*}
     * @return {Promise}
     */
    function _reduce(promisesOrValues, reduceFunc, initialValue) {

        var total, args;

        total = promisesOrValues.length;

        // Skip promisesOrValues, since it will be used as 'this' in the call
        // to the actual reduce engine below.

        // Wrap the supplied reduceFunc with one that handles promises and then
        // delegates to the supplied.

        args = [
            function (current, val, i) {
                return when(current, function (c) {
                    return when(val, function (value) {
                        return reduceFunc(c, value, i, total);
                    });
                });
            }
        ];

        if (arguments.length > 2) args.push(initialValue);

        return reduceArray.apply(promisesOrValues, args);
    }

    /**
     * Ensure that resolution of promiseOrValue will complete resolver with the completion
     * value of promiseOrValue, or instead with resolveValue if it is provided.
     *
     * @memberOf when
     *
     * @param promiseOrValue
     * @param resolver {Resolver}
     * @param [resolveValue] anything
     *
     * @returns {Promise}
     */
    function chain(promiseOrValue, resolver, resolveValue) {
        var useResolveValue = arguments.length > 2;

        return when(promiseOrValue,
            function(val) {
                if(useResolveValue) val = resolveValue;
                resolver.resolve(val);
                return val;
            },
            function(e) {
                resolver.reject(e);
                return rejected(e);
            },
            resolver.progress
        );
    }

    //
    // Utility functions
    //

    /**
     * Helper that checks arrayOfCallbacks to ensure that each element is either
     * a function, or null or undefined.
     *
     * @private
     *
     * @param arrayOfCallbacks {Array} array to check
     * @throws {Error} if any element of arrayOfCallbacks is something other than
     * a Functions, null, or undefined.
     */
    function checkCallbacks(start, arrayOfCallbacks) {
        var arg, i = arrayOfCallbacks.length;
        while(i > start) {
            arg = arrayOfCallbacks[--i];
            if (arg != null && typeof arg != 'function') throw new Error('callback is not a function');
        }
    }

    /**
     * No-Op function used in method replacement
     * @private
     */
    function noop() {}

    slice = [].slice;

    // ES5 reduce implementation if native not available
    // See: http://es5.github.com/#x15.4.4.21 as there are many
    // specifics and edge cases.
    reduceArray = [].reduce ||
        function(reduceFunc /*, initialValue */) {
            // ES5 dictates that reduce.length === 1

            // This implementation deviates from ES5 spec in the following ways:
            // 1. It does not check if reduceFunc is a Callable

            var arr, args, reduced, len, i;

            i = 0;
            arr = Object(this);
            len = arr.length >>> 0;
            args = arguments;

            // If no initialValue, use first item of array (we know length !== 0 here)
            // and adjust i to start at second item
            if(args.length <= 1) {
                // Skip to the first real element in the array
                for(;;) {
                    if(i in arr) {
                        reduced = arr[i++];
                        break;
                    }

                    // If we reached the end of the array without finding any real
                    // elements, it's a TypeError
                    if(++i >= len) {
                        throw new TypeError();
                    }
                }
            } else {
                // If initialValue provided, use it
                reduced = args[1];
            }

            // Do the actual reduce
            for(;i < len; ++i) {
                // Skip holes
                if(i in arr)
                    reduced = reduceFunc(reduced, arr[i], i, arr);
            }

            return reduced;
        };

    return when;
});
})(typeof define == 'function'
    ? define
    : function (factory) { typeof module != 'undefined'
        ? (module.exports = factory())
        : (this.when      = factory());
    }
    // Boilerplate for AMD, Node, and browser global
);

var buster = (function (setTimeout, B) {
    var isNode = typeof require == "function" && typeof module == "object";
    var div = typeof document != "undefined" && document.createElement("div");
    var F = function () {};

    var buster = {
        bind: function bind(obj, methOrProp) {
            var method = typeof methOrProp == "string" ? obj[methOrProp] : methOrProp;
            var args = Array.prototype.slice.call(arguments, 2);
            return function () {
                var allArgs = args.concat(Array.prototype.slice.call(arguments));
                return method.apply(obj, allArgs);
            };
        },

        partial: function partial(fn) {
            var args = [].slice.call(arguments, 1);
            return function () {
                return fn.apply(this, args.concat([].slice.call(arguments)));
            };
        },

        create: function create(object) {
            F.prototype = object;
            return new F();
        },

        extend: function extend(target) {
            if (!target) { return; }
            for (var i = 1, l = arguments.length, prop; i < l; ++i) {
                for (prop in arguments[i]) {
                    target[prop] = arguments[i][prop];
                }
            }
            return target;
        },

        nextTick: function nextTick(callback) {
            if (typeof process != "undefined" && process.nextTick) {
                return process.nextTick(callback);
            }
            setTimeout(callback, 0);
        },

        functionName: function functionName(func) {
            if (!func) return "";
            if (func.displayName) return func.displayName;
            if (func.name) return func.name;
            var matches = func.toString().match(/function\s+([^\(]+)/m);
            return matches && matches[1] || "";
        },

        isNode: function isNode(obj) {
            if (!div) return false;
            try {
                obj.appendChild(div);
                obj.removeChild(div);
            } catch (e) {
                return false;
            }
            return true;
        },

        isElement: function isElement(obj) {
            return obj && obj.nodeType === 1 && buster.isNode(obj);
        },

        isArray: function isArray(arr) {
            return Object.prototype.toString.call(arr) == "[object Array]";
        },

        flatten: function flatten(arr) {
            var result = [], arr = arr || [];
            for (var i = 0, l = arr.length; i < l; ++i) {
                result = result.concat(buster.isArray(arr[i]) ? flatten(arr[i]) : arr[i]);
            }
            return result;
        },

        each: function each(arr, callback) {
            for (var i = 0, l = arr.length; i < l; ++i) {
                callback(arr[i]);
            }
        },

        map: function map(arr, callback) {
            var results = [];
            for (var i = 0, l = arr.length; i < l; ++i) {
                results.push(callback(arr[i]));
            }
            return results;
        },

        parallel: function parallel(fns, callback) {
            function cb(err, res) {
                if (typeof callback == "function") {
                    callback(err, res);
                    callback = null;
                }
            }
            if (fns.length == 0) { return cb(null, []); }
            var remaining = fns.length, results = [];
            function makeDone(num) {
                return function done(err, result) {
                    if (err) { return cb(err); }
                    results[num] = result;
                    if (--remaining == 0) { cb(null, results); }
                };
            }
            for (var i = 0, l = fns.length; i < l; ++i) {
                fns[i](makeDone(i));
            }
        },

        series: function series(fns, callback) {
            function cb(err, res) {
                if (typeof callback == "function") {
                    callback(err, res);
                }
            }
            var remaining = fns.slice();
            var results = [];
            function callNext() {
                if (remaining.length == 0) return cb(null, results);
                var promise = remaining.shift()(next);
                if (promise && typeof promise.then == "function") {
                    promise.then(buster.partial(next, null), next);
                }
            }
            function next(err, result) {
                if (err) return cb(err);
                results.push(result);
                callNext();
            }
            callNext();
        },

        countdown: function countdown(num, done) {
            return function () {
                if (--num == 0) done();
            };
        }
    };

    if (typeof process === "object") {
        var crypto = require("crypto");
        var path = require("path");

        buster.tmpFile = function (fileName) {
            var hashed = crypto.createHash("sha1");
            hashed.update(fileName);
            var tmpfileName = hashed.digest("hex");

            if (process.platform == "win32") {
                return path.join(process.env["TEMP"], tmpfileName);
            } else {
                return path.join("/tmp", tmpfileName);
            }
        };
    }

    if (Array.prototype.some) {
        buster.some = function (arr, fn, thisp) {
            return arr.some(fn, thisp);
        };
    } else {
        // https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/some
        buster.some = function (arr, fun, thisp) {
            "use strict";
            if (arr == null) { throw new TypeError(); }
            arr = Object(arr);
            var len = arr.length >>> 0;
            if (typeof fun !== "function") { throw new TypeError(); }

            for (var i = 0; i < len; i++) {
                if (arr.hasOwnProperty(i) && fun.call(thisp, arr[i], i, arr)) {
                    return true;
                }
            }

            return false;
        };
    }

    if (Array.prototype.filter) {
        buster.filter = function (arr, fn, thisp) {
            return arr.filter(fn, thisp);
        };
    } else {
        // https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/filter
        buster.filter = function (fn, thisp) {
            "use strict";
            if (this == null) { throw new TypeError(); }

            var t = Object(this);
            var len = t.length >>> 0;
            if (typeof fn != "function") { throw new TypeError(); }

            var res = [];
            for (var i = 0; i < len; i++) {
                if (i in t) {
                    var val = t[i]; // in case fun mutates this
                    if (fn.call(thisp, val, i, t)) { res.push(val); }
                }
            }

            return res;
        };
    }

    if (isNode) {
        module.exports = buster;
        buster.eventEmitter = require("./buster-event-emitter");
        Object.defineProperty(buster, "defineVersionGetter", {
            get: function () {
                return require("./define-version-getter");
            }
        });
    }

    return buster.extend(B || {}, buster);
}(setTimeout, buster));

/*jslint eqeqeq: false, onevar: false, plusplus: false*/
/*global buster, require, module*/
if (typeof require == "function" && typeof module == "object") {
    var buster = require("./buster-core");
}

(function () {
    function eventListeners(eventEmitter, event) {
        if (!eventEmitter.listeners) {
            eventEmitter.listeners = {};
        }

        if (!eventEmitter.listeners[event]) {
            eventEmitter.listeners[event] = [];
        }

        return eventEmitter.listeners[event];
    }

    function throwLater(event, error) {
        buster.nextTick(function () {
            error.message = event + " listener threw error: " + error.message;
            throw error;
        });
    }

    function addSupervisor(emitter, listener, thisObject) {
        if (!emitter.supervisors) { emitter.supervisors = []; }
        emitter.supervisors.push({
            listener: listener,
            thisObject: thisObject
        });
    }

    function notifyListener(emitter, event, listener, args) {
        try {
            listener.listener.apply(listener.thisObject || emitter, args);
        } catch (e) {
            throwLater(event, e);
        }
    }

    buster.eventEmitter = {
        create: function () {
            return buster.create(this);
        },

        addListener: function addListener(event, listener, thisObject) {
            if (typeof event === "function") {
                return addSupervisor(this, event, listener);
            }
            if (typeof listener != "function") {
                throw new TypeError("Listener is not function");
            }
            eventListeners(this, event).push({
                listener: listener,
                thisObject: thisObject
            });
        },

        once: function once(event, listener, thisObject) {
            var self = this;
            this.addListener(event, listener);

            var wrapped = function () {
                self.removeListener(event, listener);
                self.removeListener(event, wrapped);
            };
            this.addListener(event, wrapped);
        },

        hasListener: function hasListener(event, listener, thisObject) {
            var listeners = eventListeners(this, event);

            for (var i = 0, l = listeners.length; i < l; i++) {
                if (listeners[i].listener === listener &&
                    listeners[i].thisObject === thisObject) {
                    return true;
                }
            }

            return false;
        },

        removeListener: function (event, listener) {
            var listeners = eventListeners(this, event);

            for (var i = 0, l = listeners.length; i < l; ++i) {
                if (listeners[i].listener == listener) {
                    listeners.splice(i, 1);
                    return;
                }
            }
        },

        emit: function emit(event) {
            var listeners = eventListeners(this, event).slice();
            var args = Array.prototype.slice.call(arguments, 1);

            for (var i = 0, l = listeners.length; i < l; i++) {
                notifyListener(this, event, listeners[i], args);
            }

            listeners = this.supervisors || [];
            args = Array.prototype.slice.call(arguments);
            for (i = 0, l = listeners.length; i < l; ++i) {
                notifyListener(this, event, listeners[i], args);
            }
        },

        bind: function (object, events) {
            var method;

            if (!events) {
                for (method in object) {
                    if (object.hasOwnProperty(method) && typeof object[method] == "function") {
                        this.addListener(method, object[method], object);
                    }
                }
            } else if (typeof events == "string" ||
                       Object.prototype.toString.call(events) == "[object Array]") {
                events = typeof events == "string" ? [events] : events;

                for (var i = 0, l = events.length; i < l; ++i) {
                    this.addListener(events[i], object[events[i]], object);
                }
            } else {
                for (var prop in events) {
                    if (events.hasOwnProperty(prop)) {
                        method = events[prop];

                        if (typeof method == "function") {
                            object[buster.functionName(method) || prop] = method;
                        } else {
                            method = object[events[prop]];
                        }

                        this.addListener(prop, method, object);
                    }
                }
            }

            return object;
        }
    };

    buster.eventEmitter.on = buster.eventEmitter.addListener;
}());

if (typeof module != "undefined") {
    module.exports = buster.eventEmitter;
}

var buster = buster || {};

if (typeof module === "object" && typeof require === "function") {
    buster = require("buster-core");
}

(function () {
    function indexOf(array, item) {
        if (array.indexOf) {
            return array.indexOf(item);
        }

        for (var i = 0, l = array.length; i < l; ++i) {
            if (array[i] == item) {
                return i;
            }
        }

        return -1;
    }

    function formatMessage(message) {
        if (!this.logFunctions && typeof message == "function") {
            return this.format(message());
        }
        return this.format(message);
    }

    function createLogger(name, level) {
        return function () {
            if (level > indexOf(this.levels, this.level)) {
                return;
            }

            var message = [];

            for (var i = 0, l = arguments.length; i < l; ++i) {
                message.push(formatMessage.call(this, arguments[i]));
            }

            this.emit("log", {
                message: message.join(" "),
                level: this.levels[level]
            });
        };
    }

    buster.eventedLogger = buster.extend(buster.create(buster.eventEmitter), {
        create: function (opt) {
            opt = opt || {};
            var logger = buster.create(this);
            logger.levels = opt.levels || ["error", "warn", "log", "debug"];
            logger.level = opt.level || logger.levels[logger.levels.length - 1];

            for (var i = 0, l = logger.levels.length; i < l; ++i) {
                logger[logger.levels[i]] = createLogger(logger.levels[i], i);
            }

            if (opt.formatter) { logger.format = opt.formatter; }
            logger.logFunctions = !!opt.logFunctions;
            return logger;
        },

        format: function (obj) {
            if (typeof obj != "object") {
                return "" + obj;
            }

            try {
                return JSON.stringify(obj);
            } catch (e) {
                return "" + obj;
            }
        }
    });
}());

if (typeof module != "undefined") {
    module.exports = buster.eventedLogger;
}

/*jslint eqeqeq: false, onevar: false, plusplus: false*/
/*global buster, require, module*/
(function () {
    var isCommonJS = typeof require == "function" && typeof module == "object";
    if (isCommonJS) buster = require("buster-core");
    var toString = Object.prototype.toString;
    var slice = Array.prototype.slice;
    var assert, refute, ba = buster.assertions = buster.eventEmitter.create();

    if (isCommonJS) {
        module.exports = buster.assertions;
    }

    function countAssertion() {
        if (typeof ba.count != "number") {
            ba.count = 0;
        }

        ba.count += 1;
    }

    ba.count = countAssertion;

    function assertEnoughArguments(name, args, num) {
        if (args.length < num) {
            ba.fail("[" + name + "] Expected to receive at least " +
                        num + " argument" + (num > 1 ? "s" : ""));
            return false;
        }

        return true;
    }

    function defineAssertion(type, name, func, fl, messageValues) {
        ba[type][name] = function () {
            var fullName = type + "." + name;
            countAssertion();
            if (!assertEnoughArguments(fullName, arguments, fl || func.length)) return;

            var failed = false;

            var ctx = {
                fail: function () {
                    failed = true;
                    var failArgs = [type, name].concat(slice.call(arguments));
                    fail.apply(this, failArgs);
                    return true;
                }
            };

            var args = slice.call(arguments, 0);

            if (typeof messageValues == "function") {
                args = messageValues.apply(this, args);
            }

            if (!func.apply(ctx, arguments)) {
                return fail.apply(ctx, [type, name, "message"].concat(args));
            }

            if (!failed) {
                ba.emit.apply(ba, ["pass", fullName].concat(args));
            }
        };
    }

    ba.add = function (name, options) {
        var refuteArgs;

        if (options.refute) {
            refuteArgs = options.refute.length;
        } else {
            refuteArgs = options.assert.length;
            options.refute = function () {
                return !options.assert.apply(this, arguments);
            };
        }

        var values = options && options.values; // TODO: Remove
        defineAssertion("assert", name, options.assert, options.assert.length, values);
        defineAssertion("refute", name, options.refute, refuteArgs, values);

        assert[name].message = options.assertMessage;
        refute[name].message = options.refuteMessage;

        if (options.expectation) {
            if (ba.expect && ba.expect.wrapAssertion) {
                ba.expect.wrapAssertion(name, options.expectation);
            } else {
                assert[name].expectationName = options.expectation;
                refute[name].expectationName = options.expectation;
            }
        }
    };

    function interpolate(string, property, value) {
        return string.replace(new RegExp("\\$\\{" + property + "\\}", "g"), value);
    }

    function interpolatePosArg(message, values) {
        var value;
        values = values || [];

        for (var i = 0, l = values.length; i < l; i++) {
            message = interpolate(message, i, ba.format(values[i]));
        }

        return message;
    }

    function interpolateProperties(msg, properties) {
        for (var prop in properties) {
            msg = interpolate(msg, prop, ba.format(properties[prop]));
        }

        return msg || "";
    }

    function fail(type, assertion, msg) {
        delete this.fail;
        var message = interpolateProperties(
            interpolatePosArg(ba[type][assertion][msg] || msg,
                              [].slice.call(arguments, 3)), this);
        ba.fail("[" + type + "." + assertion + "] " + message);
    }

    function isDate(value) {
        // Duck typed dates, allows objects to take on the role of dates
        // without actually being dates
        return typeof value.getTime == "function" &&
            value.getTime() == value.valueOf();
    }

    ba.isDate = isDate;

    // Fixes NaN === NaN (should be true) and
    // -0 === +0 (should be false)
    // http://wiki.ecmascript.org/doku.php?id=harmony:egal
    function egal(x, y) {
        if (x === y) {
            // 0 === -0, but they are not identical
            return x !== 0 || 1 / x === 1 / y;
        }
        
        // NaN !== NaN, but they are identical.
        // NaNs are the only non-reflexive value, i.e., if x !== x,
        // then x is a NaN.
        // isNaN is broken: it converts its argument to number, so
        // isNaN("foo") => true
        return x !== x && y !== y;
    }

    function areEqual(expected, actual) {
        if (egal(expected, actual)) {
            return true;
        }

        // Elements are only equal if expected === actual
        if (buster.isElement(expected) || buster.isElement(actual)) {
            return false;
        }

        // null and undefined only pass for null === null and
        // undefined === undefined
        /*jsl: ignore*/
        if (expected == null || actual == null) {
            return actual === expected;
        }
        /*jsl: end*/

        if (isDate(expected) || isDate(actual)) {
            return isDate(expected) && isDate(actual) &&
                expected.getTime() == actual.getTime();
        }

        var useCoercingEquality = typeof expected != "object" || typeof actual != "object";

        if (expected instanceof RegExp && actual instanceof RegExp) {
            if (expected.toString() != actual.toString()) {
                return false;
            }

            useCoercingEquality = false;
        }

        // Arrays can only be equal to arrays
        var expectedStr = toString.call(expected);
        var actualStr = toString.call(actual);

        // Coerce and compare when primitives are involved
        if (useCoercingEquality) {
            return expectedStr != "[object Array]" && actualStr != "[object Array]" &&
                expected == actual;
        }

        var expectedKeys = ba.keys(expected);
        var actualKeys = ba.keys(actual);

        if (isArguments(expected) || isArguments(actual)) {
            if (expected.length != actual.length) {
                return false;
            }
        } else {
            if (typeof expected != typeof actual || expectedStr != actualStr ||
                expectedKeys.length != actualKeys.length) {
                return false;
            }
        }

        var key;

        for (var i = 0, l = expectedKeys.length; i < l; i++) {
            key = expectedKeys[i];
            if (!Object.prototype.hasOwnProperty.call(actual, key) ||
                !areEqual(expected[key], actual[key])) {
                return false;
            }
        }

        return true;
    }

    ba.deepEqual = areEqual;

    assert = ba.assert = function assert(actual, message) {
        countAssertion();
        if (!assertEnoughArguments("assert", arguments, 1)) return;

        if (!actual) {
            var val = ba.format(actual);
            ba.fail(message || "[assert] Expected " + val + " to be truthy");
        } else {
            ba.emit("pass", "assert", message || "", actual);
        }
    };

    assert.toString = function () {
        return "buster.assert";
    };

    refute = ba.refute = function (actual, message) {
        countAssertion();
        if (!assertEnoughArguments("refute", arguments, 1)) return;

        if (actual) {
            var val = ba.format(actual);
            ba.fail(message || "[refute] Expected " + val + " to be falsy");
        } else {
            ba.emit("pass", "refute", message || "", actual);
        }
    };

    assert.message = "[assert] Expected ${0} to be truthy";
    ba.count = 0;

    ba.fail = function (message) {
        var exception = new Error(message);
        exception.name = "AssertionError";

        try {
            throw exception;
        } catch (e) {
            ba.emit("failure", e);
        }

        if (typeof ba.throwOnFailure != "boolean" || ba.throwOnFailure) {
            throw exception;
        }
    };

    ba.format = function (object) {
        return "" + object;
    };

    function msg(message) {
        if (!message) { return ""; }
        return message + (/[.:!?]$/.test(message) ? " " : ": ");
    }

    function actualAndExpectedMessageValues(actual, expected, message) {
        return [actual, expected, msg(message)]
    }

    function actualMessageValues(actual) {
        return [actual, msg(arguments[1])];
    }

    function actualAndTypeOfMessageValues(actual) {
        return [actual, typeof actual, msg(arguments[1])];
    }

    ba.add("same", {
        assert: function (actual, expected) {
            return egal(actual, expected);
        },
        refute: function (actual, expected) {
            return !egal(actual, expected);
        },
        assertMessage: "${2}${0} expected to be the same object as ${1}",
        refuteMessage: "${2}${0} expected not to be the same object as ${1}",
        expectation: "toBe",
        values: actualAndExpectedMessageValues
    });

    function multiLineStringDiff(actual, expected, message) {
        if (actual == expected) return true;

        var message = interpolatePosArg(assert.equals.multiLineStringHeading, [message]),
            actualLines = actual.split("\n"),
            expectedLines = expected.split("\n"),
            lineCount = Math.max(expectedLines.length, actualLines.length),
            lines = [];

        for (var i = 0; i < lineCount; ++i) {
            if (expectedLines[i] != actualLines[i]) {
                lines.push("line " + (i + 1) + ": " + (expectedLines[i] || "") +
                           "\nwas:    " + (actualLines[i] || ""));
            }
        }

        ba.fail("[assert.equals] " + message + lines.join("\n\n"));
        return false;
    }

    ba.add("equals", {
        assert: function (actual, expected) {
            if (typeof actual == "string" && typeof expected == "string" &&
                (actual.indexOf("\n") >= 0 || expected.indexOf("\n") >= 0)) {
                var message = msg(arguments[2]);
                return multiLineStringDiff.call(this, actual, expected, message);
            }

            return areEqual(actual, expected);
        },

        refute: function (actual, expected) {
            return !areEqual(actual, expected);
        },

        assertMessage: "${2}${0} expected to be equal to ${1}",
        refuteMessage: "${2}${0} expected not to be equal to ${1}",
        expectation: "toEqual",
        values: actualAndExpectedMessageValues
    });

    assert.equals.multiLineStringHeading = "${0}Expected multi-line strings to be equal:\n";

    ba.add("greater", {
        assert: function (actual, expected) {
            return actual > expected;
        },

        assertMessage: "${2}Expected ${0} to be greater than ${1}",
        refuteMessage: "${2}Expected ${0} to be less than or equal to ${1}",
        expectation: "toBeGreaterThan",
        values: actualAndExpectedMessageValues
    });

    ba.add("less", {
        assert: function (actual, expected) {
            return actual < expected;
        },

        assertMessage: "${2}Expected ${0} to be less than ${1}",
        refuteMessage: "${2}Expected ${0} to be greater than or equal to ${1}",
        expectation: "toBeLessThan",
        values: actualAndExpectedMessageValues
    });

    ba.add("defined", {
        assert: function (actual) {
            return typeof actual != "undefined";
        },
        assertMessage: "${2}Expected to be defined",
        refuteMessage: "${2}Expected ${0} (${1}) not to be defined",
        expectation: "toBeDefined",
        values: actualAndTypeOfMessageValues
    });

    ba.add("isNull", {
        assert: function (actual) {
            return actual === null;
        },
        assertMessage: "${1}Expected ${0} to be null",
        refuteMessage: "${1}Expected not to be null",
        expectation: "toBeNull",
        values: actualMessageValues
    });

    function match(object, matcher) {
        if (matcher && typeof matcher.test == "function") {
            return matcher.test(object);
        }

        if (typeof matcher == "function") {
            return matcher(object) === true;
        }

        if (typeof matcher == "string") {
            matcher = matcher.toLowerCase();
            var notNull = typeof object === "string" || !!object;
            return notNull && ("" + object).toLowerCase().indexOf(matcher) >= 0;
        }

        if (typeof matcher == "number") {
            return matcher == object;
        }

        if (typeof matcher == "boolean") {
            return matcher === object;
        }

        if (matcher && typeof matcher == "object") {
            for (var prop in matcher) {
                if (!match(object[prop], matcher[prop])) {
                    return false;
                }
            }

            return true;
        }

        throw new Error("Matcher (" + ba.format(matcher) + ") was not a " +
                        "string, a number, a function, a boolean or an object");
    }

    ba.match = match;

    ba.add("match", {
        assert: function (actual, matcher) {
            var passed;

            try {
                passed = match(actual, matcher);
            } catch (e) {
                return this.fail("exceptionMessage", e.message, msg(arguments[2]));
            }

            return passed;
        },

        refute: function (actual, matcher) {
            var passed;

            try {
                passed = match(actual, matcher);
            } catch (e) {
                return this.fail("exceptionMessage", e.message);
            }

            return !passed;
        },

        assertMessage: "${2}${0} expected to match ${1}",
        refuteMessage: "${2}${0} expected not to match ${1}",
        expectation: "toMatch",
        values: actualAndExpectedMessageValues
    });

    assert.match.exceptionMessage = "${1}${0}";
    refute.match.exceptionMessage = "${1}${0}";

    ba.add("isObject", {
        assert: function (actual) {
            return typeof actual == "object" && !!actual;
        },
        assertMessage: "${2}${0} (${1}) expected to be object and not null",
        refuteMessage: "${2}${0} expected to be null or not an object",
        expectation: "toBeObject",
        values: actualAndTypeOfMessageValues
    });

    ba.add("isFunction", {
        assert: function (actual) {
            return typeof actual == "function";
        },
        assertMessage: "${2}${0} (${1}) expected to be function",
        refuteMessage: "${2}${0} expected not to be function",
        expectation: "toBeFunction",
        values: function (actual) {
            return [("" + actual).replace("\n", ""), typeof actual, msg(arguments[1])];
        }
    });

    ba.add("isTrue", {
        assert: function (actual) {
            return actual === true;
        },
        assertMessage: "${1}Expected ${0} to be true",
        refuteMessage: "${1}Expected ${0} to not be true",
        expectation: "toBeTrue",
        values: actualMessageValues
    });

    ba.add("isFalse", {
        assert: function (actual) {
            return actual === false;
        },
        assertMessage: "${1}Expected ${0} to be false",
        refuteMessage: "${1}Expected ${0} to not be false",
        expectation: "toBeFalse",
        values: actualMessageValues
    });

    ba.add("isString", {
        assert: function (actual) {
            return typeof actual == "string";
        },
        assertMessage: "${2}Expected ${0} (${1}) to be string",
        refuteMessage: "${2}Expected ${0} not to be string",
        expectation: "toBeString",
        values: actualAndTypeOfMessageValues
    });

    ba.add("isBoolean", {
        assert: function (actual) {
            return typeof actual == "boolean";
        },
        assertMessage: "${2}Expected ${0} (${1}) to be boolean",
        refuteMessage: "${2}Expected ${0} not to be boolean",
        expectation: "toBeBoolean",
        values: actualAndTypeOfMessageValues
    });

    ba.add("isNumber", {
        assert: function (actual) {
            return typeof actual == "number" && !isNaN(actual);
        },
        assertMessage: "${2}Expected ${0} (${1}) to be a non-NaN number",
        refuteMessage: "${2}Expected ${0} to be NaN or another non-number value",
        expectation: "toBeNumber",
        values: actualAndTypeOfMessageValues
    });

    ba.add("isNaN", {
        assert: function (actual) {
            return typeof actual == "number" && isNaN(actual);
        },
        assertMessage: "${2}Expected ${0} to be NaN",
        refuteMessage: "${2}Expected not to be NaN",
        expectation: "toBeNaN",
        values: actualAndTypeOfMessageValues
    });

    ba.add("isArray", {
        assert: function (actual) {
            return toString.call(actual) == "[object Array]";
        },
        assertMessage: "${2}Expected ${0} to be array",
        refuteMessage: "${2}Expected ${0} not to be array",
        expectation: "toBeArray",
        values: actualAndTypeOfMessageValues
    });

    function isArrayLike(object) {
        return toString.call(object) == "[object Array]" ||
            (!!object && typeof object.length == "number" &&
            typeof object.splice == "function") ||
            ba.isArguments(object);
    }

    ba.isArrayLike = isArrayLike;

    ba.add("isArrayLike", {
        assert: function (actual) {
            return isArrayLike(actual);
        },
        assertMessage: "${2}Expected ${0} to be array like",
        refuteMessage: "${2}Expected ${0} not to be array like",
        expectation: "toBeArrayLike",
        values: actualAndTypeOfMessageValues
    });

    function captureException(callback) {
        try {
            callback();
        } catch (e) {
            return e;
        }

        return null;
    }

    ba.captureException = captureException;

    assert.exception = function (callback, exception, message) {
        countAssertion();
        if (!assertEnoughArguments("assert.exception", arguments, 1)) return

        if (!callback) {
            return;
        }

        var err = captureException(callback);
        message = msg(message);

        if (!err) {
            if (exception) {
                return fail.call({}, "assert", "exception", "typeNoExceptionMessage",
                                 message, exception);
            } else {
                return fail.call({}, "assert", "exception", "message",
                                 message, exception);
            }
        }

        if (exception && err.name != exception) {
            if (typeof window != "undefined" && typeof console != "undefined") {
                console.log(err);
            }

            return fail.call({}, "assert", "exception", "typeFailMessage",
                             message, exception, err.name, err.message);
        }

        ba.emit("pass", "assert.exception", message, callback, exception);
    };

    assert.exception.typeNoExceptionMessage = "${0}Expected ${1} but no exception was thrown";
    assert.exception.message = "${0}Expected exception";
    assert.exception.typeFailMessage = "${0}Expected ${1} but threw ${2} (${3})";
    assert.exception.expectationName = "toThrow";

    refute.exception = function (callback) {
        countAssertion();
        if (!assertEnoughArguments("refute.exception", arguments, 1)) return;

        var err = captureException(callback);

        if (err) {
            fail.call({}, "refute", "exception", "message",
                      msg(arguments[1]), err.name, err.message, callback);
        } else {
            ba.emit("pass", "refute.exception", callback);
        }
    };

    refute.exception.message = "${0}Expected not to throw but threw ${1} (${2})";
    refute.exception.expectationName = "toThrow";

    ba.add("near", {
        assert: function (actual, expected, delta) {
            return Math.abs(actual - expected) <= delta;
        },
        assertMessage: "${3}Expected ${0} to be equal to ${1} +/- ${2}",
        refuteMessage: "${3}Expected ${0} not to be equal to ${1} +/- ${2}",
        expectation: "toBeNear",
        values: function (actual, expected, delta, message) {
            return [actual, expected, delta, msg(message)];
        }
    });

    ba.add("hasPrototype", {
        assert: function (actual, protoObj) {
            return protoObj.isPrototypeOf(actual);
        },
        assertMessage: "${2}Expected ${0} to have ${1} on its prototype chain",
        refuteMessage: "${2}Expected ${0} not to have ${1} on its prototype chain",
        expectation: "toHavePrototype",
        values: actualAndExpectedMessageValues
    });

    ba.add("contains", {
        assert: function (haystack, needle) {
            for (var i = 0; i < haystack.length; i++) {
                if (haystack[i] === needle) {
                    return true;
                }
            }
            return false;
        },
        assertMessage: "${2}Expected [${0}] to contain ${1}",
        refuteMessage: "${2}Expected [${0}] not to contain ${1}",
        expectation: "toContain",
        values: actualAndExpectedMessageValues
    });

    ba.add("tagName", {
        assert: function (element, tagName) {
            if (!element.tagName) {
                return this.fail("noTagNameMessage", tagName, element, msg(arguments[2]));
            }

            return tagName.toLowerCase &&
                tagName.toLowerCase() == element.tagName.toLowerCase();
        },
        assertMessage: "${2}Expected tagName to be ${0} but was ${1}",
        refuteMessage: "${2}Expected tagName not to be ${0}",
        expectation: "toHaveTagName",
        values: function (element, tagName, message) {
            return [tagName, element.tagName, msg(message)];
        }
    });

    assert.tagName.noTagNameMessage = "${2}Expected ${1} to have tagName property";
    refute.tagName.noTagNameMessage = "${2}Expected ${1} to have tagName property";

    function indexOf(arr, item) {
        for (var i = 0, l = arr.length; i < l; i++) {
            if (arr[i] == item) {
                return i;
            }
        }

        return -1;
    }

    ba.add("className", {
        assert: function (element, className) {
            if (typeof element.className == "undefined") {
                return this.fail("noClassNameMessage", className, element, msg(arguments[2]));
            }

            var expected = typeof className == "string" ? className.split(" ") : className;
            var actual = element.className.split(" ");

            for (var i = 0, l = expected.length; i < l; i++) {
                if (indexOf(actual, expected[i]) < 0) {
                    return false;
                }
            }

            return true;
        },
        assertMessage: "${2}Expected object's className to include ${0} but was ${1}",
        refuteMessage: "${2}Expected object's className not to include ${0}",
        expectation: "toHaveClassName",
        values: function (element, className, message) {
            return [className, element.className, msg(message)];
        }
    });

    assert.className.noClassNameMessage = "${2}Expected object to have className property";
    refute.className.noClassNameMessage = "${2}Expected object to have className property";

    if (typeof module != "undefined") {
        ba.expect = function () {
            ba.expect = require("./buster-assertions/expect");
            return ba.expect.apply(exports, arguments);
        };
    }

    function isArguments(obj) {
        if (typeof obj != "object" || typeof obj.length != "number" ||
            toString.call(obj) == "[object Array]") {
            return false;
        }

        if (typeof obj.callee == "function") {
            return true;
        }

        try {
            obj[obj.length] = 6;
            delete obj[obj.length];
        } catch (e) {
            return true;
        }

        return false;
    }

    ba.isArguments = isArguments;

    ba.keys = function (object) {
        var keys = [];

        for (var prop in object) {
            if (Object.prototype.hasOwnProperty.call(object, prop)) {
                keys.push(prop);
            }
        }

        return keys;
    };
}());

if (typeof module == "object" && typeof require == "function") {
    var buster = require("buster-core");
    buster.assertions = require("../buster-assertions");
}

(function (ba) {
    ba.expectation = {};

    ba.expect = function (actual) {
        var expectation = buster.extend(buster.create(ba.expectation), {
            actual: actual,
            assertMode: true
        });
        expectation.not = buster.create(expectation);
        expectation.not.assertMode = false;
        return expectation;
    };

    ba.expect.wrapAssertion = function (assertion, expectation) {
        ba.expectation[expectation] = function () {
            var args = [this.actual].concat(Array.prototype.slice.call(arguments));
            var type = this.assertMode ? "assert" : "refute";
            var callFunc;

            if (assertion === "assert") {
                callFunc = this.assertMode ? ba.assert : ba.refute;
            } else if (assertion === "refute") {
                callFunc = this.assertMode ? ba.refute : ba.assert;
            } else {
                callFunc = ba[type][assertion];
            }

            try {
                return callFunc.apply(ba.expect, args);
            } catch (e) {
                e.message = (e.message || "").replace(
                    "[" + type + "." + assertion + "]",
                    "[expect." + (this.assertMode ? "" : "not.") + expectation + "]");
                throw e;
            }
        };
    };

    var prop, expectationName;

    for (prop in ba.assert) {
        if (ba.assert[prop].expectationName) {
            expectationName = ba.assert[prop].expectationName;
            ba.expect.wrapAssertion(prop, expectationName);
        }
    }

    ba.expect.wrapAssertion("assert", "toBeTruthy");
    ba.expect.wrapAssertion("refute", "toBeFalsy");

    if (ba.expectation.toBeNear) {
        ba.expectation.toBeCloseTo = ba.expectation.toBeNear;
    }

    if (typeof module == "object") {
        module.exports = ba.expect;
    }
}(buster.assertions));

if (typeof buster === "undefined") {
    var buster = {};
}

if (typeof module === "object" && typeof require === "function") {
    buster = require("buster-core");
}

buster.format = buster.format || {};
buster.format.excludeConstructors = ["Object", /^.$/];
buster.format.quoteStrings = true;

buster.format.ascii = (function () {
    "use strict";

    var hasOwn = Object.prototype.hasOwnProperty;

    var specialObjects = [];
    if (typeof global != "undefined") {
        specialObjects.push({ obj: global, value: "[object global]" });
    }
    if (typeof document != "undefined") {
        specialObjects.push({ obj: document, value: "[object HTMLDocument]" });
    }
    if (typeof window != "undefined") {
        specialObjects.push({ obj: window, value: "[object Window]" });
    }

    function keys(object) {
        var k = Object.keys && Object.keys(object) || [];

        if (k.length == 0) {
            for (var prop in object) {
                if (hasOwn.call(object, prop)) {
                    k.push(prop);
                }
            }
        }

        return k.sort();
    }

    function isCircular(object, objects) {
        if (typeof object != "object") {
            return false;
        }

        for (var i = 0, l = objects.length; i < l; ++i) {
            if (objects[i] === object) {
                return true;
            }
        }

        return false;
    }

    function ascii(object, processed, indent) {
        if (typeof object == "string") {
            var quote = typeof this.quoteStrings != "boolean" || this.quoteStrings;
            return processed || quote ? '"' + object + '"' : object;
        }

        if (typeof object == "function" && !(object instanceof RegExp)) {
            return ascii.func(object);
        }

        processed = processed || [];

        if (isCircular(object, processed)) {
            return "[Circular]";
        }

        if (Object.prototype.toString.call(object) == "[object Array]") {
            return ascii.array.call(this, object, processed);
        }

        if (!object) {
            return "" + object;
        }

        if (buster.isElement(object)) {
            return ascii.element(object);
        }

        if (typeof object.toString == "function" &&
            object.toString !== Object.prototype.toString) {
            return object.toString();
        }

        for (var i = 0, l = specialObjects.length; i < l; i++) {
            if (object === specialObjects[i].obj) {
                return specialObjects[i].value;
            }
        }

        return ascii.object.call(this, object, processed, indent);
    }

    ascii.func = function (func) {
        return "function " + buster.functionName(func) + "() {}";
    };

    ascii.array = function (array, processed) {
        processed = processed || [];
        processed.push(array);
        var pieces = [];

        for (var i = 0, l = array.length; i < l; ++i) {
            pieces.push(ascii.call(this, array[i], processed));
        }

        return "[" + pieces.join(", ") + "]";
    };

    ascii.object = function (object, processed, indent) {
        processed = processed || [];
        processed.push(object);
        indent = indent || 0;
        var pieces = [], properties = keys(object), prop, str, obj;
        var is = "";
        var length = 3;

        for (var i = 0, l = indent; i < l; ++i) {
            is += " ";
        }

        for (i = 0, l = properties.length; i < l; ++i) {
            prop = properties[i];
            obj = object[prop];

            if (isCircular(obj, processed)) {
                str = "[Circular]";
            } else {
                str = ascii.call(this, obj, processed, indent + 2);
            }

            str = (/\s/.test(prop) ? '"' + prop + '"' : prop) + ": " + str;
            length += str.length;
            pieces.push(str);
        }

        var cons = ascii.constructorName.call(this, object);
        var prefix = cons ? "[" + cons + "] " : ""

        return (length + indent) > 80 ?
            prefix + "{\n  " + is + pieces.join(",\n  " + is) + "\n" + is + "}" :
            prefix + "{ " + pieces.join(", ") + " }";
    };

    ascii.element = function (element) {
        var tagName = element.tagName.toLowerCase();
        var attrs = element.attributes, attribute, pairs = [], attrName;

        for (var i = 0, l = attrs.length; i < l; ++i) {
            attribute = attrs.item(i);
            attrName = attribute.nodeName.toLowerCase().replace("html:", "");

            if (attrName == "contenteditable" && attribute.nodeValue == "inherit") {
                continue;
            }

            if (!!attribute.nodeValue) {
                pairs.push(attrName + "=\"" + attribute.nodeValue + "\"");
            }
        }

        var formatted = "<" + tagName + (pairs.length > 0 ? " " : "");
        var content = element.innerHTML;

        if (content.length > 20) {
            content = content.substr(0, 20) + "[...]";
        }

        var res = formatted + pairs.join(" ") + ">" + content + "</" + tagName + ">";

        return res.replace(/ contentEditable="inherit"/, "");
    };

    ascii.constructorName = function (object) {
        var name = buster.functionName(object && object.constructor);
        var excludes = this.excludeConstructors || buster.format.excludeConstructors || [];

        for (var i = 0, l = excludes.length; i < l; ++i) {
            if (typeof excludes[i] == "string" && excludes[i] == name) {
                return "";
            } else if (excludes[i].test && excludes[i].test(name)) {
                return "";
            }
        }

        return name;
    };

    return ascii;
}());

if (typeof module != "undefined") {
    module.exports = buster.format;
}

/*jslint eqeqeq: false, onevar: false, forin: true, nomen: false, regexp: false, plusplus: false*/
/*global module, require, __dirname, document*/
/**
 * Sinon core utilities. For internal use only.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

var sinon = (function (buster) {
    var div = typeof document != "undefined" && document.createElement("div");
    var hasOwn = Object.prototype.hasOwnProperty;

    function isDOMNode(obj) {
        var success = false;

        try {
            obj.appendChild(div);
            success = div.parentNode == obj;
        } catch (e) {
            return false;
        } finally {
            try {
                obj.removeChild(div);
            } catch (e) {
                // Remove failed, not much we can do about that
            }
        }

        return success;
    }

    function isElement(obj) {
        return div && obj && obj.nodeType === 1 && isDOMNode(obj);
    }

    function isFunction(obj) {
        return !!(obj && obj.constructor && obj.call && obj.apply);
    }

    function mirrorProperties(target, source) {
        for (var prop in source) {
            if (!hasOwn.call(target, prop)) {
                target[prop] = source[prop];
            }
        }
    }

    var sinon = {
        wrapMethod: function wrapMethod(object, property, method) {
            if (!object) {
                throw new TypeError("Should wrap property of object");
            }

            if (typeof method != "function") {
                throw new TypeError("Method wrapper should be function");
            }

            var wrappedMethod = object[property];

            if (!isFunction(wrappedMethod)) {
                throw new TypeError("Attempted to wrap " + (typeof wrappedMethod) + " property " +
                                    property + " as function");
            }

            if (wrappedMethod.restore && wrappedMethod.restore.sinon) {
                throw new TypeError("Attempted to wrap " + property + " which is already wrapped");
            }

            if (wrappedMethod.calledBefore) {
                var verb = !!wrappedMethod.returns ? "stubbed" : "spied on";
                throw new TypeError("Attempted to wrap " + property + " which is already " + verb);
            }

            // IE 8 does not support hasOwnProperty on the window object.
            var owned = hasOwn.call(object, property);
            object[property] = method;
            method.displayName = property;

            method.restore = function () {
                // For prototype properties try to reset by delete first.
                // If this fails (ex: localStorage on mobile safari) then force a reset
                // via direct assignment.
                if (!owned) {
                    delete object[property];
                }
                if (object[property] === method) {
                    object[property] = wrappedMethod;
                }
            };

            method.restore.sinon = true;
            mirrorProperties(method, wrappedMethod);

            return method;
        },

        extend: function extend(target) {
            for (var i = 1, l = arguments.length; i < l; i += 1) {
                for (var prop in arguments[i]) {
                    if (arguments[i].hasOwnProperty(prop)) {
                        target[prop] = arguments[i][prop];
                    }

                    // DONT ENUM bug, only care about toString
                    if (arguments[i].hasOwnProperty("toString") &&
                        arguments[i].toString != target.toString) {
                        target.toString = arguments[i].toString;
                    }
                }
            }

            return target;
        },

        create: function create(proto) {
            var F = function () {};
            F.prototype = proto;
            return new F();
        },

        deepEqual: function deepEqual(a, b) {
            if (sinon.match && sinon.match.isMatcher(a)) {
                return a.test(b);
            }
            if (typeof a != "object" || typeof b != "object") {
                return a === b;
            }

            if (isElement(a) || isElement(b)) {
                return a === b;
            }

            if (a === b) {
                return true;
            }

            var aString = Object.prototype.toString.call(a);
            if (aString != Object.prototype.toString.call(b)) {
                return false;
            }

            if (aString == "[object Array]") {
                if (a.length !== b.length) {
                    return false;
                }

                for (var i = 0, l = a.length; i < l; i += 1) {
                    if (!deepEqual(a[i], b[i])) {
                        return false;
                    }
                }

                return true;
            }

            var prop, aLength = 0, bLength = 0;

            for (prop in a) {
                aLength += 1;

                if (!deepEqual(a[prop], b[prop])) {
                    return false;
                }
            }

            for (prop in b) {
                bLength += 1;
            }

            if (aLength != bLength) {
                return false;
            }

            return true;
        },

        functionName: function functionName(func) {
            var name = func.displayName || func.name;

            // Use function decomposition as a last resort to get function
            // name. Does not rely on function decomposition to work - if it
            // doesn't debugging will be slightly less informative
            // (i.e. toString will say 'spy' rather than 'myFunc').
            if (!name) {
                var matches = func.toString().match(/function ([^\s\(]+)/);
                name = matches && matches[1];
            }

            return name;
        },

        functionToString: function toString() {
            if (this.getCall && this.callCount) {
                var thisValue, prop, i = this.callCount;

                while (i--) {
                    thisValue = this.getCall(i).thisValue;

                    for (prop in thisValue) {
                        if (thisValue[prop] === this) {
                            return prop;
                        }
                    }
                }
            }

            return this.displayName || "sinon fake";
        },

        getConfig: function (custom) {
            var config = {};
            custom = custom || {};
            var defaults = sinon.defaultConfig;

            for (var prop in defaults) {
                if (defaults.hasOwnProperty(prop)) {
                    config[prop] = custom.hasOwnProperty(prop) ? custom[prop] : defaults[prop];
                }
            }

            return config;
        },

        format: function (val) {
            return "" + val;
        },

        defaultConfig: {
            injectIntoThis: true,
            injectInto: null,
            properties: ["spy", "stub", "mock", "clock", "server", "requests"],
            useFakeTimers: true,
            useFakeServer: true
        },

        timesInWords: function timesInWords(count) {
            return count == 1 && "once" ||
                count == 2 && "twice" ||
                count == 3 && "thrice" ||
                (count || 0) + " times";
        },

        calledInOrder: function (spies) {
            for (var i = 1, l = spies.length; i < l; i++) {
                if (!spies[i - 1].calledBefore(spies[i])) {
                    return false;
                }
            }

            return true;
        },

        orderByFirstCall: function (spies) {
            return spies.sort(function (a, b) {
                // uuid, won't ever be equal
                var aCall = a.getCall(0);
                var bCall = b.getCall(0);
                var aId = aCall && aCall.callId || -1;
                var bId = bCall && bCall.callId || -1;

                return aId < bId ? -1 : 1;
            });
        },

        log: function () {},

        logError: function (label, err) {
            var msg = label + " threw exception: "
            sinon.log(msg + "[" + err.name + "] " + err.message);
            if (err.stack) { sinon.log(err.stack); }

            setTimeout(function () {
                err.message = msg + err.message;
                throw err;
            }, 0);
        },

        typeOf: function (value) {
            if (value === null) {
              return "null";
            }
            var string = Object.prototype.toString.call(value);
            return string.substring(8, string.length - 1).toLowerCase();
        }
    };

    var isNode = typeof module == "object" && typeof require == "function";

    if (isNode) {
        try {
            buster = { format: require("buster-format") };
        } catch (e) {}
        module.exports = sinon;
        module.exports.spy = require("./sinon/spy");
        module.exports.stub = require("./sinon/stub");
        module.exports.mock = require("./sinon/mock");
        module.exports.collection = require("./sinon/collection");
        module.exports.assert = require("./sinon/assert");
        module.exports.sandbox = require("./sinon/sandbox");
        module.exports.test = require("./sinon/test");
        module.exports.testCase = require("./sinon/test_case");
        module.exports.assert = require("./sinon/assert");
        module.exports.match = require("./sinon/match");
    }

    if (buster) {
        var formatter = sinon.create(buster.format);
        formatter.quoteStrings = false;
        sinon.format = function () {
            return formatter.ascii.apply(formatter, arguments);
        };
    } else if (isNode) {
        try {
            var util = require("util");
            sinon.format = function (value) {
                return typeof value == "object" && value.toString === Object.prototype.toString ? util.inspect(value) : value;
            };
        } catch (e) {
            /* Node, but no util module - would be very old, but better safe than
             sorry */
        }
    }

    return sinon;
}(typeof buster == "object" && buster));

/**
 * @depend ../sinon.js
 * @depend match.js
 */
/*jslint eqeqeq: false, onevar: false, plusplus: false*/
/*global module, require, sinon*/
/**
 * Spy functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon) {
    var commonJSModule = typeof module == "object" && typeof require == "function";
    var spyCall;
    var callId = 0;
    var push = [].push;
    var slice = Array.prototype.slice;

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon) {
        return;
    }

    function spy(object, property) {
        if (!property && typeof object == "function") {
            return spy.create(object);
        }

        if (!object && !property) {
            return spy.create(function () {});
        }

        var method = object[property];
        return sinon.wrapMethod(object, property, spy.create(method));
    }

    sinon.extend(spy, (function () {

        function delegateToCalls(api, method, matchAny, actual, notCalled) {
            api[method] = function () {
                if (!this.called) {
                    if (notCalled) {
                        return notCalled.apply(this, arguments);
                    }
                    return false;
                }

                var currentCall;
                var matches = 0;

                for (var i = 0, l = this.callCount; i < l; i += 1) {
                    currentCall = this.getCall(i);

                    if (currentCall[actual || method].apply(currentCall, arguments)) {
                        matches += 1;

                        if (matchAny) {
                            return true;
                        }
                    }
                }

                return matches === this.callCount;
            };
        }

        function matchingFake(fakes, args, strict) {
            if (!fakes) {
                return;
            }

            var alen = args.length;

            for (var i = 0, l = fakes.length; i < l; i++) {
                if (fakes[i].matches(args, strict)) {
                    return fakes[i];
                }
            }
        }

        function incrementCallCount() {
            this.called = true;
            this.callCount += 1;
            this.notCalled = false;
            this.calledOnce = this.callCount == 1;
            this.calledTwice = this.callCount == 2;
            this.calledThrice = this.callCount == 3;
        }

        function createCallProperties() {
            this.firstCall = this.getCall(0);
            this.secondCall = this.getCall(1);
            this.thirdCall = this.getCall(2);
            this.lastCall = this.getCall(this.callCount - 1);
        }

        var uuid = 0;

        // Public API
        var spyApi = {
            reset: function () {
                this.called = false;
                this.notCalled = true;
                this.calledOnce = false;
                this.calledTwice = false;
                this.calledThrice = false;
                this.callCount = 0;
                this.firstCall = null;
                this.secondCall = null;
                this.thirdCall = null;
                this.lastCall = null;
                this.args = [];
                this.returnValues = [];
                this.thisValues = [];
                this.exceptions = [];
                this.callIds = [];
                if (this.fakes) {
                    for (var i = 0; i < this.fakes.length; i++) {
                        this.fakes[i].reset();
                    }
                }
            },

            create: function create(func) {
                var name;

                if (typeof func != "function") {
                    func = function () {};
                } else {
                    name = sinon.functionName(func);
                }

                function proxy() {
                    return proxy.invoke(func, this, slice.call(arguments));
                }

                sinon.extend(proxy, spy);
                delete proxy.create;
                sinon.extend(proxy, func);

                proxy.reset();
                proxy.prototype = func.prototype;
                proxy.displayName = name || "spy";
                proxy.toString = sinon.functionToString;
                proxy._create = sinon.spy.create;
                proxy.id = "spy#" + uuid++;

                return proxy;
            },

            invoke: function invoke(func, thisValue, args) {
                var matching = matchingFake(this.fakes, args);
                var exception, returnValue;

                incrementCallCount.call(this);
                push.call(this.thisValues, thisValue);
                push.call(this.args, args);
                push.call(this.callIds, callId++);

                try {
                    if (matching) {
                        returnValue = matching.invoke(func, thisValue, args);
                    } else {
                        returnValue = (this.func || func).apply(thisValue, args);
                    }
                } catch (e) {
                    push.call(this.returnValues, undefined);
                    exception = e;
                    throw e;
                } finally {
                    push.call(this.exceptions, exception);
                }

                push.call(this.returnValues, returnValue);

                createCallProperties.call(this);

                return returnValue;
            },

            getCall: function getCall(i) {
                if (i < 0 || i >= this.callCount) {
                    return null;
                }

                return spyCall.create(this, this.thisValues[i], this.args[i],
                                      this.returnValues[i], this.exceptions[i],
                                      this.callIds[i]);
            },

            calledBefore: function calledBefore(spyFn) {
                if (!this.called) {
                    return false;
                }

                if (!spyFn.called) {
                    return true;
                }

                return this.callIds[0] < spyFn.callIds[spyFn.callIds.length - 1];
            },

            calledAfter: function calledAfter(spyFn) {
                if (!this.called || !spyFn.called) {
                    return false;
                }

                return this.callIds[this.callCount - 1] > spyFn.callIds[spyFn.callCount - 1];
            },

            withArgs: function () {
                var args = slice.call(arguments);

                if (this.fakes) {
                    var match = matchingFake(this.fakes, args, true);

                    if (match) {
                        return match;
                    }
                } else {
                    this.fakes = [];
                }

                var original = this;
                var fake = this._create();
                fake.matchingAguments = args;
                push.call(this.fakes, fake);

                fake.withArgs = function () {
                    return original.withArgs.apply(original, arguments);
                };

                for (var i = 0; i < this.args.length; i++) {
                    if (fake.matches(this.args[i])) {
                        incrementCallCount.call(fake);
                        push.call(fake.thisValues, this.thisValues[i]);
                        push.call(fake.args, this.args[i]);
                        push.call(fake.returnValues, this.returnValues[i]);
                        push.call(fake.exceptions, this.exceptions[i]);
                        push.call(fake.callIds, this.callIds[i]);
                    }
                }
                createCallProperties.call(fake);

                return fake;
            },

            matches: function (args, strict) {
                var margs = this.matchingAguments;

                if (margs.length <= args.length &&
                    sinon.deepEqual(margs, args.slice(0, margs.length))) {
                    return !strict || margs.length == args.length;
                }
            },

            printf: function (format) {
                var spy = this;
                var args = slice.call(arguments, 1);
                var formatter;

                return (format || "").replace(/%(.)/g, function (match, specifyer) {
                    formatter = spyApi.formatters[specifyer];

                    if (typeof formatter == "function") {
                        return formatter.call(null, spy, args);
                    } else if (!isNaN(parseInt(specifyer), 10)) {
                        return sinon.format(args[specifyer - 1]);
                    }

                    return "%" + specifyer;
                });
            }
        };

        delegateToCalls(spyApi, "calledOn", true);
        delegateToCalls(spyApi, "alwaysCalledOn", false, "calledOn");
        delegateToCalls(spyApi, "calledWith", true);
        delegateToCalls(spyApi, "calledWithMatch", true);
        delegateToCalls(spyApi, "alwaysCalledWith", false, "calledWith");
        delegateToCalls(spyApi, "alwaysCalledWithMatch", false, "calledWithMatch");
        delegateToCalls(spyApi, "calledWithExactly", true);
        delegateToCalls(spyApi, "alwaysCalledWithExactly", false, "calledWithExactly");
        delegateToCalls(spyApi, "neverCalledWith", false, "notCalledWith",
            function () { return true; });
        delegateToCalls(spyApi, "neverCalledWithMatch", false, "notCalledWithMatch",
            function () { return true; });
        delegateToCalls(spyApi, "threw", true);
        delegateToCalls(spyApi, "alwaysThrew", false, "threw");
        delegateToCalls(spyApi, "returned", true);
        delegateToCalls(spyApi, "alwaysReturned", false, "returned");
        delegateToCalls(spyApi, "calledWithNew", true);
        delegateToCalls(spyApi, "alwaysCalledWithNew", false, "calledWithNew");
        delegateToCalls(spyApi, "callArg", false, "callArgWith", function () {
            throw new Error(this.toString() + " cannot call arg since it was not yet invoked.");
        });
        spyApi.callArgWith = spyApi.callArg;
        delegateToCalls(spyApi, "yield", false, "yield", function () {
            throw new Error(this.toString() + " cannot yield since it was not yet invoked.");
        });
        // "invokeCallback" is an alias for "yield" since "yield" is invalid in strict mode.
        spyApi.invokeCallback = spyApi.yield;
        delegateToCalls(spyApi, "yieldTo", false, "yieldTo", function (property) {
            throw new Error(this.toString() + " cannot yield to '" + property +
                "' since it was not yet invoked.");
        });

        spyApi.formatters = {
            "c": function (spy) {
                return sinon.timesInWords(spy.callCount);
            },

            "n": function (spy) {
                return spy.toString();
            },

            "C": function (spy) {
                var calls = [];

                for (var i = 0, l = spy.callCount; i < l; ++i) {
                    push.call(calls, "    " + spy.getCall(i).toString());
                }

                return calls.length > 0 ? "\n" + calls.join("\n") : "";
            },

            "t": function (spy) {
                var objects = [];

                for (var i = 0, l = spy.callCount; i < l; ++i) {
                    push.call(objects, sinon.format(spy.thisValues[i]));
                }

                return objects.join(", ");
            },

            "*": function (spy, args) {
                var formatted = [];

                for (var i = 0, l = args.length; i < l; ++i) {
                    push.call(formatted, sinon.format(args[i]));
                }

                return formatted.join(", ");
            }
        };

        return spyApi;
    }()));

    spyCall = (function () {

        function throwYieldError(proxy, text, args) {
            var msg = sinon.functionName(proxy) + text;
            if (args.length) {
                msg += " Received [" + slice.call(args).join(", ") + "]";
            }
            throw new Error(msg);
        }

        return {
            create: function create(spy, thisValue, args, returnValue, exception, id) {
                var proxyCall = sinon.create(spyCall);
                delete proxyCall.create;
                proxyCall.proxy = spy;
                proxyCall.thisValue = thisValue;
                proxyCall.args = args;
                proxyCall.returnValue = returnValue;
                proxyCall.exception = exception;
                proxyCall.callId = typeof id == "number" && id || callId++;

                return proxyCall;
            },

            calledOn: function calledOn(thisValue) {
                return this.thisValue === thisValue;
            },

            calledWith: function calledWith() {
                for (var i = 0, l = arguments.length; i < l; i += 1) {
                    if (!sinon.deepEqual(arguments[i], this.args[i])) {
                        return false;
                    }
                }

                return true;
            },

            calledWithMatch: function calledWithMatch() {
              for (var i = 0, l = arguments.length; i < l; i += 1) {
                  var actual = this.args[i];
                  var expectation = arguments[i];
                  if (!sinon.match || !sinon.match(expectation).test(actual)) {
                      return false;
                  }
              }
              return true;
            },

            calledWithExactly: function calledWithExactly() {
                return arguments.length == this.args.length &&
                    this.calledWith.apply(this, arguments);
            },

            notCalledWith: function notCalledWith() {
                return !this.calledWith.apply(this, arguments);
            },

            notCalledWithMatch: function notCalledWithMatch() {
              return !this.calledWithMatch.apply(this, arguments);
            },

            returned: function returned(value) {
                return sinon.deepEqual(value, this.returnValue);
            },

            threw: function threw(error) {
                if (typeof error == "undefined" || !this.exception) {
                    return !!this.exception;
                }

                if (typeof error == "string") {
                    return this.exception.name == error;
                }

                return this.exception === error;
            },

            calledWithNew: function calledWithNew(thisValue) {
                return this.thisValue instanceof this.proxy;
            },

            calledBefore: function (other) {
                return this.callId < other.callId;
            },

            calledAfter: function (other) {
                return this.callId > other.callId;
            },

            callArg: function (pos) {
                this.args[pos]();
            },

            callArgWith: function (pos) {
                var args = slice.call(arguments, 1);
                this.args[pos].apply(null, args);
            },

            "yield": function () {
                var args = this.args;
                for (var i = 0, l = args.length; i < l; ++i) {
                    if (typeof args[i] === "function") {
                        args[i].apply(null, slice.call(arguments));
                        return;
                    }
                }
                throwYieldError(this.proxy, " cannot yield since no callback was passed.", args);
            },

            yieldTo: function (prop) {
                var args = this.args;
                for (var i = 0, l = args.length; i < l; ++i) {
                    if (args[i] && typeof args[i][prop] === "function") {
                        args[i][prop].apply(null, slice.call(arguments, 1));
                        return;
                    }
                }
                throwYieldError(this.proxy, " cannot yield to '" + prop +
                    "' since no callback was passed.", args);
            },

            toString: function () {
                var callStr = this.proxy.toString() + "(";
                var args = [];

                for (var i = 0, l = this.args.length; i < l; ++i) {
                    push.call(args, sinon.format(this.args[i]));
                }

                callStr = callStr + args.join(", ") + ")";

                if (typeof this.returnValue != "undefined") {
                    callStr += " => " + sinon.format(this.returnValue);
                }

                if (this.exception) {
                    callStr += " !" + this.exception.name;

                    if (this.exception.message) {
                        callStr += "(" + this.exception.message + ")";
                    }
                }

                return callStr;
            }
        };
    }());

    spy.spyCall = spyCall;

    // This steps outside the module sandbox and will be removed
    sinon.spyCall = spyCall;

    if (commonJSModule) {
        module.exports = spy;
    } else {
        sinon.spy = spy;
    }
}(typeof sinon == "object" && sinon || null));

/**
 * @depend ../sinon.js
 * @depend spy.js
 */
/*jslint eqeqeq: false, onevar: false*/
/*global module, require, sinon*/
/**
 * Stub functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon) {
    var commonJSModule = typeof module == "object" && typeof require == "function";

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon) {
        return;
    }

    function stub(object, property, func) {
        if (!!func && typeof func != "function") {
            throw new TypeError("Custom stub should be function");
        }

        var wrapper;

        if (func) {
            wrapper = sinon.spy && sinon.spy.create ? sinon.spy.create(func) : func;
        } else {
            wrapper = stub.create();
        }

        if (!object && !property) {
            return sinon.stub.create();
        }

        if (!property && !!object && typeof object == "object") {
            for (var prop in object) {
                if (typeof object[prop] === "function") {
                    stub(object, prop);
                }
            }

            return object;
        }

        return sinon.wrapMethod(object, property, wrapper);
    }

    function getCallback(stub, args) {
        if (stub.callArgAt < 0) {
            for (var i = 0, l = args.length; i < l; ++i) {
                if (!stub.callArgProp && typeof args[i] == "function") {
                    return args[i];
                }

                if (stub.callArgProp && args[i] &&
                    typeof args[i][stub.callArgProp] == "function") {
                    return args[i][stub.callArgProp];
                }
            }

            return null;
        }

        return args[stub.callArgAt];
    }

    var join = Array.prototype.join;

    function getCallbackError(stub, func, args) {
        if (stub.callArgAt < 0) {
            var msg;

            if (stub.callArgProp) {
                msg = sinon.functionName(stub) +
                    " expected to yield to '" + stub.callArgProp +
                    "', but no object with such a property was passed."
            } else {
                msg = sinon.functionName(stub) +
                            " expected to yield, but no callback was passed."
            }

            if (args.length > 0) {
                msg += " Received [" + join.call(args, ", ") + "]";
            }

            return msg;
        }

        return "argument at index " + stub.callArgAt + " is not a function: " + func;
    }

    var nextTick = (function () {
        if (typeof process === "object" && typeof process.nextTick === "function") {
            return process.nextTick;
        } else if (typeof msSetImmediate === "function") {
            return msSetImmediate.bind(window);
        } else if (typeof setImmediate === "function") {
            return setImmediate;
        } else {
            return function (callback) {
                setTimeout(callback, 0);
            };
        }
    })();

    function callCallback(stub, args) {
        if (typeof stub.callArgAt == "number") {
            var func = getCallback(stub, args);

            if (typeof func != "function") {
                throw new TypeError(getCallbackError(stub, func, args));
            }

            if (stub.callbackAsync) {
                nextTick(function() {
                    func.apply(stub.callbackContext, stub.callbackArguments);
                });
            } else {
                func.apply(stub.callbackContext, stub.callbackArguments);
            }
        }
    }

    var uuid = 0;

    sinon.extend(stub, (function () {
        var slice = Array.prototype.slice, proto;

        function throwsException(error, message) {
            if (typeof error == "string") {
                this.exception = new Error(message || "");
                this.exception.name = error;
            } else if (!error) {
                this.exception = new Error("Error");
            } else {
                this.exception = error;
            }

            return this;
        }

        proto = {
            create: function create() {
                var functionStub = function () {

                    callCallback(functionStub, arguments);

                    if (functionStub.exception) {
                        throw functionStub.exception;
                    } else if (typeof functionStub.returnArgAt == 'number') {
                        return arguments[functionStub.returnArgAt];
                    } else if (functionStub.returnThis) {
                        return this;
                    }
                    return functionStub.returnValue;
                };

                functionStub.id = "stub#" + uuid++;
                var orig = functionStub;
                functionStub = sinon.spy.create(functionStub);
                functionStub.func = orig;

                sinon.extend(functionStub, stub);
                functionStub._create = sinon.stub.create;
                functionStub.displayName = "stub";
                functionStub.toString = sinon.functionToString;

                return functionStub;
            },

            returns: function returns(value) {
                this.returnValue = value;

                return this;
            },

            returnsArg: function returnsArg(pos) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }

                this.returnArgAt = pos;

                return this;
            },

            returnsThis: function returnsThis() {
                this.returnThis = true;

                return this;
            },

            "throws": throwsException,
            throwsException: throwsException,

            callsArg: function callsArg(pos) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }

                this.callArgAt = pos;
                this.callbackArguments = [];

                return this;
            },

            callsArgOn: function callsArgOn(pos, context) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = pos;
                this.callbackArguments = [];
                this.callbackContext = context;

                return this;
            },

            callsArgWith: function callsArgWith(pos) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }

                this.callArgAt = pos;
                this.callbackArguments = slice.call(arguments, 1);

                return this;
            },

            callsArgOnWith: function callsArgWith(pos, context) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = pos;
                this.callbackArguments = slice.call(arguments, 2);
                this.callbackContext = context;

                return this;
            },

            yields: function () {
                this.callArgAt = -1;
                this.callbackArguments = slice.call(arguments, 0);

                return this;
            },

            yieldsOn: function (context) {
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = -1;
                this.callbackArguments = slice.call(arguments, 1);
                this.callbackContext = context;

                return this;
            },

            yieldsTo: function (prop) {
                this.callArgAt = -1;
                this.callArgProp = prop;
                this.callbackArguments = slice.call(arguments, 1);

                return this;
            },

            yieldsToOn: function (prop, context) {
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = -1;
                this.callArgProp = prop;
                this.callbackArguments = slice.call(arguments, 2);
                this.callbackContext = context;

                return this;
            }
        };
        
        // create asynchronous versions of callsArg* and yields* methods
        for (var method in proto) {
            if (proto.hasOwnProperty(method) && method.match(/^(callsArg|yields)/)) {
                proto[method + 'Async'] = (function (syncFnName) {
                    return function () {
                        this.callbackAsync = true;
                        return this[syncFnName].apply(this, arguments);
                    };
                })(method);
            }
        }
        
        return proto;
        
    }()));

    if (commonJSModule) {
        module.exports = stub;
    } else {
        sinon.stub = stub;
    }
}(typeof sinon == "object" && sinon || null));

/**
 * @depend ../sinon.js
 * @depend stub.js
 */
/*jslint eqeqeq: false, onevar: false, nomen: false*/
/*global module, require, sinon*/
/**
 * Mock functions.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon) {
    var commonJSModule = typeof module == "object" && typeof require == "function";
    var push = [].push;

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon) {
        return;
    }

    function mock(object) {
        if (!object) {
            return sinon.expectation.create("Anonymous mock");
        }

        return mock.create(object);
    }

    sinon.mock = mock;

    sinon.extend(mock, (function () {
        function each(collection, callback) {
            if (!collection) {
                return;
            }

            for (var i = 0, l = collection.length; i < l; i += 1) {
                callback(collection[i]);
            }
        }

        return {
            create: function create(object) {
                if (!object) {
                    throw new TypeError("object is null");
                }

                var mockObject = sinon.extend({}, mock);
                mockObject.object = object;
                delete mockObject.create;

                return mockObject;
            },

            expects: function expects(method) {
                if (!method) {
                    throw new TypeError("method is falsy");
                }

                if (!this.expectations) {
                    this.expectations = {};
                    this.proxies = [];
                }

                if (!this.expectations[method]) {
                    this.expectations[method] = [];
                    var mockObject = this;

                    sinon.wrapMethod(this.object, method, function () {
                        return mockObject.invokeMethod(method, this, arguments);
                    });

                    push.call(this.proxies, method);
                }

                var expectation = sinon.expectation.create(method);
                push.call(this.expectations[method], expectation);

                return expectation;
            },

            restore: function restore() {
                var object = this.object;

                each(this.proxies, function (proxy) {
                    if (typeof object[proxy].restore == "function") {
                        object[proxy].restore();
                    }
                });
            },

            verify: function verify() {
                var expectations = this.expectations || {};
                var messages = [], met = [];

                each(this.proxies, function (proxy) {
                    each(expectations[proxy], function (expectation) {
                        if (!expectation.met()) {
                            push.call(messages, expectation.toString());
                        } else {
                            push.call(met, expectation.toString());
                        }
                    });
                });

                this.restore();

                if (messages.length > 0) {
                    sinon.expectation.fail(messages.concat(met).join("\n"));
                } else {
                    sinon.expectation.pass(messages.concat(met).join("\n"));
                }

                return true;
            },

            invokeMethod: function invokeMethod(method, thisValue, args) {
                var expectations = this.expectations && this.expectations[method];
                var length = expectations && expectations.length || 0, i;

                for (i = 0; i < length; i += 1) {
                    if (!expectations[i].met() &&
                        expectations[i].allowsCall(thisValue, args)) {
                        return expectations[i].apply(thisValue, args);
                    }
                }

                var messages = [], available, exhausted = 0;

                for (i = 0; i < length; i += 1) {
                    if (expectations[i].allowsCall(thisValue, args)) {
                        available = available || expectations[i];
                    } else {
                        exhausted += 1;
                    }
                    push.call(messages, "    " + expectations[i].toString());
                }

                if (exhausted === 0) {
                    return available.apply(thisValue, args);
                }

                messages.unshift("Unexpected call: " + sinon.spyCall.toString.call({
                    proxy: method,
                    args: args
                }));

                sinon.expectation.fail(messages.join("\n"));
            }
        };
    }()));

    var times = sinon.timesInWords;

    sinon.expectation = (function () {
        var slice = Array.prototype.slice;
        var _invoke = sinon.spy.invoke;

        function callCountInWords(callCount) {
            if (callCount == 0) {
                return "never called";
            } else {
                return "called " + times(callCount);
            }
        }

        function expectedCallCountInWords(expectation) {
            var min = expectation.minCalls;
            var max = expectation.maxCalls;

            if (typeof min == "number" && typeof max == "number") {
                var str = times(min);

                if (min != max) {
                    str = "at least " + str + " and at most " + times(max);
                }

                return str;
            }

            if (typeof min == "number") {
                return "at least " + times(min);
            }

            return "at most " + times(max);
        }

        function receivedMinCalls(expectation) {
            var hasMinLimit = typeof expectation.minCalls == "number";
            return !hasMinLimit || expectation.callCount >= expectation.minCalls;
        }

        function receivedMaxCalls(expectation) {
            if (typeof expectation.maxCalls != "number") {
                return false;
            }

            return expectation.callCount == expectation.maxCalls;
        }

        return {
            minCalls: 1,
            maxCalls: 1,

            create: function create(methodName) {
                var expectation = sinon.extend(sinon.stub.create(), sinon.expectation);
                delete expectation.create;
                expectation.method = methodName;

                return expectation;
            },

            invoke: function invoke(func, thisValue, args) {
                this.verifyCallAllowed(thisValue, args);

                return _invoke.apply(this, arguments);
            },

            atLeast: function atLeast(num) {
                if (typeof num != "number") {
                    throw new TypeError("'" + num + "' is not number");
                }

                if (!this.limitsSet) {
                    this.maxCalls = null;
                    this.limitsSet = true;
                }

                this.minCalls = num;

                return this;
            },

            atMost: function atMost(num) {
                if (typeof num != "number") {
                    throw new TypeError("'" + num + "' is not number");
                }

                if (!this.limitsSet) {
                    this.minCalls = null;
                    this.limitsSet = true;
                }

                this.maxCalls = num;

                return this;
            },

            never: function never() {
                return this.exactly(0);
            },

            once: function once() {
                return this.exactly(1);
            },

            twice: function twice() {
                return this.exactly(2);
            },

            thrice: function thrice() {
                return this.exactly(3);
            },

            exactly: function exactly(num) {
                if (typeof num != "number") {
                    throw new TypeError("'" + num + "' is not a number");
                }

                this.atLeast(num);
                return this.atMost(num);
            },

            met: function met() {
                return !this.failed && receivedMinCalls(this);
            },

            verifyCallAllowed: function verifyCallAllowed(thisValue, args) {
                if (receivedMaxCalls(this)) {
                    this.failed = true;
                    sinon.expectation.fail(this.method + " already called " + times(this.maxCalls));
                }

                if ("expectedThis" in this && this.expectedThis !== thisValue) {
                    sinon.expectation.fail(this.method + " called with " + thisValue + " as thisValue, expected " +
                        this.expectedThis);
                }

                if (!("expectedArguments" in this)) {
                    return;
                }

                if (!args) {
                    sinon.expectation.fail(this.method + " received no arguments, expected " +
                        this.expectedArguments.join());
                }

                if (args.length < this.expectedArguments.length) {
                    sinon.expectation.fail(this.method + " received too few arguments (" + args.join() +
                        "), expected " + this.expectedArguments.join());
                }

                if (this.expectsExactArgCount &&
                    args.length != this.expectedArguments.length) {
                    sinon.expectation.fail(this.method + " received too many arguments (" + args.join() +
                        "), expected " + this.expectedArguments.join());
                }

                for (var i = 0, l = this.expectedArguments.length; i < l; i += 1) {
                    if (!sinon.deepEqual(this.expectedArguments[i], args[i])) {
                        sinon.expectation.fail(this.method + " received wrong arguments (" + args.join() +
                            "), expected " + this.expectedArguments.join());
                    }
                }
            },

            allowsCall: function allowsCall(thisValue, args) {
                if (this.met() && receivedMaxCalls(this)) {
                    return false;
                }

                if ("expectedThis" in this && this.expectedThis !== thisValue) {
                    return false;
                }

                if (!("expectedArguments" in this)) {
                    return true;
                }

                args = args || [];

                if (args.length < this.expectedArguments.length) {
                    return false;
                }

                if (this.expectsExactArgCount &&
                    args.length != this.expectedArguments.length) {
                    return false;
                }

                for (var i = 0, l = this.expectedArguments.length; i < l; i += 1) {
                    if (!sinon.deepEqual(this.expectedArguments[i], args[i])) {
                        return false;
                    }
                }

                return true;
            },

            withArgs: function withArgs() {
                this.expectedArguments = slice.call(arguments);
                return this;
            },

            withExactArgs: function withExactArgs() {
                this.withArgs.apply(this, arguments);
                this.expectsExactArgCount = true;
                return this;
            },

            on: function on(thisValue) {
                this.expectedThis = thisValue;
                return this;
            },

            toString: function () {
                var args = (this.expectedArguments || []).slice();

                if (!this.expectsExactArgCount) {
                    push.call(args, "[...]");
                }

                var callStr = sinon.spyCall.toString.call({
                    proxy: this.method, args: args
                });

                var message = callStr.replace(", [...", "[, ...") + " " +
                    expectedCallCountInWords(this);

                if (this.met()) {
                    return "Expectation met: " + message;
                }

                return "Expected " + message + " (" +
                    callCountInWords(this.callCount) + ")";
            },

            verify: function verify() {
                if (!this.met()) {
                    sinon.expectation.fail(this.toString());
                } else {
                    sinon.expectation.pass(this.toString());
                }

                return true;
            },

            pass: function(message) {
              sinon.assert.pass(message);
            },
            fail: function (message) {
                var exception = new Error(message);
                exception.name = "ExpectationError";

                throw exception;
            }
        };
    }());

    if (commonJSModule) {
        module.exports = mock;
    } else {
        sinon.mock = mock;
    }
}(typeof sinon == "object" && sinon || null));

/**
 * @depend ../sinon.js
 * @depend stub.js
 * @depend mock.js
 */
/*jslint eqeqeq: false, onevar: false, forin: true*/
/*global module, require, sinon*/
/**
 * Collections of stubs, spies and mocks.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon) {
    var commonJSModule = typeof module == "object" && typeof require == "function";
    var push = [].push;

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon) {
        return;
    }

    function getFakes(fakeCollection) {
        if (!fakeCollection.fakes) {
            fakeCollection.fakes = [];
        }

        return fakeCollection.fakes;
    }

    function each(fakeCollection, method) {
        var fakes = getFakes(fakeCollection);

        for (var i = 0, l = fakes.length; i < l; i += 1) {
            if (typeof fakes[i][method] == "function") {
                fakes[i][method]();
            }
        }
    }

    function compact(fakeCollection) {
        var fakes = getFakes(fakeCollection);
        var i = 0;
        while (i < fakes.length) {
          fakes.splice(i, 1);
        }
    }

    var collection = {
        verify: function resolve() {
            each(this, "verify");
        },

        restore: function restore() {
            each(this, "restore");
            compact(this);
        },

        verifyAndRestore: function verifyAndRestore() {
            var exception;

            try {
                this.verify();
            } catch (e) {
                exception = e;
            }

            this.restore();

            if (exception) {
                throw exception;
            }
        },

        add: function add(fake) {
            push.call(getFakes(this), fake);
            return fake;
        },

        spy: function spy() {
            return this.add(sinon.spy.apply(sinon, arguments));
        },

        stub: function stub(object, property, value) {
            if (property) {
                var original = object[property];

                if (typeof original != "function") {
                    if (!object.hasOwnProperty(property)) {
                        throw new TypeError("Cannot stub non-existent own property " + property);
                    }

                    object[property] = value;

                    return this.add({
                        restore: function () {
                            object[property] = original;
                        }
                    });
                }
            }
            if (!property && !!object && typeof object == "object") {
                var stubbedObj = sinon.stub.apply(sinon, arguments);

                for (var prop in stubbedObj) {
                    if (typeof stubbedObj[prop] === "function") {
                        this.add(stubbedObj[prop]);
                    }
                }

                return stubbedObj;
            }

            return this.add(sinon.stub.apply(sinon, arguments));
        },

        mock: function mock() {
            return this.add(sinon.mock.apply(sinon, arguments));
        },

        inject: function inject(obj) {
            var col = this;

            obj.spy = function () {
                return col.spy.apply(col, arguments);
            };

            obj.stub = function () {
                return col.stub.apply(col, arguments);
            };

            obj.mock = function () {
                return col.mock.apply(col, arguments);
            };

            return obj;
        }
    };

    if (commonJSModule) {
        module.exports = collection;
    } else {
        sinon.collection = collection;
    }
}(typeof sinon == "object" && sinon || null));

/**
 * @depend ../sinon.js
 * @depend collection.js
 * @depend util/fake_timers.js
 * @depend util/fake_server_with_clock.js
 */
/*jslint eqeqeq: false, onevar: false, plusplus: false*/
/*global require, module*/
/**
 * Manages fake collections as well as fake utilities such as Sinon's
 * timers and fake XHR implementation in one convenient object.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

if (typeof module == "object" && typeof require == "function") {
    var sinon = require("../sinon");
    sinon.extend(sinon, require("./util/fake_timers"));
}

(function () {
    var push = [].push;

    function exposeValue(sandbox, config, key, value) {
        if (!value) {
            return;
        }

        if (config.injectInto) {
            config.injectInto[key] = value;
        } else {
            push.call(sandbox.args, value);
        }
    }

    function prepareSandboxFromConfig(config) {
        var sandbox = sinon.create(sinon.sandbox);

        if (config.useFakeServer) {
            if (typeof config.useFakeServer == "object") {
                sandbox.serverPrototype = config.useFakeServer;
            }

            sandbox.useFakeServer();
        }

        if (config.useFakeTimers) {
            if (typeof config.useFakeTimers == "object") {
                sandbox.useFakeTimers.apply(sandbox, config.useFakeTimers);
            } else {
                sandbox.useFakeTimers();
            }
        }

        return sandbox;
    }

    sinon.sandbox = sinon.extend(sinon.create(sinon.collection), {
        useFakeTimers: function useFakeTimers() {
            this.clock = sinon.useFakeTimers.apply(sinon, arguments);

            return this.add(this.clock);
        },

        serverPrototype: sinon.fakeServer,

        useFakeServer: function useFakeServer() {
            var proto = this.serverPrototype || sinon.fakeServer;

            if (!proto || !proto.create) {
                return null;
            }

            this.server = proto.create();
            return this.add(this.server);
        },

        inject: function (obj) {
            sinon.collection.inject.call(this, obj);

            if (this.clock) {
                obj.clock = this.clock;
            }

            if (this.server) {
                obj.server = this.server;
                obj.requests = this.server.requests;
            }

            return obj;
        },

        create: function (config) {
            if (!config) {
                return sinon.create(sinon.sandbox);
            }

            var sandbox = prepareSandboxFromConfig(config);
            sandbox.args = sandbox.args || [];
            var prop, value, exposed = sandbox.inject({});

            if (config.properties) {
                for (var i = 0, l = config.properties.length; i < l; i++) {
                    prop = config.properties[i];
                    value = exposed[prop] || prop == "sandbox" && sandbox;
                    exposeValue(sandbox, config, prop, value);
                }
            } else {
                exposeValue(sandbox, config, "sandbox", value);
            }

            return sandbox;
        }
    });

    sinon.sandbox.useFakeXMLHttpRequest = sinon.sandbox.useFakeServer;

    if (typeof module == "object" && typeof require == "function") {
        module.exports = sinon.sandbox;
    }
}());

/**
 * @depend ../sinon.js
 * @depend stub.js
 * @depend mock.js
 * @depend sandbox.js
 */
/*jslint eqeqeq: false, onevar: false, forin: true, plusplus: false*/
/*global module, require, sinon*/
/**
 * Test function, sandboxes fakes
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon) {
    var commonJSModule = typeof module == "object" && typeof require == "function";

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon) {
        return;
    }

    function test(callback) {
        var type = typeof callback;

        if (type != "function") {
            throw new TypeError("sinon.test needs to wrap a test function, got " + type);
        }

        return function () {
            var config = sinon.getConfig(sinon.config);
            config.injectInto = config.injectIntoThis && this || config.injectInto;
            var sandbox = sinon.sandbox.create(config);
            var exception, result;
            var args = Array.prototype.slice.call(arguments).concat(sandbox.args);

            try {
                result = callback.apply(this, args);
            } finally {
                sandbox.verifyAndRestore();
            }

            return result;
        };
    }

    test.config = {
        injectIntoThis: true,
        injectInto: null,
        properties: ["spy", "stub", "mock", "clock", "server", "requests"],
        useFakeTimers: true,
        useFakeServer: true
    };

    if (commonJSModule) {
        module.exports = test;
    } else {
        sinon.test = test;
    }
}(typeof sinon == "object" && sinon || null));

/**
 * @depend ../sinon.js
 * @depend test.js
 */
/*jslint eqeqeq: false, onevar: false, eqeqeq: false*/
/*global module, require, sinon*/
/**
 * Test case, sandboxes all test functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon) {
    var commonJSModule = typeof module == "object" && typeof require == "function";

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon || !Object.prototype.hasOwnProperty) {
        return;
    }

    function createTest(property, setUp, tearDown) {
        return function () {
            if (setUp) {
                setUp.apply(this, arguments);
            }

            var exception, result;

            try {
                result = property.apply(this, arguments);
            } catch (e) {
                exception = e;
            }

            if (tearDown) {
                tearDown.apply(this, arguments);
            }

            if (exception) {
                throw exception;
            }

            return result;
        };
    }

    function testCase(tests, prefix) {
        /*jsl:ignore*/
        if (!tests || typeof tests != "object") {
            throw new TypeError("sinon.testCase needs an object with test functions");
        }
        /*jsl:end*/

        prefix = prefix || "test";
        var rPrefix = new RegExp("^" + prefix);
        var methods = {}, testName, property, method;
        var setUp = tests.setUp;
        var tearDown = tests.tearDown;

        for (testName in tests) {
            if (tests.hasOwnProperty(testName)) {
                property = tests[testName];

                if (/^(setUp|tearDown)$/.test(testName)) {
                    continue;
                }

                if (typeof property == "function" && rPrefix.test(testName)) {
                    method = property;

                    if (setUp || tearDown) {
                        method = createTest(property, setUp, tearDown);
                    }

                    methods[testName] = sinon.test(method);
                } else {
                    methods[testName] = tests[testName];
                }
            }
        }

        return methods;
    }

    if (commonJSModule) {
        module.exports = testCase;
    } else {
        sinon.testCase = testCase;
    }
}(typeof sinon == "object" && sinon || null));

/**
 * @depend ../sinon.js
 * @depend stub.js
 */
/*jslint eqeqeq: false, onevar: false, nomen: false, plusplus: false*/
/*global module, require, sinon*/
/**
 * Assertions matching the test spy retrieval interface.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function (sinon, global) {
    var commonJSModule = typeof module == "object" && typeof require == "function";
    var slice = Array.prototype.slice;
    var assert;

    if (!sinon && commonJSModule) {
        sinon = require("../sinon");
    }

    if (!sinon) {
        return;
    }

    function verifyIsStub() {
        var method;

        for (var i = 0, l = arguments.length; i < l; ++i) {
            method = arguments[i];

            if (!method) {
                assert.fail("fake is not a spy");
            }

            if (typeof method != "function") {
                assert.fail(method + " is not a function");
            }

            if (typeof method.getCall != "function") {
                assert.fail(method + " is not stubbed");
            }
        }
    }

    function failAssertion(object, msg) {
        object = object || global;
        var failMethod = object.fail || assert.fail;
        failMethod.call(object, msg);
    }

    function mirrorPropAsAssertion(name, method, message) {
        if (arguments.length == 2) {
            message = method;
            method = name;
        }

        assert[name] = function (fake) {
            verifyIsStub(fake);

            var args = slice.call(arguments, 1);
            var failed = false;

            if (typeof method == "function") {
                failed = !method(fake);
            } else {
                failed = typeof fake[method] == "function" ?
                    !fake[method].apply(fake, args) : !fake[method];
            }

            if (failed) {
                failAssertion(this, fake.printf.apply(fake, [message].concat(args)));
            } else {
                assert.pass(name);
            }
        };
    }

    function exposedName(prefix, prop) {
        return !prefix || /^fail/.test(prop) ? prop :
            prefix + prop.slice(0, 1).toUpperCase() + prop.slice(1);
    };

    assert = {
        failException: "AssertError",

        fail: function fail(message) {
            var error = new Error(message);
            error.name = this.failException || assert.failException;

            throw error;
        },

        pass: function pass(assertion) {},

        callOrder: function assertCallOrder() {
            verifyIsStub.apply(null, arguments);
            var expected = "", actual = "";

            if (!sinon.calledInOrder(arguments)) {
                try {
                    expected = [].join.call(arguments, ", ");
                    actual = sinon.orderByFirstCall(slice.call(arguments)).join(", ");
                } catch (e) {
                    // If this fails, we'll just fall back to the blank string
                }

                failAssertion(this, "expected " + expected + " to be " +
                              "called in order but were called as " + actual);
            } else {
                assert.pass("callOrder");
            }
        },

        callCount: function assertCallCount(method, count) {
            verifyIsStub(method);

            if (method.callCount != count) {
                var msg = "expected %n to be called " + sinon.timesInWords(count) +
                    " but was called %c%C";
                failAssertion(this, method.printf(msg));
            } else {
                assert.pass("callCount");
            }
        },

        expose: function expose(target, options) {
            if (!target) {
                throw new TypeError("target is null or undefined");
            }

            var o = options || {};
            var prefix = typeof o.prefix == "undefined" && "assert" || o.prefix;
            var includeFail = typeof o.includeFail == "undefined" || !!o.includeFail;

            for (var method in this) {
                if (method != "export" && (includeFail || !/^(fail)/.test(method))) {
                    target[exposedName(prefix, method)] = this[method];
                }
            }

            return target;
        }
    };

    mirrorPropAsAssertion("called", "expected %n to have been called at least once but was never called");
    mirrorPropAsAssertion("notCalled", function (spy) { return !spy.called; },
                          "expected %n to not have been called but was called %c%C");
    mirrorPropAsAssertion("calledOnce", "expected %n to be called once but was called %c%C");
    mirrorPropAsAssertion("calledTwice", "expected %n to be called twice but was called %c%C");
    mirrorPropAsAssertion("calledThrice", "expected %n to be called thrice but was called %c%C");
    mirrorPropAsAssertion("calledOn", "expected %n to be called with %1 as this but was called with %t");
    mirrorPropAsAssertion("alwaysCalledOn", "expected %n to always be called with %1 as this but was called with %t");
    mirrorPropAsAssertion("calledWithNew", "expected %n to be called with new");
    mirrorPropAsAssertion("alwaysCalledWithNew", "expected %n to always be called with new");
    mirrorPropAsAssertion("calledWith", "expected %n to be called with arguments %*%C");
    mirrorPropAsAssertion("calledWithMatch", "expected %n to be called with match %*%C");
    mirrorPropAsAssertion("alwaysCalledWith", "expected %n to always be called with arguments %*%C");
    mirrorPropAsAssertion("alwaysCalledWithMatch", "expected %n to always be called with match %*%C");
    mirrorPropAsAssertion("calledWithExactly", "expected %n to be called with exact arguments %*%C");
    mirrorPropAsAssertion("alwaysCalledWithExactly", "expected %n to always be called with exact arguments %*%C");
    mirrorPropAsAssertion("neverCalledWith", "expected %n to never be called with arguments %*%C");
    mirrorPropAsAssertion("neverCalledWithMatch", "expected %n to never be called with match %*%C");
    mirrorPropAsAssertion("threw", "%n did not throw exception%C");
    mirrorPropAsAssertion("alwaysThrew", "%n did not always throw exception%C");

    if (commonJSModule) {
        module.exports = assert;
    } else {
        sinon.assert = assert;
    }
}(typeof sinon == "object" && sinon || null, typeof window != "undefined" ? window : global));

/*jslint eqeqeq: false, onevar: false*/
/*global sinon, module, require, ActiveXObject, XMLHttpRequest, DOMParser*/
/**
 * Minimal Event interface implementation
 *
 * Original implementation by Sven Fuchs: https://gist.github.com/995028
 * Modifications and tests by Christian Johansen.
 *
 * @author Sven Fuchs (svenfuchs@artweb-design.de)
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2011 Sven Fuchs, Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    this.sinon = {};
}

(function () {
    var push = [].push;

    sinon.Event = function Event(type, bubbles, cancelable) {
        this.initEvent(type, bubbles, cancelable);
    };

    sinon.Event.prototype = {
        initEvent: function(type, bubbles, cancelable) {
            this.type = type;
            this.bubbles = bubbles;
            this.cancelable = cancelable;
        },

        stopPropagation: function () {},

        preventDefault: function () {
            this.defaultPrevented = true;
        }
    };

    sinon.EventTarget = {
        addEventListener: function addEventListener(event, listener, useCapture) {
            this.eventListeners = this.eventListeners || {};
            this.eventListeners[event] = this.eventListeners[event] || [];
            push.call(this.eventListeners[event], listener);
        },

        removeEventListener: function removeEventListener(event, listener, useCapture) {
            var listeners = this.eventListeners && this.eventListeners[event] || [];

            for (var i = 0, l = listeners.length; i < l; ++i) {
                if (listeners[i] == listener) {
                    return listeners.splice(i, 1);
                }
            }
        },

        dispatchEvent: function dispatchEvent(event) {
            var type = event.type;
            var listeners = this.eventListeners && this.eventListeners[type] || [];

            for (var i = 0; i < listeners.length; i++) {
                if (typeof listeners[i] == "function") {
                    listeners[i].call(this, event);
                } else {
                    listeners[i].handleEvent(event);
                }
            }

            return !!event.defaultPrevented;
        }
    };
}());

/**
 * @depend event.js
 */
/*jslint eqeqeq: false, onevar: false*/
/*global sinon, module, require, ActiveXObject, XMLHttpRequest, DOMParser*/
/**
 * Fake XMLHttpRequest object
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    this.sinon = {};
}
sinon.xhr = { XMLHttpRequest: this.XMLHttpRequest };

// wrapper for global
(function(global) {
    var xhr = sinon.xhr;
    xhr.GlobalXMLHttpRequest = global.XMLHttpRequest;
    xhr.GlobalActiveXObject = global.ActiveXObject;
    xhr.supportsActiveX = typeof xhr.GlobalActiveXObject != "undefined";
    xhr.supportsXHR = typeof xhr.GlobalXMLHttpRequest != "undefined";
    xhr.workingXHR = xhr.supportsXHR ? xhr.GlobalXMLHttpRequest : xhr.supportsActiveX
                                     ? function() { return new xhr.GlobalActiveXObject("MSXML2.XMLHTTP.3.0") } : false;

    /*jsl:ignore*/
    var unsafeHeaders = {
        "Accept-Charset": true,
        "Accept-Encoding": true,
        "Connection": true,
        "Content-Length": true,
        "Cookie": true,
        "Cookie2": true,
        "Content-Transfer-Encoding": true,
        "Date": true,
        "Expect": true,
        "Host": true,
        "Keep-Alive": true,
        "Referer": true,
        "TE": true,
        "Trailer": true,
        "Transfer-Encoding": true,
        "Upgrade": true,
        "User-Agent": true,
        "Via": true
    };
    /*jsl:end*/

    function FakeXMLHttpRequest() {
        this.readyState = FakeXMLHttpRequest.UNSENT;
        this.requestHeaders = {};
        this.requestBody = null;
        this.status = 0;
        this.statusText = "";

        if (typeof FakeXMLHttpRequest.onCreate == "function") {
            FakeXMLHttpRequest.onCreate(this);
        }
    }

    function verifyState(xhr) {
        if (xhr.readyState !== FakeXMLHttpRequest.OPENED) {
            throw new Error("INVALID_STATE_ERR");
        }

        if (xhr.sendFlag) {
            throw new Error("INVALID_STATE_ERR");
        }
    }

    // filtering to enable a white-list version of Sinon FakeXhr,
    // where whitelisted requests are passed through to real XHR
    function each(collection, callback) {
        if (!collection) return;
        for (var i = 0, l = collection.length; i < l; i += 1) {
            callback(collection[i]);
        }
    }
    function some(collection, callback) {
        for (var index = 0; index < collection.length; index++) {
            if(callback(collection[index]) === true) return true;
        };
        return false;
    }
    // largest arity in XHR is 5 - XHR#open
    var apply = function(obj,method,args) {
        switch(args.length) {
        case 0: return obj[method]();
        case 1: return obj[method](args[0]);
        case 2: return obj[method](args[0],args[1]);
        case 3: return obj[method](args[0],args[1],args[2]);
        case 4: return obj[method](args[0],args[1],args[2],args[3]);
        case 5: return obj[method](args[0],args[1],args[2],args[3],args[4]);
        };
    };

    FakeXMLHttpRequest.filters = [];
    FakeXMLHttpRequest.addFilter = function(fn) {
        this.filters.push(fn)
    };
    var IE6Re = /MSIE 6/;
    FakeXMLHttpRequest.defake = function(fakeXhr,xhrArgs) {
        var xhr = new sinon.xhr.workingXHR();
        each(["open","setRequestHeader","send","abort","getResponseHeader",
              "getAllResponseHeaders","addEventListener","overrideMimeType","removeEventListener"],
             function(method) {
                 fakeXhr[method] = function() {
                   return apply(xhr,method,arguments);
                 };
             });

        var copyAttrs = function(args) {
            each(args, function(attr) {
              try {
                fakeXhr[attr] = xhr[attr]
              } catch(e) {
                if(!IE6Re.test(navigator.userAgent)) throw e;
              }
            });
        };

        var stateChange = function() {
            fakeXhr.readyState = xhr.readyState;
            if(xhr.readyState >= FakeXMLHttpRequest.HEADERS_RECEIVED) {
                copyAttrs(["status","statusText"]);
            }
            if(xhr.readyState >= FakeXMLHttpRequest.LOADING) {
                copyAttrs(["responseText"]);
            }
            if(xhr.readyState === FakeXMLHttpRequest.DONE) {
                copyAttrs(["responseXML"]);
            }
            if(fakeXhr.onreadystatechange) fakeXhr.onreadystatechange.call(fakeXhr);
        };
        if(xhr.addEventListener) {
          for(var event in fakeXhr.eventListeners) {
              if(fakeXhr.eventListeners.hasOwnProperty(event)) {
                  each(fakeXhr.eventListeners[event],function(handler) {
                      xhr.addEventListener(event, handler);
                  });
              }
          }
          xhr.addEventListener("readystatechange",stateChange);
        } else {
          xhr.onreadystatechange = stateChange;
        }
        apply(xhr,"open",xhrArgs);
    };
    FakeXMLHttpRequest.useFilters = false;

    function verifyRequestSent(xhr) {
        if (xhr.readyState == FakeXMLHttpRequest.DONE) {
            throw new Error("Request done");
        }
    }

    function verifyHeadersReceived(xhr) {
        if (xhr.async && xhr.readyState != FakeXMLHttpRequest.HEADERS_RECEIVED) {
            throw new Error("No headers received");
        }
    }

    function verifyResponseBodyType(body) {
        if (typeof body != "string") {
            var error = new Error("Attempted to respond to fake XMLHttpRequest with " +
                                 body + ", which is not a string.");
            error.name = "InvalidBodyException";
            throw error;
        }
    }

    sinon.extend(FakeXMLHttpRequest.prototype, sinon.EventTarget, {
        async: true,

        open: function open(method, url, async, username, password) {
            this.method = method;
            this.url = url;
            this.async = typeof async == "boolean" ? async : true;
            this.username = username;
            this.password = password;
            this.responseText = null;
            this.responseXML = null;
            this.requestHeaders = {};
            this.sendFlag = false;
            if(sinon.FakeXMLHttpRequest.useFilters === true) {
                var xhrArgs = arguments;
                var defake = some(FakeXMLHttpRequest.filters,function(filter) {
                    return filter.apply(this,xhrArgs)
                });
                if (defake) {
                  return sinon.FakeXMLHttpRequest.defake(this,arguments);
                }
            }
            this.readyStateChange(FakeXMLHttpRequest.OPENED);
        },

        readyStateChange: function readyStateChange(state) {
            this.readyState = state;

            if (typeof this.onreadystatechange == "function") {
                try {
                    this.onreadystatechange();
                } catch (e) {
                    sinon.logError("Fake XHR onreadystatechange handler", e);
                }
            }

            this.dispatchEvent(new sinon.Event("readystatechange"));
        },

        setRequestHeader: function setRequestHeader(header, value) {
            verifyState(this);

            if (unsafeHeaders[header] || /^(Sec-|Proxy-)/.test(header)) {
                throw new Error("Refused to set unsafe header \"" + header + "\"");
            }

            if (this.requestHeaders[header]) {
                this.requestHeaders[header] += "," + value;
            } else {
                this.requestHeaders[header] = value;
            }
        },

        // Helps testing
        setResponseHeaders: function setResponseHeaders(headers) {
            this.responseHeaders = {};

            for (var header in headers) {
                if (headers.hasOwnProperty(header)) {
                    this.responseHeaders[header] = headers[header];
                }
            }

            if (this.async) {
                this.readyStateChange(FakeXMLHttpRequest.HEADERS_RECEIVED);
            }
        },

        // Currently treats ALL data as a DOMString (i.e. no Document)
        send: function send(data) {
            verifyState(this);

            if (!/^(get|head)$/i.test(this.method)) {
                if (this.requestHeaders["Content-Type"]) {
                    var value = this.requestHeaders["Content-Type"].split(";");
                    this.requestHeaders["Content-Type"] = value[0] + ";charset=utf-8";
                } else {
                    this.requestHeaders["Content-Type"] = "text/plain;charset=utf-8";
                }

                this.requestBody = data;
            }

            this.errorFlag = false;
            this.sendFlag = this.async;
            this.readyStateChange(FakeXMLHttpRequest.OPENED);

            if (typeof this.onSend == "function") {
                this.onSend(this);
            }
        },

        abort: function abort() {
            this.aborted = true;
            this.responseText = null;
            this.errorFlag = true;
            this.requestHeaders = {};

            if (this.readyState > sinon.FakeXMLHttpRequest.UNSENT && this.sendFlag) {
                this.readyStateChange(sinon.FakeXMLHttpRequest.DONE);
                this.sendFlag = false;
            }

            this.readyState = sinon.FakeXMLHttpRequest.UNSENT;
        },

        getResponseHeader: function getResponseHeader(header) {
            if (this.readyState < FakeXMLHttpRequest.HEADERS_RECEIVED) {
                return null;
            }

            if (/^Set-Cookie2?$/i.test(header)) {
                return null;
            }

            header = header.toLowerCase();

            for (var h in this.responseHeaders) {
                if (h.toLowerCase() == header) {
                    return this.responseHeaders[h];
                }
            }

            return null;
        },

        getAllResponseHeaders: function getAllResponseHeaders() {
            if (this.readyState < FakeXMLHttpRequest.HEADERS_RECEIVED) {
                return "";
            }

            var headers = "";

            for (var header in this.responseHeaders) {
                if (this.responseHeaders.hasOwnProperty(header) &&
                    !/^Set-Cookie2?$/i.test(header)) {
                    headers += header + ": " + this.responseHeaders[header] + "\r\n";
                }
            }

            return headers;
        },

        setResponseBody: function setResponseBody(body) {
            verifyRequestSent(this);
            verifyHeadersReceived(this);
            verifyResponseBodyType(body);

            var chunkSize = this.chunkSize || 10;
            var index = 0;
            this.responseText = "";

            do {
                if (this.async) {
                    this.readyStateChange(FakeXMLHttpRequest.LOADING);
                }

                this.responseText += body.substring(index, index + chunkSize);
                index += chunkSize;
            } while (index < body.length);

            var type = this.getResponseHeader("Content-Type");

            if (this.responseText &&
                (!type || /(text\/xml)|(application\/xml)|(\+xml)/.test(type))) {
                try {
                    this.responseXML = FakeXMLHttpRequest.parseXML(this.responseText);
                } catch (e) {
                    // Unable to parse XML - no biggie
                }
            }

            if (this.async) {
                this.readyStateChange(FakeXMLHttpRequest.DONE);
            } else {
                this.readyState = FakeXMLHttpRequest.DONE;
            }
        },

        respond: function respond(status, headers, body) {
            this.setResponseHeaders(headers || {});
            this.status = typeof status == "number" ? status : 200;
            this.statusText = FakeXMLHttpRequest.statusCodes[this.status];
            this.setResponseBody(body || "");
        }
    });

    sinon.extend(FakeXMLHttpRequest, {
        UNSENT: 0,
        OPENED: 1,
        HEADERS_RECEIVED: 2,
        LOADING: 3,
        DONE: 4
    });

    // Borrowed from JSpec
    FakeXMLHttpRequest.parseXML = function parseXML(text) {
        var xmlDoc;

        if (typeof DOMParser != "undefined") {
            var parser = new DOMParser();
            xmlDoc = parser.parseFromString(text, "text/xml");
        } else {
            xmlDoc = new ActiveXObject("Microsoft.XMLDOM");
            xmlDoc.async = "false";
            xmlDoc.loadXML(text);
        }

        return xmlDoc;
    };

    FakeXMLHttpRequest.statusCodes = {
        100: "Continue",
        101: "Switching Protocols",
        200: "OK",
        201: "Created",
        202: "Accepted",
        203: "Non-Authoritative Information",
        204: "No Content",
        205: "Reset Content",
        206: "Partial Content",
        300: "Multiple Choice",
        301: "Moved Permanently",
        302: "Found",
        303: "See Other",
        304: "Not Modified",
        305: "Use Proxy",
        307: "Temporary Redirect",
        400: "Bad Request",
        401: "Unauthorized",
        402: "Payment Required",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        407: "Proxy Authentication Required",
        408: "Request Timeout",
        409: "Conflict",
        410: "Gone",
        411: "Length Required",
        412: "Precondition Failed",
        413: "Request Entity Too Large",
        414: "Request-URI Too Long",
        415: "Unsupported Media Type",
        416: "Requested Range Not Satisfiable",
        417: "Expectation Failed",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        505: "HTTP Version Not Supported"
    };

    sinon.useFakeXMLHttpRequest = function () {
        sinon.FakeXMLHttpRequest.restore = function restore(keepOnCreate) {
            if (xhr.supportsXHR) {
                global.XMLHttpRequest = xhr.GlobalXMLHttpRequest;
            }

            if (xhr.supportsActiveX) {
                global.ActiveXObject = xhr.GlobalActiveXObject;
            }

            delete sinon.FakeXMLHttpRequest.restore;

            if (keepOnCreate !== true) {
                delete sinon.FakeXMLHttpRequest.onCreate;
            }
        };
        if (xhr.supportsXHR) {
            global.XMLHttpRequest = sinon.FakeXMLHttpRequest;
        }

        if (xhr.supportsActiveX) {
            global.ActiveXObject = function ActiveXObject(objId) {
                if (objId == "Microsoft.XMLHTTP" || /^Msxml2\.XMLHTTP/i.test(objId)) {

                    return new sinon.FakeXMLHttpRequest();
                }

                return new xhr.GlobalActiveXObject(objId);
            };
        }

        return sinon.FakeXMLHttpRequest;
    };

    sinon.FakeXMLHttpRequest = FakeXMLHttpRequest;
})(this);

if (typeof module == "object" && typeof require == "function") {
    module.exports = sinon;
}

/*jslint eqeqeq: false, plusplus: false, evil: true, onevar: false, browser: true, forin: false*/
/*global module, require, window*/
/**
 * Fake timer API
 * setTimeout
 * setInterval
 * clearTimeout
 * clearInterval
 * tick
 * reset
 * Date
 *
 * Inspired by jsUnitMockTimeOut from JsUnit
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    var sinon = {};
}

(function (global) {
    var id = 1;

    function addTimer(args, recurring) {
        if (args.length === 0) {
            throw new Error("Function requires at least 1 parameter");
        }

        var toId = id++;
        var delay = args[1] || 0;

        if (!this.timeouts) {
            this.timeouts = {};
        }

        this.timeouts[toId] = {
            id: toId,
            func: args[0],
            callAt: this.now + delay,
            invokeArgs: Array.prototype.slice.call(args, 2)
        };

        if (recurring === true) {
            this.timeouts[toId].interval = delay;
        }

        return toId;
    }

    function parseTime(str) {
        if (!str) {
            return 0;
        }

        var strings = str.split(":");
        var l = strings.length, i = l;
        var ms = 0, parsed;

        if (l > 3 || !/^(\d\d:){0,2}\d\d?$/.test(str)) {
            throw new Error("tick only understands numbers and 'h:m:s'");
        }

        while (i--) {
            parsed = parseInt(strings[i], 10);

            if (parsed >= 60) {
                throw new Error("Invalid time " + str);
            }

            ms += parsed * Math.pow(60, (l - i - 1));
        }

        return ms * 1000;
    }

    function createObject(object) {
        var newObject;

        if (Object.create) {
            newObject = Object.create(object);
        } else {
            var F = function () {};
            F.prototype = object;
            newObject = new F();
        }

        newObject.Date.clock = newObject;
        return newObject;
    }

    sinon.clock = {
        now: 0,

        create: function create(now) {
            var clock = createObject(this);

            if (typeof now == "number") {
                clock.now = now;
            }

            if (!!now && typeof now == "object") {
                throw new TypeError("now should be milliseconds since UNIX epoch");
            }

            return clock;
        },

        setTimeout: function setTimeout(callback, timeout) {
            return addTimer.call(this, arguments, false);
        },

        clearTimeout: function clearTimeout(timerId) {
            if (!this.timeouts) {
                this.timeouts = [];
            }

            if (timerId in this.timeouts) {
                delete this.timeouts[timerId];
            }
        },

        setInterval: function setInterval(callback, timeout) {
            return addTimer.call(this, arguments, true);
        },

        clearInterval: function clearInterval(timerId) {
            this.clearTimeout(timerId);
        },

        tick: function tick(ms) {
            ms = typeof ms == "number" ? ms : parseTime(ms);
            var tickFrom = this.now, tickTo = this.now + ms, previous = this.now;
            var timer = this.firstTimerInRange(tickFrom, tickTo);

            var firstException;
            while (timer && tickFrom <= tickTo) {
                if (this.timeouts[timer.id]) {
                    tickFrom = this.now = timer.callAt;
                    try {
                      this.callTimer(timer);
                    } catch (e) {
                      firstException = firstException || e;
                    }
                }

                timer = this.firstTimerInRange(previous, tickTo);
                previous = tickFrom;
            }

            this.now = tickTo;

            if (firstException) {
              throw firstException;
            }
        },

        firstTimerInRange: function (from, to) {
            var timer, smallest, originalTimer;

            for (var id in this.timeouts) {
                if (this.timeouts.hasOwnProperty(id)) {
                    if (this.timeouts[id].callAt < from || this.timeouts[id].callAt > to) {
                        continue;
                    }

                    if (!smallest || this.timeouts[id].callAt < smallest) {
                        originalTimer = this.timeouts[id];
                        smallest = this.timeouts[id].callAt;

                        timer = {
                            func: this.timeouts[id].func,
                            callAt: this.timeouts[id].callAt,
                            interval: this.timeouts[id].interval,
                            id: this.timeouts[id].id,
                            invokeArgs: this.timeouts[id].invokeArgs
                        };
                    }
                }
            }

            return timer || null;
        },

        callTimer: function (timer) {
            if (typeof timer.interval == "number") {
                this.timeouts[timer.id].callAt += timer.interval;
            } else {
                delete this.timeouts[timer.id];
            }

            try {
                if (typeof timer.func == "function") {
                    timer.func.apply(null, timer.invokeArgs);
                } else {
                    eval(timer.func);
                }
            } catch (e) {
              var exception = e;
            }

            if (!this.timeouts[timer.id]) {
                if (exception) {
                  throw exception;
                }
                return;
            }

            if (exception) {
              throw exception;
            }
        },

        reset: function reset() {
            this.timeouts = {};
        },

        Date: (function () {
            var NativeDate = Date;

            function ClockDate(year, month, date, hour, minute, second, ms) {
                // Defensive and verbose to avoid potential harm in passing
                // explicit undefined when user does not pass argument
                switch (arguments.length) {
                case 0:
                    return new NativeDate(ClockDate.clock.now);
                case 1:
                    return new NativeDate(year);
                case 2:
                    return new NativeDate(year, month);
                case 3:
                    return new NativeDate(year, month, date);
                case 4:
                    return new NativeDate(year, month, date, hour);
                case 5:
                    return new NativeDate(year, month, date, hour, minute);
                case 6:
                    return new NativeDate(year, month, date, hour, minute, second);
                default:
                    return new NativeDate(year, month, date, hour, minute, second, ms);
                }
            }

            return mirrorDateProperties(ClockDate, NativeDate);
        }())
    };

    function mirrorDateProperties(target, source) {
        if (source.now) {
            target.now = function now() {
                return target.clock.now;
            };
        } else {
            delete target.now;
        }

        if (source.toSource) {
            target.toSource = function toSource() {
                return source.toSource();
            };
        } else {
            delete target.toSource;
        }

        target.toString = function toString() {
            return source.toString();
        };

        target.prototype = source.prototype;
        target.parse = source.parse;
        target.UTC = source.UTC;
        target.prototype.toUTCString = source.prototype.toUTCString;
        return target;
    }

    var methods = ["Date", "setTimeout", "setInterval",
                   "clearTimeout", "clearInterval"];

    function restore() {
        var method;

        for (var i = 0, l = this.methods.length; i < l; i++) {
            method = this.methods[i];
            if (global[method].hadOwnProperty) {
                global[method] = this["_" + method];
            } else {
                delete global[method];
            }
        }

        // Prevent multiple executions which will completely remove these props
        this.methods = [];
    }

    function stubGlobal(method, clock) {
        clock[method].hadOwnProperty = Object.prototype.hasOwnProperty.call(global, method);
        clock["_" + method] = global[method];

        if (method == "Date") {
            var date = mirrorDateProperties(clock[method], global[method]);
            global[method] = date;
        } else {
            global[method] = function () {
                return clock[method].apply(clock, arguments);
            };

            for (var prop in clock[method]) {
                if (clock[method].hasOwnProperty(prop)) {
                    global[method][prop] = clock[method][prop];
                }
            }
        }

        global[method].clock = clock;
    }

    sinon.useFakeTimers = function useFakeTimers(now) {
        var clock = sinon.clock.create(now);
        clock.restore = restore;
        clock.methods = Array.prototype.slice.call(arguments,
                                                   typeof now == "number" ? 1 : 0);

        if (clock.methods.length === 0) {
            clock.methods = methods;
        }

        for (var i = 0, l = clock.methods.length; i < l; i++) {
            stubGlobal(clock.methods[i], clock);
        }

        return clock;
    };
}(typeof global != "undefined" && typeof global !== "function" ? global : this));

sinon.timers = {
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    setInterval: setInterval,
    clearInterval: clearInterval,
    Date: Date
};

if (typeof module == "object" && typeof require == "function") {
    module.exports = sinon;
}

/**
 * @depend fake_xml_http_request.js
 */
/*jslint eqeqeq: false, onevar: false, regexp: false, plusplus: false*/
/*global module, require, window*/
/**
 * The Sinon "server" mimics a web server that receives requests from
 * sinon.FakeXMLHttpRequest and provides an API to respond to those requests,
 * both synchronously and asynchronously. To respond synchronuously, canned
 * answers have to be provided upfront.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    var sinon = {};
}

sinon.fakeServer = (function () {
    var push = [].push;
    function F() {}

    function create(proto) {
        F.prototype = proto;
        return new F();
    }

    function responseArray(handler) {
        var response = handler;

        if (Object.prototype.toString.call(handler) != "[object Array]") {
            response = [200, {}, handler];
        }

        if (typeof response[2] != "string") {
            throw new TypeError("Fake server response body should be string, but was " +
                                typeof response[2]);
        }

        return response;
    }

    var wloc = typeof window !== "undefined" ? window.location : {};
    var rCurrLoc = new RegExp("^" + wloc.protocol + "//" + wloc.host);

    function matchOne(response, reqMethod, reqUrl) {
        var rmeth = response.method;
        var matchMethod = !rmeth || rmeth.toLowerCase() == reqMethod.toLowerCase();
        var url = response.url;
        var matchUrl = !url || url == reqUrl || (typeof url.test == "function" && url.test(reqUrl));

        return matchMethod && matchUrl;
    }

    function match(response, request) {
        var requestMethod = this.getHTTPMethod(request);
        var requestUrl = request.url;

        if (!/^https?:\/\//.test(requestUrl) || rCurrLoc.test(requestUrl)) {
            requestUrl = requestUrl.replace(rCurrLoc, "");
        }

        if (matchOne(response, this.getHTTPMethod(request), requestUrl)) {
            if (typeof response.response == "function") {
                var ru = response.url;
                var args = [request].concat(!ru ? [] : requestUrl.match(ru).slice(1));
                return response.response.apply(response, args);
            }

            return true;
        }

        return false;
    }

    return {
        create: function () {
            var server = create(this);
            this.xhr = sinon.useFakeXMLHttpRequest();
            server.requests = [];

            this.xhr.onCreate = function (xhrObj) {
                server.addRequest(xhrObj);
            };

            return server;
        },

        addRequest: function addRequest(xhrObj) {
            var server = this;
            push.call(this.requests, xhrObj);

            xhrObj.onSend = function () {
                server.handleRequest(this);
            };

            if (this.autoRespond && !this.responding) {
                setTimeout(function () {
                    server.responding = false;
                    server.respond();
                }, this.autoRespondAfter || 10);

                this.responding = true;
            }
        },

        getHTTPMethod: function getHTTPMethod(request) {
            if (this.fakeHTTPMethods && /post/i.test(request.method)) {
                var matches = (request.requestBody || "").match(/_method=([^\b;]+)/);
                return !!matches ? matches[1] : request.method;
            }

            return request.method;
        },

        handleRequest: function handleRequest(xhr) {
            if (xhr.async) {
                if (!this.queue) {
                    this.queue = [];
                }

                push.call(this.queue, xhr);
            } else {
                this.processRequest(xhr);
            }
        },

        respondWith: function respondWith(method, url, body) {
            if (arguments.length == 1 && typeof method != "function") {
                this.response = responseArray(method);
                return;
            }

            if (!this.responses) { this.responses = []; }

            if (arguments.length == 1) {
                body = method;
                url = method = null;
            }

            if (arguments.length == 2) {
                body = url;
                url = method;
                method = null;
            }

            push.call(this.responses, {
                method: method,
                url: url,
                response: typeof body == "function" ? body : responseArray(body)
            });
        },

        respond: function respond() {
            if (arguments.length > 0) this.respondWith.apply(this, arguments);
            var queue = this.queue || [];
            var request;

            while(request = queue.shift()) {
                this.processRequest(request);
            }
        },

        processRequest: function processRequest(request) {
            try {
                if (request.aborted) {
                    return;
                }

                var response = this.response || [404, {}, ""];

                if (this.responses) {
                    for (var i = 0, l = this.responses.length; i < l; i++) {
                        if (match.call(this, this.responses[i], request)) {
                            response = this.responses[i].response;
                            break;
                        }
                    }
                }

                if (request.readyState != 4) {
                    request.respond(response[0], response[1], response[2]);
                }
            } catch (e) {
                sinon.logError("Fake server request processing", e);
            }
        },

        restore: function restore() {
            return this.xhr.restore && this.xhr.restore.apply(this.xhr, arguments);
        }
    };
}());

if (typeof module == "object" && typeof require == "function") {
    module.exports = sinon;
}

/**
 * @depend fake_server.js
 * @depend fake_timers.js
 */
/*jslint browser: true, eqeqeq: false, onevar: false*/
/*global sinon*/
/**
 * Add-on for sinon.fakeServer that automatically handles a fake timer along with
 * the FakeXMLHttpRequest. The direct inspiration for this add-on is jQuery
 * 1.3.x, which does not use xhr object's onreadystatehandler at all - instead,
 * it polls the object for completion with setInterval. Dispite the direct
 * motivation, there is nothing jQuery-specific in this file, so it can be used
 * in any environment where the ajax implementation depends on setInterval or
 * setTimeout.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2011 Christian Johansen
 */
"use strict";

(function () {
    function Server() {}
    Server.prototype = sinon.fakeServer;

    sinon.fakeServerWithClock = new Server();

    sinon.fakeServerWithClock.addRequest = function addRequest(xhr) {
        if (xhr.async) {
            if (typeof setTimeout.clock == "object") {
                this.clock = setTimeout.clock;
            } else {
                this.clock = sinon.useFakeTimers();
                this.resetClock = true;
            }

            if (!this.longestTimeout) {
                var clockSetTimeout = this.clock.setTimeout;
                var clockSetInterval = this.clock.setInterval;
                var server = this;

                this.clock.setTimeout = function (fn, timeout) {
                    server.longestTimeout = Math.max(timeout, server.longestTimeout || 0);

                    return clockSetTimeout.apply(this, arguments);
                };

                this.clock.setInterval = function (fn, timeout) {
                    server.longestTimeout = Math.max(timeout, server.longestTimeout || 0);

                    return clockSetInterval.apply(this, arguments);
                };
            }
        }

        return sinon.fakeServer.addRequest.call(this, xhr);
    };

    sinon.fakeServerWithClock.respond = function respond() {
        var returnVal = sinon.fakeServer.respond.apply(this, arguments);

        if (this.clock) {
            this.clock.tick(this.longestTimeout || 0);
            this.longestTimeout = 0;

            if (this.resetClock) {
                this.clock.restore();
                this.resetClock = false;
            }
        }

        return returnVal;
    };

    sinon.fakeServerWithClock.restore = function restore() {
        if (this.clock) {
            this.clock.restore();
        }

        return sinon.fakeServer.restore.apply(this, arguments);
    };
}());

if (typeof module === "object" && typeof require === "function") {
    var buster = require("buster-core");
}

buster.stackFilter = function (stack, cwd) {
    var lines = (stack || "").split("\n");
    var stackLines = [], line, replacer = "./";

    if (typeof cwd == "string") {
        cwd = cwd.replace(/\/?$/, "/");
    }

    if (cwd instanceof RegExp && !/\/\/$/.test(cwd)) {
        replacer = ".";
    }

    for (var i = 0, l = lines.length; i < l; ++i) {
        if (/(\d+)?:\d+\)?$/.test(lines[i])) {
            if (!buster.stackFilter.match(lines[i])) {
                line = lines[i].replace(/^\s+|\s+$/g, "");

                if (cwd) {
                    line = line.replace(cwd, replacer);
                }

                stackLines.push(line);
            }
        }
    }

    return stackLines;
};

var regexpes = {};

buster.stackFilter.match = function (line) {
    var filters = buster.stackFilter.filters;

    for (var i = 0, l = filters.length; i < l; ++i) {
        if (!regexpes[filters[i]]) {
            regexpes[filters[i]] = new RegExp(filters[i]);
        }

        if (regexpes[filters[i]].test(line)) {
            return true;
        }
    }

    return false;
}

buster.stackFilter.filters = ["buster-assertions/lib",
                              "buster-test/lib", 
                              "buster-util/lib",
                              "buster-core/lib",
                              "node.js",
                              "/buster/lib",
                              "/buster/node_modules",
                              "static/runner.js"/* JsTestDriver */];

if (typeof module != "undefined") {
    module.exports = buster.stackFilter;
}

if (typeof module === "object" && typeof require === "function") {
    var buster = require("buster-core");
    var when = require("when");
}

buster.testContext = (function () {
    var bctx = buster.eventEmitter.create();

    function empty(context) {
        return context.tests.length == 0 &&
            context.contexts.length == 0;
    }

    function filterContexts(contexts, filter, prefix) {
        return reduce(contexts, [], function (filtered, context) {
            var ctx = bctx.filter(context, filter, prefix);
            if (ctx.tests.length > 0 || ctx.contexts.length > 0) {
                filtered.push(ctx);
            }
            return filtered;
        });
    }

    function filterTests(tests, filter, prefix) {
        return reduce(tests, [], function (filtered, test) {
            if (!filter || filter.test(prefix + test.name)) {
                filtered.push(test);
            }
            return filtered;
        });
    }

    function makeFilter(filter) {
        if (typeof filter == "string") return new RegExp(filter, "i");
        if (Object.prototype.toString.call(filter) != "[object Array]") return filter;

        return {
            test: function (string) {
                return filter.length == 0 || buster.some(filter, function (f) {
                    return new RegExp(f).test(string);
                });
            }
        };
    }

    function parse(context) {
        if (!context.tests && typeof context.parse == "function") {
            return context.parse();
        }
        return context;
    }

    function compile(contexts, filter) {
        return reduce(contexts, [], function (compiled, ctx) {
            if (when.isPromise(ctx)) {
                var deferred = when.defer();
                ctx.then(function (context) {
                    deferred.resolve(bctx.filter(parse(context), filter));
                });
                compiled.push(deferred.promise);
            } else {
                ctx = bctx.filter(parse(ctx), filter);
                if (!empty(ctx)) compiled.push(ctx);
            }
            return compiled;
        });
    }

    function filter(context, filter, name) {
        filter = makeFilter(filter);
        name = (name || "") + context.name + " ";

        return buster.extend({}, context, {
            tests: filterTests(context.tests || [], filter, name),
            contexts: filterContexts(context.contexts || [], filter, name)
        });
    }

    function reduce(arr, acc, fn) {
        if (arr.reduce) { return arr.reduce(fn, acc); }
        for (var i = 0, l = arr.length; i < l; ++i) {
            acc = fn.call(null, acc, arr[i], i, arr);
        }
        return acc;
    }

    bctx.compile = compile;
    bctx.filter = filter;
    return bctx;
}());

if (typeof module == "object") {
    module.exports = buster.testContext;
}

(function (B, when) {
    var testContext = B && B.testContext;

    if (typeof require == "function" && typeof module == "object") {
        B = require("buster-core");
        when = require("when");
        testContext = require("./test-context");
    }

    var current = [];
    var bspec = {};
    var bddNames = { contextSetUp: "beforeAll", contextTearDown: "afterAll" };

    function supportRequirement(property) {
        return function (requirements) {
            return {
                describe: function () {
                    var context = bspec.describe.apply(bspec, arguments);
                    context[property] = requirements;
                    return context;
                }
            };
        };
    }

    bspec.ifAllSupported = supportRequirement("requiresSupportForAll");
    bspec.ifAnySupported = supportRequirement("requiresSupportForAny");
    bspec.ifSupported = bspec.ifAllSupported;

    function addContext(parent, name, spec) {
        var context = bspec.describe.context.create(name, spec, parent).parse();
        parent.contexts.push(context);
        return context;
    }

    function createContext(name, spec) {
        return bspec.describe.context.create(name, spec).parse();
    }

    function asyncContext(name, callback) {
        var d = when.defer();
        callback(function (spec) {
            d.resolver.resolve(createContext(name, spec));
        });
        d.promise.name = "deferred " + name;
        testContext.emit("create", d.promise);
        return d.promise;
    }

    var FOCUS_ROCKET = /^\s*=>\s*/;

    function markFocused(block, parent) {
        var focused = block.focused || (parent && parent.forceFocus);
        block.focused = focused || FOCUS_ROCKET.test(block.name);
        block.name = block.name.replace(FOCUS_ROCKET, "");
        while (parent) {
            parent.focused = parent.focused || block.focused;
            parent = parent.parent;
        }
    }

    bspec.describe = function (name, spec) {
        if (current.length > 0) {
            return addContext(current[current.length - 1], name, spec);
        }
        if (spec && spec.length > 0) {
            return asyncContext(name, spec);
        }
        var context = createContext(name, spec);
        testContext.emit("create", context);
        return context;
    };

    B.extend(bspec.describe, B.eventEmitter);

    function markDeferred(spec, func) {
        spec.deferred = typeof func != "function";

        if (!spec.deferred && /^\/\//.test(spec.name)) {
            spec.deferred = true;
            spec.name = spec.name.replace(/^\/\/\s*/, "");
        }

        spec.comment = spec.deferred ? func : "";
    }

    bspec.it = function (name, func) {
        var context = current[current.length - 1];

        var spec = {
            name: name,
            func: arguments.length == 3 ? arguments[2] : func,
            context: context
        };

        markDeferred(spec, func);
        markFocused(spec, context);
        context.tests.push(spec);
        return spec;
    };

    bspec.itEventually = function (name, comment, func) {
        if (typeof comment == "function") {
            func = comment;
            comment = "";
        }

        return bspec.it(name, comment, func);
    };

    bspec.before = bspec.beforeEach = function (func) {
        var context = current[current.length - 1];
        context.setUp = func;
    };

    bspec.after = bspec.afterEach = function (func) {
        var context = current[current.length - 1];
        context.tearDown = func;
    };

    bspec.beforeAll = function (func) {
        var context = current[current.length - 1];
        context.contextSetUp = func;
    };

    bspec.afterAll = function (func) {
        var context = current[current.length - 1];
        context.contextTearDown = func;
    };

    bspec.describe.context = {
        create: function (name, spec, parent) {
            if (!name || typeof name != "string") {
                throw new Error("Spec name required");
            }

            if (!spec || typeof spec != "function") {
                throw new Error("spec should be a function");
            }

            var context = B.create(this);
            context.name = name;
            context.parent = parent;
            context.spec = spec;
            markFocused(context, parent);
            context.forceFocus = context.focused;

            return context;
        },

        parse: function () {
            if (!this.spec) {
                return this;
            }

            this.testCase = {
                before: bspec.before,
                beforeEach: bspec.beforeEach,
                beforeAll: bspec.beforeAll,
                after: bspec.after,
                afterEach: bspec.afterEach,
                afterAll: bspec.afterAll,
                it: bspec.it,
                itEventually: bspec.itEventually,
                describe: bspec.describe,
                name: function (thing) { return bddNames[thing] || thing; }
            };

            this.tests = [];
            current.push(this);
            this.contexts = [];
            this.spec.call(this.testCase);
            current.pop();
            delete this.spec;

            return this;
        }
    };

    var g = typeof global != "undefined" && global || this;

    bspec.expose = function (env) {
        env = env || g;
        env.describe = bspec.describe;
        env.it = bspec.it;
        env.itEventually = bspec.itEventually;
        env.beforeAll = bspec.beforeAll;
        env.before = bspec.before;
        env.beforeEach = bspec.beforeEach;
        env.afterAll = bspec.afterAll;
        env.after = bspec.after;
        env.afterEach = bspec.afterEach;
    };

    if (typeof module == "object") {
        module.exports = bspec;
    } else {
        B.spec = bspec;
    }
}(typeof buster !== "undefined" ? buster : {},
  typeof when === "function" ? when : function () {}));

(function (B, when) {
    var testContext = B && B.testContext;

    if (typeof require == "function" && typeof module == "object") {
        B = require("buster-core");
        when = require("when");
        testContext = require("./test-context");
    }

    var xUnitNames = { contextSetUp: "prepare", contextTearDown: "conclude" };

    var testCase = function (name, tests) {
        if (!name || typeof name != "string") {
            throw new Error("Test case name required");
        }

        if (!tests || (typeof tests != "object" && typeof tests != "function")) {
            throw new Error("Tests should be an object or a function");
        }

        var context = testCase.context.create(name, tests);
        var d = when.defer();
        when(context).then(function (ctx) { d.resolver.resolve(ctx.parse()); });
        var promise = context.then ? d.promise : context;
        B.testContext.emit("create", promise);
        return promise;
    };

    if (typeof module != "undefined") {
        module.exports = testCase;
    } else {
        B.testCase = testCase;
    }

    B.extend(testCase, B.eventEmitter);

    function nonTestNames(context) {
        return {
            prepare: true,
            conclude: true,
            setUp: true,
            tearDown: true,
            requiresSupportFor: true,
            requiresSupportForAll: true
        };
    }

    var DEFERRED_PREFIX = /^\s*\/\/\s*/;
    var FOCUSED_PREFIX = /^\s*=>\s*/;

    function createContext(context, name, tests, parent) {
        return B.extend(context, {
            name: name,
            content: tests,
            parent: parent,
            testCase: {
                name: function (thing) { return xUnitNames[thing] || thing; }
            }
        });
    }

    function asyncContext(context, name, callback, parent) {
        var d = when.defer();
        callback(function (tests) {
            d.resolver.resolve(createContext(context, name, tests, parent));
        });
        return d.promise;
    }

    testCase.context = {
        create: function (name, tests, parent) {
            var context = B.create(this);
            if (typeof tests == "function") {
                return asyncContext(context, name, tests, parent);
            }
            return createContext(context, name, tests, parent);
        },

        parse: function (forceFocus) {
            this.getSupportRequirements();
            this.deferred = DEFERRED_PREFIX.test(this.name);
            this.focused = forceFocus || FOCUSED_PREFIX.test(this.name);
            this.name = this.name.replace(DEFERRED_PREFIX, "").replace(FOCUSED_PREFIX, "");
            this.tests = this.getTests(this.focused);
            this.contexts = this.getContexts(this.focused);
            this.focused = this.focused || this.contexts.focused || this.tests.focused;
            delete this.tests.focused;
            delete this.contexts.focused;
            this.contextSetUp = this.getContextSetUp();
            this.contextTearDown = this.getContextTearDown();
            this.setUp = this.getSetUp();
            this.tearDown = this.getTearDown();
            return this;
        },

        getSupportRequirements: function () {
            this.requiresSupportForAll = this.content.requiresSupportForAll || this.content.requiresSupportFor;
            delete this.content.requiresSupportForAll;
            delete this.content.requiresSupportFor;
            this.requiresSupportForAny = this.content.requiresSupportForAny;
            delete this.content.requiresSupportForAny;
        },

        getTests: function (focused) {
            var tests = [], isFunc;

            for (var prop in this.content) {
                isFunc = typeof this.content[prop] == "function";
                if (this.isTest(prop)) {
                    var testFocused = focused || FOCUSED_PREFIX.test(prop);
                    tests.focused = tests.focused || testFocused;
                    tests.push({
                        name: prop.replace(DEFERRED_PREFIX, "").replace(FOCUSED_PREFIX, ""),
                        func: this.content[prop],
                        context: this,
                        deferred: this.deferred || DEFERRED_PREFIX.test(prop) || !isFunc,
                        focused: testFocused,
                        comment: !isFunc ? this.content[prop] : ""
                    });
                }
            }

            return tests;
        },

        getContexts: function (focused) {
            var contexts = [], ctx;
            contexts.focused = focused;

            for (var prop in this.content) {
                if (!this.isContext(prop)) { continue; }
                ctx = testCase.context.create(prop, this.content[prop], this);
                ctx = ctx.parse(focused);
                contexts.focused = contexts.focused || ctx.focused;
                contexts.push(ctx);
            }

            return contexts;
        },

        getContextSetUp: function () {
            return this.content.prepare;
        },

        getContextTearDown: function () {
            return this.content.conclude;
        },

        getSetUp: function () {
            return this.content.setUp;
        },

        getTearDown: function () {
            return this.content.tearDown;
        },

        isTest: function (prop) {
            var type = typeof this.content[prop];
            return this.content.hasOwnProperty(prop) &&
                (type == "function" || type == "string") &&
                !nonTestNames(this)[prop];
        },

        isContext: function (prop) {
            return this.content.hasOwnProperty(prop) &&
                typeof this.content[prop] == "object" &&
                !!this.content[prop];
        }
    };
}(typeof buster !== "undefined" ? buster : {},
  typeof when === "function" ? when : function () {}));

/*jslint maxlen: 88*/
var buster = buster || {};

(function (B, when) {
    var isNode = typeof require === "function" && typeof module === "object";
    var onUncaught = function () {};

    if (isNode) {
        B = require("buster-core");
        when = require("when");
    }

    var partial = B.partial, bind = B.bind;
    var each = B.each, map = B.map, series = B.series;

    // Events

    var errorEvents = {
        "TimeoutError": "test:timeout",
        "AssertionError": "test:failure",
        "DeferredTestError": "test:deferred"
    };

    function emit(runner, event, test, err, thisp) {
        var data = { name: test.name };
        if (err) { data.error = err; }
        if (typeof test.func === "string") { data.comment = test.func; }
        if (thisp) { data.testCase = thisp; }
        if (event === "test:success") { data.assertions = runner.assertionCount(); }
        runner.emit(event, data);
    }

    function emitTestAsync(runner, test) {
        if (test && !test.async && !test.deferred) {
            test.async = true;
            emit(runner, "test:async", test);
        }
    }

    function testResult(runner, test, err) {
        if (!test) { return runner.emit("uncaughtException", err); }
        if (test.complete) { return; }
        test.complete = true;
        var event = "test:success";

        if (err) {
            event = errorEvents[err.name] || "test:error";
            if (err.name == "TimeoutError") { emitTestAsync(runner, test); }
        }

        emit(runner, event, test, err);

        if (event === "test:error") { runner.results.errors += 1; }
        if (event === "test:failure") { runner.results.failures += 1; }
        if (event === "test:timeout") { runner.results.timeouts += 1; }
        if (event === "test:deferred") {
            runner.results.deferred += 1;
        } else {
            runner.results.assertions += runner.assertionCount();
            runner.results.tests += 1;
        }
    }

    function emitIfAsync(runner, test, isAsync) {
        if (isAsync) { emitTestAsync(runner, test); }
    }

    function emitUnsupported(runner, context, requirements) {
        runner.emit("context:unsupported", {
            context: context,
            unsupported: requirements
        });
    }

    // Data helper functions
    function byRandom() {
        return Math.round(Math.random() * 2) - 1;
    }

    function tests(context) {
        return context.tests.sort(byRandom);
    }

    function setUps(context) {
        var setUpFns = [];
        while (context) {
            if (context.setUp) {
                setUpFns.unshift(context.setUp);
            }
            context = context.parent;
        }
        return setUpFns;
    }

    function tearDowns(context) {
        var tearDownFns = [];
        while (context) {
            if (context.tearDown) {
                tearDownFns.push(context.tearDown);
            }
            context = context.parent;
        }
        return tearDownFns;
    }

    function satiesfiesRequirement(requirement) {
        if (typeof requirement === "function") {
            return !!requirement();
        }
        return !!requirement;
    }

    function unsatiesfiedRequirements(context) {
        var name, requirements = context.requiresSupportForAll || {};
        for (name in requirements) {
            if (!satiesfiesRequirement(requirements[name])) {
                return [name];
            }
        }
        var unsatiesfied = [];
        requirements = context.requiresSupportForAny || {};
        for (name in requirements) {
            if (satiesfiesRequirement(requirements[name])) {
                return [];
            } else {
                unsatiesfied.push(name);
            }
        }
        return unsatiesfied;
    }

    function isAssertionError(err) {
        return err && err.name === "AssertionError";
    }

    function prepareResults(results) {
        return B.extend(results, {
            ok: results.failures + results.errors + results.timeouts === 0
        });
    }

    function propWithDefault(obj, prop, defaultValue) {
        return obj && obj.hasOwnProperty(prop) ? obj[prop] : defaultValue;
    }

    // Async flow

    function promiseSeries(objects, fn) {
        var deferred = when.defer();
        B.series(map(objects, function (obj) {
            return function () {
                return fn(obj);
            };
        }), function (err) {
            if (err) {
                return deferred.resolver.reject(err);
            }
            deferred.resolver.resolve();
        });
        return deferred.promise;
    }

    function asyncDone(resolver) {
        function resolve(method, err) {
            try {
                resolver[method](err);
            } catch (e) {
                throw new Error("done() was already called");
            }
        }

        return function (fn) {
            if (typeof fn != "function") { return resolve("resolve"); }
            return function () {
                try {
                    var retVal = fn.apply(this, arguments);
                    resolve("resolve");
                    return retVal;
                } catch (up) {
                    resolve("reject", up);
                }
            };
        };
    }

    function asyncFunction(fn, thisp) {
        if (fn.length > 0) {
            var deferred = when.defer();
            fn.call(thisp, asyncDone(deferred.resolver));
            return deferred.promise;
        }
        return fn.call(thisp);
    }

    function timeoutError(ms) {
        return {
            name: "TimeoutError",
            message: "Timed out after " + ms + "ms"
        };
    }

    function timebox(promise, timeout, callbacks) {
        var timedout, complete, timer;
        function handler(method) {
            return function () {
                complete = true;
                clearTimeout(timer);
                if (!timedout) { callbacks[method].apply(this, arguments); }
            };
        }
        when(promise).then(handler("resolve"), handler("reject"));
        var ms = typeof timeout === "function" ? timeout() : timeout;
        timer = setTimeout(function () {
            timedout = true;
            if (!complete) { callbacks.timeout(timeoutError(ms)); }
        }, ms);
    }

    function callAndWait(func, thisp, timeout, next) {
        var reject = function (err) { next(err || {}); };
        var promise = asyncFunction(func, thisp);
        timebox(promise, timeout, {
            resolve: partial(next, null),
            reject: reject,
            timeout: reject
        });
        return promise;
    }

    function callSerially(functions, thisp, timeout, source) {
        var d = when.defer();
        var fns = functions.slice();
        var isAsync = false;
        function next(err) {
            if (err) {
                err.source = source;
                return d.resolver.reject(err);
            }
            if (fns.length === 0) { return d.resolver.resolve(isAsync); }
            try {
                var promise = callAndWait(fns.shift(), thisp, timeout, next);
                isAsync = isAsync || when.isPromise(promise);
            } catch (e) {
                return d.resolver.reject(e);
            }
        }
        next();
        return d.promise;
    }

    function asyncWhen(value) {
        if (when.isPromise(value)) {
            return value;
        } else {
            var d = when.defer();
            B.nextTick(partial(d.resolver.resolve, value));
            return d.promise;
        }
    }

    function chainPromises(fn, resolution) {
        var r = typeof resolution === "function" ?
            [resolution, resolution] : resolution;
        return function () {
            fn().then(partial(resolution, null), r[0], r[1]);
        };
    }

    function rejected(deferred) {
        if (!deferred) {
            deferred = when.defer();
        }
        deferred.resolver.reject();
        return deferred.promise;
    }

    function listenForUncaughtExceptions() {
        var listener, listening = false;
        onUncaught = function (l) {
            listener = l;

            if (!listening) {
                listening = true;
                process.on("uncaughtException", function (e) {
                    if (listener) { listener(e); }
                });
            }
        };
    }

    // Private runner functions

    function callTestFn(runner, test, thisp, next) {
        emit(runner, "test:start", test, null, thisp);
        if (test.deferred) { return next({ name: "DeferredTestError" }); }

        try {
            var promise = asyncFunction(test.func, thisp);
            if (when.isPromise(promise)) { emitTestAsync(runner, test); }
            timebox(promise, thisp.timeout || runner.timeout, {
                resolve: next,
                reject: next,
                timeout: function (err) {
                    err.source = "test function";
                    next(err);
                }
            });
        } catch (e) {
            next(e);
        }
    }

    function checkAssertions(runner, expected) {
        if (runner.failOnNoAssertions && runner.assertionCount() === 0) {
            return { name: "AssertionError", message: "No assertions!" };
        }
        var actual = runner.assertionCount();
        if (typeof expected === "number" && actual !== expected) {
            return {
                name: "AssertionError",
                message: "Expected " + expected + " assertions, ran " + actual
            };
        }
    }

    function triggerOnCreate(listeners, runner) {
        each(listeners, function (listener) {
            listener(runner);
        });
    }

    function initializeResults() {
        return {
            contexts: 0,
            tests: 0,
            errors: 0,
            failures: 0,
            assertions: 0,
            timeouts: 0,
            deferred: 0
        };
    }

    function focused(items) {
        return B.filter(items, function (item) { return item.focused; });
    }

    function dynamicTimeout(testCase, runner) {
        return function () {
            return testCase.timeout || runner.timeout;
        };
    }

    var testRunner = B.extend(B.eventEmitter.create(), {
        timeout: 250,
        onCreateListeners: [],

        create: function (opt) {
            var runner = B.create(this);
            triggerOnCreate(this.onCreateListeners, runner);
            runner.results = initializeResults();
            var instance = B.extend(runner, {
                failOnNoAssertions: propWithDefault(opt, "failOnNoAssertions", false)
            });
            if (opt && typeof opt.timeout === "number") {
                instance.timeout = opt.timeout;
            }
            return instance;
        },

        onCreate: function (listener) {
            this.onCreateListeners.push(listener);
        },

        runSuite: function (contexts) {
            this.focusMode = B.some(contexts, function (c) { return c.focused; });
            this.results = initializeResults();
            onUncaught(bind(this, function (err) {
                testResult(this, this.currentTest, err);
            }));
            var d = when.defer();
            this.emit("suite:start");
            if (this.focusMode) { this.emit("runner:focus"); }
            this.results.contexts = contexts.length;
            this.runContexts(contexts).then(bind(this, function () {
                var res = prepareResults(this.results);
                this.emit("suite:end", res);
                d.resolver.resolve(res);
            }), d.resolver.reject);
            return d.promise;
        },

        runContexts: function (contexts) {
            if (this.focusMode) { contexts = focused(contexts); }
            return promiseSeries((contexts || []).sort(byRandom),
                                 bind(this, "runContext"));
        },

        runContext: function (context) {
            if (!context) { return rejected(); }
            var reqs = unsatiesfiedRequirements(context);
            if (reqs.length > 0) {
                return when(emitUnsupported(this, context, reqs));
            }
            var d = when.defer(), s = this, thisp, ctx;
            var emitAndResolve = function () {
                s.emit("context:end", context);
                d.resolver.resolve();
            };
            var end = function (err) {
                s.runContextUpDown(ctx, "contextTearDown", thisp).then(
                    emitAndResolve,
                    emitAndResolve
                );
            };
            this.emit("context:start", context);
            asyncWhen(context).then(function (c) {
                ctx = c;
                thisp = B.create(c.testCase);
                var runTests = chainPromises(
                    bind(s, "runTests", tests(c), setUps(c), tearDowns(c), thisp),
                    end
                );
                s.runContextUpDown(ctx, "contextSetUp", thisp).then(function () {
                    s.runContexts(c.contexts).then(runTests);
                }, end);
            });
            return d;
        },

        runContextUpDown: function (context, prop, thisp) {
            var fn = context[prop];
            if (!fn) { return when(); }
            var d = when.defer();
            var s = this;
            var reject = function (err) {
                err = err || new Error();
                err.message = context.name + " " + thisp.name(prop)+ "(n) "+
                     (/Timeout/.test(err.name) ?
                     "timed out" : "failed") + ": " + err.message;
                s.emit("uncaughtException", err);
                d.reject(err);
            };
            try {
                var timeout = dynamicTimeout(thisp, this);
                timebox(asyncFunction(fn, thisp), timeout, {
                    resolve: d.resolve,
                    reject: reject,
                    timeout: reject
                });
            } catch (e) {
                reject(e);
            }
            return d.promise;
        },

        callSetUps: function (test, setUps, thisp) {
            if (test.deferred) { return when(); }
            emit(this, "test:setUp", test, null, thisp);
            var timeout = dynamicTimeout(thisp, this);
            var emitAsync = partial(emitIfAsync, this, test);
            return callSerially(setUps, thisp, timeout, "setUp").then(emitAsync);
        },

        callTearDowns: function (test, tearDowns, thisp) {
            if (test.deferred) { return when(); }
            emit(this, "test:tearDown", test, null, thisp);
            var timeout = dynamicTimeout(thisp, this);
            var emitAsync = partial(emitIfAsync, this, test);
            return callSerially(tearDowns, thisp, timeout, "tearDown").then(emitAsync);
        },

        runTests: function (tests, setUps, tearDowns, thisp) {
            if (this.focusMode) { tests = focused(tests); }
            return promiseSeries(tests, bind(this, function (test) {
                return this.runTest(test, setUps, tearDowns, B.create(thisp));
            }));
        },

        runTest: function (test, setUps, tearDowns, thisp) {
            this.running = true;
            var d = when.defer();
            test = B.create(test);
            this.currentTest = test;
            var callSetUps = bind(this, "callSetUps", test, setUps, thisp);
            var callTearDowns = bind(this, "callTearDowns", test, tearDowns, thisp);
            var callTest = partial(callTestFn, this, test, thisp);
            var tearDownEmitAndResolve = bind(this, function (err) {
                var resolution = bind(this, function (err2) {
                    var e = err || err2 || this.queued;
                    this.running = false;
                    this.queued = null;
                    e = e || checkAssertions(this, thisp.expectedAssertions);
                    testResult(this, test, e);
                    delete this.currentTest;
                    d.resolver.resolve();
                });
                callTearDowns().then(partial(resolution, null), resolution);
            });
            var callTestAndTearDowns = partial(callTest, tearDownEmitAndResolve);
            callSetUps().then(callTestAndTearDowns, tearDownEmitAndResolve);
            return d.promise;
        },

        assertionCount: function () {
            return 0;
        },

        error: function (error, test) {
            if (this.running) {
                if (!this.queued) {
                    this.queued = error;
                }
                return;
            }
            testResult(this, test || this.currentTest, error);
        },

        // To be removed
        assertionFailure: function (error) {
            this.error(error);
        }
    });

    // Export module

    if (isNode) {
        listenForUncaughtExceptions();
        module.exports = testRunner;
    } else {
        B.testRunner = testRunner;
    }
}(buster, typeof when === "function" ? when : function () {}));

if (typeof module === "object" && typeof require === "function") {
    module.exports = {
        specification: require("./reporters/specification"),
        jsonProxy: require("./reporters/json-proxy"),
        quiet: require("./reporters/quiet"),
        xml: require("./reporters/xml"),
        tap: require("./reporters/tap"),
        dots: require("./reporters/dots"),
        html: require("./reporters/html"),
        teamcity: require("./reporters/teamcity"),

        load: function (reporter) {
            if (module.exports[reporter]) {
                return module.exports[reporter];
            }

            return require(reporter);
        }
    };
} else {
    buster.reporters = buster.reporters || {};
    buster.reporters.load = function (reporter) {
        return buster.reporters[reporter];
    };
}
(function () {
    var isNodeJS = typeof module === "object" && typeof require === "function";

    if (isNodeJS) {
        buster = require("buster-core");
        buster.stackFilter = require("../stack-filter");
        var util = require("util");

        try {
            var jsdom = require("jsdom").jsdom;
        } catch (e) {
            // Is handled when someone actually tries using the HTML reporter
            // on node without jsdom
        }
    }

    var htmlReporter = {
        create: function (opt) {
            var reporter = buster.create(this);
            opt = opt || {};
            reporter.contexts = [];
            reporter.doc = getDoc(opt);
            reporter.setRoot(opt.root || reporter.doc.body);
            reporter.io = opt.io || (isNodeJS && require("util"));

            return reporter;
        },

        setRoot: function (root) {
            this.root = root;
            this.root.className += " buster-test";
            var body = this.doc.body;

            if (this.root == body) {
                var head = this.doc.getElementsByTagName("head")[0];
                head.parentNode.className += " buster-test";

                head.appendChild(el(this.doc, "meta", {
                    "name": "viewport",
                    "content": "width=device-width, initial-scale=1.0"
                }));

                head.appendChild(el(this.doc, "meta", {
                    "http-equiv": "Content-Type",
                    "content": "text/html; charset=utf-8"
                }));

                addCSS(head);
                insertTitle(this.doc, body, this.doc.title || "Buster.JS Test case");
                insertLogo(this.doc.getElementsByTagName("h1")[0]);
            }
        },

        listen: function (runner) {
            runner.bind(this, {
                "context:start": "contextStart", "context:end": "contextEnd",
                "test:success": "testSuccess", "test:failure": "testFailure",
                "test:error": "testError", "test:timeout": "testTimeout",
                "test:deferred": "testDeferred", "suite:end": "addStats"
            });

            if (runner.console) {
                runner.console.bind(this, "log");
            }

            return this;
        },

        contextStart: function (context) {
            if (this.contexts.length == 0) {
                this.root.appendChild(el(this.doc, "h2", { text: context.name }));
            }

            this.startedAt = new Date();
            this.contexts.push(context.name);
        },

        contextEnd: function (context) {
            this.contexts.pop();
            this._list = null;
        },

        testSuccess: function (test) {
            var li = addListItem.call(this, "h3", test, "success");
            this.addMessages(li);
        },

        testFailure: function (test) {
            var li = addListItem.call(this, "h3", test, "failure");
            this.addMessages(li);
            addException(li, test.error);
        },

        testError: function (test) {
            var li = addListItem.call(this, "h3", test, "error");
            this.addMessages(li);
            addException(li, test.error);
        },

        testDeferred: function (test) {
            var li = addListItem.call(this, "h3", test, "deferred");
        },

        testTimeout: function (test) {
            var li = addListItem.call(this, "h3", test, "timeout");
            var source = test.error && test.error.source;
            if (source) {
                li.firstChild.innerHTML += " (" + source + " timed out)";
            }
            this.addMessages(li);
        },

        log: function (msg) {
            this.messages = this.messages || [];
            this.messages.push(msg);
        },

        addMessages: function (li) {
            var messages = this.messages || [];
            var html = "";

            if (messages.length == 0) {
                return;
            }

            for (var i = 0, l = messages.length; i < l; ++i) {
                html += "<li class=\"" + messages[i].level + "\">";
                html += messages[i].message + "</li>";
            }

            li.appendChild(el(this.doc, "ul", {
                className: "messages",
                innerHTML: html
            }));

            this.messages = [];
        },

        success: function (stats) {
            return stats.failures == 0 && stats.errors == 0 &&
                stats.tests > 0 && stats.assertions > 0;
        },

        addStats: function (stats) {
            var diff = (new Date() - this.startedAt) / 1000;

            var className = "stats " + (this.success(stats) ? "success" : "failure");
            var statsEl = el(this.doc, "div", { className: className });

            var h1 = this.doc.getElementsByTagName("h1")[0];
            this.root.insertBefore(statsEl, h1.nextSibling);

            statsEl.appendChild(el(this.doc, "h2", {
                text: this.success(stats) ? "Tests OK" : "Test failures!"
            }));

            var html = "";
            html += "<li>" + pluralize(stats.contexts, "test case") + "</li>";
            html += "<li>" + pluralize(stats.tests, "test") + "</li>";
            html += "<li>" + pluralize(stats.assertions, "assertion") + "</li>";
            html += "<li>" + pluralize(stats.failures, "failure") + "</li>";
            html += "<li>" + pluralize(stats.errors, "error") + "</li>";
            html += "<li>" + pluralize(stats.timeouts, "timeout") + "</li>";

            if (stats.deferred > 0) {
                html += "<li>" + stats.deferred + " deferred</li>";
            }

            statsEl.appendChild(el(this.doc, "ul", { innerHTML: html }));
            statsEl.appendChild(el(this.doc, "p", {
                className: "time",
                innerHTML: "Finished in " + diff + "s"
            }));

            this.writeIO();
        },

        list: function () {
            if (!this._list) {
                this._list = el(this.doc, "ul", { className: "test-results" });
                this.root.appendChild(this._list);
            }

            return this._list;
        },

        writeIO: function () {
            if (!this.io) return;
            this.io.puts(this.doc.doctype.toString());
            this.io.puts(this.doc.innerHTML);
        }
    };

    function getDoc(options) {
        return options && options.document ||
            (typeof document != "undefined" ? document : createDocument());
    }

    function addCSS(head) {
        if (isNodeJS) {
            var fs = require("fs");
            var path = require("path");

            head.appendChild(el(head.ownerDocument, "style", {
                type: "text/css",
                innerHTML: fs.readFileSync(path.join(__dirname, "../../../resources/buster-test.css"))
            }));
        } else {
            head.appendChild(el(document, "link", {
                rel: "stylesheet",
                type: "text/css",
                media: "all",
                href: busterTestPath() + "buster-test.css"
            }));
        }
    }

    function insertTitle(doc, body, title) {
        if (doc.getElementsByTagName("h1").length == 0) {
            body.insertBefore(el(doc, "h1", {
                innerHTML: "<span class=\"title\">" + title + "</span>"
            }), body.firstChild);
        }
    }

    function insertLogo(h1) {
        h1.innerHTML = "<span class=\"buster-logo\"></span>" + h1.innerHTML;
    }

    function createDocument() {
        if (!jsdom) {
            util.puts("Unable to load jsdom, html reporter will not work " +
                      "for node runs. Spectacular fail coming up.");
        }
        var dom = jsdom("<!DOCTYPE html><html><head></head><body></body></html>");
        return dom.createWindow().document;
    }

    function pluralize(num, phrase) {
        num = typeof num == "undefined" ? 0 : num;
        return num + " " + (num == 1 ? phrase : phrase + "s");
    }

    function el(doc, tagName, properties) {
        var el = doc.createElement(tagName), value;

        for (var prop in properties) {
            value = properties[prop];

            if (prop == "http-equiv") {
                el.setAttribute(prop, value);
            }

            if (prop == "text") {
                prop = "innerHTML";
            }

            el[prop] = value;
        }

        return el;
    }

    function addListItem(tagName, test, className) {
        var prefix = tagName ? "<" + tagName + ">" : "";
        var suffix = tagName ? "</" + tagName + ">" : "";
        var name = this.contexts.slice(1).join(" ") + " " + test.name;

        var item = el(this.doc, "li", {
            className: className,
            text: prefix + name.replace(/^\s+|\s+$/, "") + suffix
        });

        this.list().appendChild(item);
        return item;
    }

    function addException(li, error) {
        if (!error) {
            return;
        }

        var name = error.name == "AssertionError" ? "" : error.name + ": ";

        li.appendChild(el(li.ownerDocument || document, "p", {
            innerHTML: name + error.message,
            className: "error-message"
        }));

        var stack = buster.stackFilter(error.stack) || [];

        if (stack.length > 0) {
            if (stack[0].indexOf(error.message) >= 0) {
                stack.shift();
            }

            li.appendChild(el(li.ownerDocument || document, "ul", {
                className: "stack",
                innerHTML: "<li>" + stack.join("</li><li>") + "</li>"
            }));
        }
    }

    function busterTestPath() {
        var scripts = document.getElementsByTagName("script");

        for (var i = 0, l = scripts.length; i < l; ++i) {
            if (/buster-test\.js$/.test(scripts[i].src)) {
                return scripts[i].src.replace("buster-test.js", "");
            }
        }

        return "";
    }

    if (typeof module == "object" && module.exports) {
        module.exports = htmlReporter;
    } else {
        buster.reporters = buster.reporters || {};
        buster.reporters.html = htmlReporter;
    }
}());

if (typeof module === "object" && typeof require === "function") {
    var buster = require("buster-core");
    buster.testRunner = require("./test-runner");
    buster.reporters = require("./reporters");
    buster.testContext = require("./test-context");
}

(function () {
    function env() {
        return (typeof process !== "undefined" && process.env) || {};
    }

    buster.autoRun = function (opt, callbacks) {
        var runners = 0, contexts = [], timer;

        buster.testRunner.onCreate(function (runner) {
            runners += 1;
        });

        if (typeof opt === "function") {
            callbacks = opt;
            opt = {};
        }

        if (typeof callbacks !== "object") {
            callbacks = { end: callbacks };
        }

        return function (tc) {
            contexts.push(tc);
            clearTimeout(timer);

            timer = setTimeout(function () {
                if (runners === 0) {
                    opt = buster.extend(buster.autoRun.envOptions(env()), opt);
                    buster.autoRun.run(contexts, opt, callbacks);
                }
            }, 10);
        };
    };

    buster.autoRun.envOptions = function (env) {
        return {
            reporter: env.BUSTER_REPORTER,
            filters: (env.BUSTER_FILTERS || "").split(","),
            color: env.BUSTER_COLOR === "false" ? false : true,
            bright: env.BUSTER_BRIGHT === "false" ? false : true,
            timeout: env.BUSTER_TIMEOUT && parseInt(env.BUSTER_TIMEOUT, 10),
            failOnNoAssertions: env.BUSTER_FAIL_ON_NO_ASSERTIONS === "false" ?
                false : true
        };
    };

    function initializeReporter(runner, opt) {
        var reporter;

        if (typeof document !== "undefined" && document.getElementById) {
            reporter = "html";
            opt.root = document.getElementById("buster") || document.body;
        } else {
            reporter = opt.reporter || "dots";
        }

        reporter = buster.reporters.load(reporter).create(opt);
        reporter.listen(runner);

        if (typeof reporter.log === "function" &&
                typeof buster.console === "function") {
            buster.console.bind(reporter, ["log"]);
        }
    }

    buster.autoRun.run = function (contexts, opt, callbacks) {
        callbacks = callbacks || {};
        if (contexts.length === 0) { return; }
        opt = buster.extend({ color: true, bright: true }, opt);

        var runner = buster.testRunner.create(buster.extend({
            timeout: 750,
            failOnNoAssertions: false
        }, opt));

        if (typeof callbacks.start === "function") {
            callbacks.start(runner);
        }

        initializeReporter(runner, opt);

        if (typeof callbacks.end === "function") {
            runner.on("suite:end", callbacks.end);
        }

        runner.runSuite(buster.testContext.compile(contexts, opt.filters));
    };
}());

if (typeof module !== "undefined") {
    module.exports = buster.autoRun;
}

if (typeof module === "object" && typeof require === "function") {
    module.exports = {
        testCase: require("./buster-test/test-case"),
        spec: require("./buster-test/spec"),
        testRunner: require("./buster-test/test-runner"),
        testContext: require("./buster-test/test-context"),
        reporters: require("./buster-test/reporters"),
        autoRun: require("./buster-test/auto-run"),
        stackFilter: require("./buster-test/stack-filter")
    };
}

/*jslint onevar: false, eqeqeq: false*/
/*global require*/
(function (buster, sinon) {
    var ba, testRunner, stackFilter, format;

    if (typeof require == "function" && typeof module == "object") {
        sinon = require("sinon");
        buster = require("buster-core");
        ba = require("buster-assertions");
        format = require("buster-format");
        testRunner = require("buster-test").testRunner;
        stackFilter = require("buster-test").stackFilter;
    } else {
        ba = buster.assertions;
        format = buster.format;
        testRunner = buster.testRunner;
        stackFilter = buster.stackFilter;
    }

    if (stackFilter && stackFilter.filters) {
        stackFilter.filters.push("lib/sinon");
    }

    sinon.expectation.pass = function (assertion) {
        ba.emit("pass", assertion);
    };

    sinon.expectation.fail = function (message) {
        ba.fail(message);
    };

    if (testRunner) {
        testRunner.onCreate(function (runner) {
            runner.on("test:setUp", function (test) {
                var config = sinon.getConfig(sinon.config);
                config.useFakeServer = false;
                var sandbox = sinon.sandbox.create();
                sandbox.inject(test.testCase);

                test.testCase.useFakeTimers = function () {
                    return sandbox.useFakeTimers.apply(sandbox, arguments);
                };

                test.testCase.useFakeServer = function () {
                    return sandbox.useFakeServer.apply(sandbox, arguments);
                };

                test.testCase.sandbox = sandbox;
                var testFunc = test.func;
            });

            runner.on("test:tearDown", function (test) {
                try {
                    test.testCase.sandbox.verifyAndRestore();
                } catch (e) {
                    runner.assertionFailure(e);
                }
            });
        });
    }

    if (format) {
        var formatter = buster.create(format);
        formatter.quoteStrings = false;
        sinon.format = buster.bind(formatter, "ascii");
    }

    if (!ba || !sinon) { return; }

    // Sinon assertions for buster
    function verifyFakes() {
        var method, isNot;

        for (var i = 0, l = arguments.length; i < l; ++i) {
            method = arguments[i];
            isNot = (method || "fake") + " is not ";

            if (!method) this.fail(isNot + "a spy");
            if (typeof method != "function") this.fail(isNot + "a function");
            if (typeof method.getCall != "function") this.fail(isNot + "stubbed");
        }

        return true;
    }

    var sf = sinon.spy.formatters;
    var spyValues = function (spy) { return [spy, sf.c(spy), sf.C(spy)]; };

    ba.add("called", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.called;
        },
        assertMessage: "Expected ${0} to be called at least once but was never called",
        refuteMessage: "Expected ${0} to not be called but was called ${1}${2}",
        expectation: "toHaveBeenCalled",
        values: spyValues
    });

    function slice(arr, from, to) {
        return [].slice.call(arr, from, to);
    }

    ba.add("callOrder", {
        assert: function (spy) {
            var args = buster.isArray(spy) ? spy : arguments;
            verifyFakes.apply(this, args);
            if (sinon.calledInOrder(args)) return true;

            this.expected = [].join.call(args, ", ");
            this.actual = sinon.orderByFirstCall(slice(args)).join(", ");
        },

        assertMessage: "Expected ${expected} to be called in order but were called as ${actual}",
        refuteMessage: "Expected ${expected} not to be called in order"
    });

    function addCallCountAssertion(count) {
        var c = count.toLowerCase();

        ba.add("called" + count, {
            assert: function (spy) {
                verifyFakes.call(this, spy);
                return spy["called" + count];
            },
            assertMessage: "Expected ${0} to be called " + c + " but was called ${1}${2}",
            refuteMessage: "Expected ${0} to not be called exactly " + c + "${2}",
            expectation: "toHaveBeenCalled" + count,
            values: spyValues
        });
    }

    addCallCountAssertion("Once");
    addCallCountAssertion("Twice");
    addCallCountAssertion("Thrice");

    function valuesWithThis(spy, thisObj) {
        return [spy, thisObj, spy.printf && spy.printf("%t") || ""];
    }

    ba.add("calledOn", {
        assert: function (spy, thisObj) {
            verifyFakes.call(this, spy);
            return spy.calledOn(thisObj);
        },
        assertMessage: "Expected ${0} to be called with ${1} as this but was called on ${2}",
        refuteMessage: "Expected ${0} not to be called with ${1} as this",
        expectation: "toHaveBeenCalledOn",
        values: valuesWithThis
    });

    ba.add("alwaysCalledOn", {
        assert: function (spy, thisObj) {
            verifyFakes.call(this, spy);
            return spy.alwaysCalledOn(thisObj);
        },
        assertMessage: "Expected ${0} to always be called with ${1} as this but was called on ${2}",
        refuteMessage: "Expected ${0} not to always be called with ${1} as this",
        expectation: "toHaveAlwaysBeenCalledOn",
        values: valuesWithThis
    });

    function formattedArgs(args, i) {
        for (var l = args.length, result = []; i < l; ++i) {
            result.push(sinon.format(args[i]));
        }

        return result.join(", ");
    }

    function spyAndCalls(spy) {
        return [spy, formattedArgs(arguments, 1), spy.printf && spy.printf("%C")];
    }

    ba.add("calledWith", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.calledWith.apply(spy, slice(arguments, 1));
        },
        assertMessage: "Expected ${0} to be called with arguments ${1}${2}",
        refuteMessage: "Expected ${0} not to be called with arguments ${1}${2}",
        expectation: "toHaveBeenCalledWith",
        values: spyAndCalls
    });

    ba.add("alwaysCalledWith", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.alwaysCalledWith.apply(spy, slice(arguments, 1));
        },
        assertMessage: "Expected ${0} to always be called with arguments ${1}${2}",
        refuteMessage: "Expected ${0} not to always be called with arguments${1}${2}",
        expectation: "toHaveAlwaysBeenCalledWith",
        values: spyAndCalls
    });

    ba.add("calledOnceWith", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.calledOnce && spy.calledWith.apply(spy, slice(arguments, 1));
        },
        assertMessage: "Expected ${0} to be called once with arguments ${1}${2}",
        refuteMessage: "Expected ${0} not to be called once with arguments ${1}${2}",
        expectation: "toHaveBeenCalledOnceWith",
        values: spyAndCalls
    });

    ba.add("calledWithExactly", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.calledWithExactly.apply(spy, slice(arguments, 1));
        },
        assertMessage: "Expected ${0} to be called with exact arguments ${1}${2}",
        refuteMessage: "Expected ${0} not to be called with exact arguments${1}${2}",
        expectation: "toHaveBeenCalledWithExactly",
        values: spyAndCalls
    });

    ba.add("alwaysCalledWithExactly", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.alwaysCalledWithExactly.apply(spy, slice(arguments, 1));
        },
        assertMessage: "Expected ${0} to always be called with exact arguments ${1}${2}",
        refuteMessage: "Expected ${0} not to always be called with exact arguments${1}${2}",
        expectation: "toHaveAlwaysBeenCalledWithExactly",
        values: spyAndCalls
    });

    function spyAndException(spy, exception) {
        return [spy, spy.printf && spy.printf("%C")];
    }

    ba.add("threw", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.threw(arguments[1]);
        },
        assertMessage: "Expected ${0} to throw an exception${1}",
        refuteMessage: "Expected ${0} not to throw an exception${1}",
        expectation: "toHaveThrown",
        values: spyAndException
    });

    ba.add("alwaysThrew", {
        assert: function (spy) {
            verifyFakes.call(this, spy);
            return spy.alwaysThrew(arguments[1]);
        },
        assertMessage: "Expected ${0} to always throw an exception${1}",
        refuteMessage: "Expected ${0} not to always throw an exception${1}",
        expectation: "toAlwaysHaveThrown",
        values: spyAndException
    });
}(typeof buster == "object" ? buster : null, typeof sinon == "object" ? sinon : null));

(function (glbl, buster) {
    if (typeof require == "function" && typeof module == "object") {
        buster = require("buster-core");

        module.exports = buster.extend(buster, require("buster-test"), {
            assertions: require("buster-assertions"),
            format: require("buster-format"),
            eventedLogger: require("buster-evented-logger")
        });

        buster.defineVersionGetter(module.exports, __dirname);
        require("buster-sinon");
    }

    if (buster.format) {
        var logFormatter = buster.create(buster.format);
        logFormatter.quoteStrings = false;
        var asciiFormat = buster.bind(logFormatter, "ascii");
    }

    if (buster.eventedLogger) {
        if (asciiFormat) {
            buster.console = buster.eventedLogger.create({
                formatter: asciiFormat,
                logFunctions: true
            });
        }
        buster.log = buster.bind(buster.console, "log");

        buster.captureConsole = function () {
            glbl.console = buster.console;

            if (glbl.console !== buster.console) {
                glbl.console.log = buster.bind(buster.console, "log");
            }
        };
    }

    if (buster.assertions) {
        if (asciiFormat) {
            buster.assertions.format = asciiFormat;
        }
        buster.assert = buster.assertions.assert;
        buster.refute = buster.assertions.refute;

        // TMP, will add mechanism for avoiding this
        glbl.assert = buster.assert;
        glbl.refute = buster.refute;
        glbl.expect = buster.assertions.expect;

        // Assertion counting
        var assertions = 0;
        var count = function () { assertions += 1; };
        buster.assertions.on("pass", count);
        buster.assertions.on("failure", count);
    }

    if (buster.testRunner) {
        buster.testRunner.onCreate(function (runner) {
            buster.assertions.bind(runner, { "failure": "assertionFailure" });
            runner.console = buster.console;

            runner.on("test:async", function () {
                buster.assertions.throwOnFailure = false;
            });

            runner.on("test:setUp", function () {
                buster.assertions.throwOnFailure = true;
            });

            runner.on("test:start", function () {
                assertions = 0;
            });

            runner.on("context:start", function (context) {
                if (context.testCase) {
                    context.testCase.log = buster.bind(buster.console, "log");
                }
            });
        });

        buster.testRunner.assertionCount = function () {
            return assertions;
        };
    }
}(typeof global != "undefined" ? global : this, typeof buster == "object" ? buster : null));
if (typeof module === "object" && typeof require === "function") {
    var buster = module.exports = require("./buster/buster-wiring");
}

(function (glbl) {
    glbl.buster = buster;

    var tc = buster.testContext;
    if (tc.listeners && (tc.listeners.create || []).length > 0) { return; }

    var runner = buster.autoRun({
        cwd: typeof process != "undefined" ? process.cwd() : null
    });

    tc.on("create", runner);
}(typeof global != "undefined" ? global : this));
;return buster; }());