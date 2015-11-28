/*!
 * raw-body
 * Copyright(c) 2013-2014 Jonathan Ong
 * Copyright(c) 2014-2015 Douglas Christopher Wilson
 * MIT Licensed
 */

'use strict'

/**
 * Module dependencies.
 * @private
 */

var bytes = require('bytes')
var iconv = require('iconv-lite')
var unpipe = require('unpipe')

/**
 * Module exports.
 * @public
 */

module.exports = getRawBody

/**
 * Module variables.
 * @private
 */

var iconvEncodingMessageRegExp = /^Encoding not recognized: /

/**
 * Get the decoder for a given encoding.
 *
 * @param {string} encoding
 * @private
 */

function getDecoder(encoding) {
  if (!encoding) return null

  try {
    return iconv.getDecoder(encoding)
  } catch (e) {
    // error getting decoder
    if (!iconvEncodingMessageRegExp.test(e.message)) throw e

    // the encoding was not found
    throw createError(415, 'specified encoding unsupported', 'encoding.unsupported', {
      encoding: encoding
    })
  }
}

/**
 * Get the raw body of a stream (typically HTTP).
 *
 * @param {object} stream
 * @param {object|string|function} [options]
 * @param {function} [callback]
 * @public
 */

function getRawBody(stream, options, callback) {
  var done = callback
  var opts = options || {}

  if (options === true || typeof options === 'string') {
    // short cut for encoding
    opts = {
      encoding: options
    }
  }

  if (typeof options === 'function') {
    done = options
    opts = {}
  }

  // validate callback is a function, if provided
  if (done !== undefined && typeof done !== 'function') {
    throw new TypeError('argument callback must be a function')
  }

  // require the callback without promises
  if (!done && !global.Promise) {
    throw new TypeError('argument callback is required')
  }

  // get encoding
  var encoding = opts.encoding !== true
    ? opts.encoding
    : 'utf-8'

  // convert the limit to an integer
  var limit = bytes.parse(opts.limit)

  // convert the expected length to an integer
  var length = opts.length != null && !isNaN(opts.length)
    ? parseInt(opts.length, 10)
    : null

  if (done) {
    // classic callback style
    return readStream(stream, encoding, length, limit, done)
  }

  return new Promise(function executor(resolve, reject) {
    readStream(stream, encoding, length, limit, function onRead(err, buf) {
      if (err) return reject(err)
      resolve(buf)
    })
  })
}

/**
 * Halt a stream.
 *
 * @param {Object} stream
 * @private
 */

function halt(stream) {
  // unpipe everything from the stream
  unpipe(stream)

  // pause stream
  if (typeof stream.pause === 'function') {
    stream.pause()
  }
}

/**
 * Make a serializable error object.
 *
 * To create serializable errors you must re-set message so
 * that it is enumerable and you must re configure the type
 * property so that is writable and enumerable.
 *
 * @param {number} status
 * @param {string} message
 * @param {string} type
 * @param {object} props
 * @private
 */

function createError(status, message, type, props) {
  var error = new Error()

  // capture stack trace
  Error.captureStackTrace(error, createError)

  // set free-form properties
  for (var prop in props) {
    error[prop] = props[prop]
  }

  // set message
  error.message = message

  // set status
  error.status = status
  error.statusCode = status

  // set type
  Object.defineProperty(error, 'type', {
    value: type,
    enumerable: true,
    writable: true,
    configurable: true
  })

  return error
}

/**
 * Read the data from the stream.
 *
 * @param {object} stream
 * @param {string} encoding
 * @param {number} length
 * @param {number} limit
 * @param {function} callback
 * @public
 */

function readStream(stream, encoding, length, limit, callback) {
  var complete = false
  var sync = true

  // check the length and limit options.
  // note: we intentionally leave the stream paused,
  // so users should handle the stream themselves.
  if (limit !== null && length !== null && length > limit) {
    return done(createError(413, 'request entity too large', 'entity.too.large', {
      expected: length,
      length: length,
      limit: limit
    }))
  }

  // streams1: assert request encoding is buffer.
  // streams2+: assert the stream encoding is buffer.
  //   stream._decoder: streams1
  //   state.encoding: streams2
  //   state.decoder: streams2, specifically < 0.10.6
  var state = stream._readableState
  if (stream._decoder || (state && (state.encoding || state.decoder))) {
    // developer error
    return done(createError(500, 'stream encoding should not be set', 'stream.encoding.set'))
  }

  var received = 0
  var decoder

  try {
    decoder = getDecoder(encoding)
  } catch (err) {
    return done(err)
  }

  var buffer = decoder
    ? ''
    : []

  // attach listeners
  stream.on('aborted', onAborted)
  stream.on('close', cleanup)
  stream.on('data', onData)
  stream.on('end', onEnd)
  stream.on('error', onEnd)

  // mark sync section complete
  sync = false

  function done() {
    var args = new Array(arguments.length)

    // copy arguments
    for (var i = 0; i < args.length; i++) {
      args[i] = arguments[i]
    }

    // mark complete
    complete = true

    if (sync) {
      process.nextTick(invokeCallback)
    } else {
      invokeCallback()
    }

    function invokeCallback() {
      cleanup()

      if (args[0]) {
        // halt the stream on error
        halt(stream)
      }

      callback.apply(null, args)
    }
  }

  function onAborted() {
    if (complete) return

    done(createError(400, 'request aborted', 'request.aborted', {
      code: 'ECONNABORTED',
      expected: length,
      length: length,
      received: received
    }))
  }

  function onData(chunk) {
    if (complete) return

    received += chunk.length
    decoder
      ? buffer += decoder.write(chunk)
      : buffer.push(chunk)

    if (limit !== null && received > limit) {
      done(createError(413, 'request entity too large', 'entity.too.large', {
        limit: limit,
        received: received
      }))
    }
  }

  function onEnd(err) {
    if (complete) return
    if (err) return done(err)

    if (length !== null && received !== length) {
      done(createError(400, 'request size did not match content length', 'request.size.invalid', {
        expected: length,
        length: length,
        received: received
      }))
    } else {
      var string = decoder
        ? buffer + (decoder.end() || '')
        : Buffer.concat(buffer)
      cleanup()
      done(null, string)
    }
  }

  function cleanup() {
    buffer = null

    stream.removeListener('aborted', onAborted)
    stream.removeListener('data', onData)
    stream.removeListener('end', onEnd)
    stream.removeListener('error', onEnd)
    stream.removeListener('close', cleanup)
  }
}
