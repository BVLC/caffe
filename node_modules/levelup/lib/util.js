/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License
 * <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var extend = require('xtend')
  , errors = require('./errors')
  , bops   = require('bops')

  , encodingNames = [
        'hex'
      , 'utf8'
      , 'utf-8'
      , 'ascii'
      , 'binary'
      , 'base64'
      , 'ucs2'
      , 'ucs-2'
      , 'utf16le'
      , 'utf-16le'
    ]

  , defaultOptions = {
        createIfMissing : true
      , errorIfExists   : false
      , keyEncoding     : 'utf8'
      , valueEncoding   : 'utf8'
      , compression     : true
    }

  , leveldown

  , isBinary = function (data) {
      return data === undefined || data === null || bops.is(data)
    }

  , encodings = (function () {
      var encodings = {}
      encodings.utf8 = encodings['utf-8'] = {
          encode : function (data) {
            return isBinary(data) ? data : String(data)
          }
        , decode : function (data) { return data }
        , buffer : false
        , type   : 'utf8'
      }
      encodings.json = {
          encode : JSON.stringify
        , decode : JSON.parse
        , buffer : false
        , type   : 'json'
      }
      encodingNames.forEach(function (type) {
        if (encodings[type])
          return
        encodings[type] = {
            encode : function (data) {
              return isBinary(data) ? data : bops.from(data, type)
            }
          , decode : function (buffer) {
              return bops.from(buffer, type)
            }
          , buffer : true
          , type   : type // useful for debugging purposes
        }
      })
      return encodings
    })()

  , copy = function (srcdb, dstdb, callback) {
      srcdb.readStream()
        .pipe(dstdb.writeStream())
        .on('close', callback ? callback : function () {})
        .on('error', callback ? callback : function (err) { throw err })
    }

  , setImmediate = global.setImmediate || process.nextTick

  , encodingOpts = (function () {
      var eo = {}
      encodingNames.forEach(function (e) {
        eo[e] = { valueEncoding : e }
      })
      return eo
    }())

  , getOptions = function (levelup, options) {
      var s = typeof options == 'string' // just an encoding
      if (!s && options && options.encoding && !options.valueEncoding)
        options.valueEncoding = options.encoding
      return extend(
          (levelup && levelup.options) || {}
        , s ? encodingOpts[options] || encodingOpts[defaultOptions.valueEncoding]
            : options
      )
    }

  , getLevelDOWN = function () {
      if (leveldown)
        return leveldown

      var requiredVersion       = require('../package.json').devDependencies.leveldown
        , missingLevelDOWNError = 'Could not locate LevelDOWN, try `npm install leveldown`'
        , leveldownVersion

      try {
        leveldownVersion = require('leveldown/package').version
      } catch (e) {
        throw new errors.LevelUPError(missingLevelDOWNError)
      }

      if (!require('semver').satisfies(leveldownVersion, requiredVersion)) {
        throw new errors.LevelUPError(
            'Installed version of LevelDOWN ('
          + leveldownVersion
          + ') does not match required version ('
          + requiredVersion
          + ')'
        )
      }

      try {
        return leveldown = require('leveldown')
      } catch (e) {
        throw new errors.LevelUPError(missingLevelDOWNError)
      }
    }

  , getKeyEncoder = function (options, op) {
      var type = ((op && op.keyEncoding) || options.keyEncoding) || 'utf8'
      return encodings[type] || type
    }

  , getValueEncoder = function (options, op) {
      var type = (((op && (op.valueEncoding || op.encoding))
          || options.valueEncoding || options.encoding)) || 'utf8'
      return encodings[type] || type
    }

  , encodeKey = function (key, options, op) {
      return getKeyEncoder(options, op).encode(key)
    }

  , encodeValue = function (value, options, op) {
      return getValueEncoder(options, op).encode(value)
    }

  , decodeKey = function (key, options) {
      return getKeyEncoder(options).decode(key)
    }

  , decodeValue = function (value, options) {
      return getValueEncoder(options).decode(value)
    }

  , isValueAsBuffer = function (options, op) {
      return getValueEncoder(options, op).buffer
    }

  , isKeyAsBuffer = function (options, op) {
      return getKeyEncoder(options, op).buffer
    }

module.exports = {
    defaultOptions      : defaultOptions
  , copy                : copy
  , setImmediate        : setImmediate
  , getOptions          : getOptions
  , getLevelDOWN        : getLevelDOWN
  , encodeKey           : encodeKey
  , encodeValue         : encodeValue
  , isValueAsBuffer     : isValueAsBuffer
  , isKeyAsBuffer       : isKeyAsBuffer
  , decodeValue         : decodeValue
  , decodeKey           : decodeKey
}
