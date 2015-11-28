/*!
 * compressible
 * Copyright(c) 2013 Jonathan Ong
 * Copyright(c) 2014 Jeremiah Senkpiel
 * Copyright(c) 2015 Douglas Christopher Wilson
 * MIT Licensed
 */

'use strict'

/**
 * Module dependencies.
 * @private
 */

var db = require('mime-db')

/**
 * Module variables.
 * @private
 */

var compressibleTypeRegExp = /^text\/|\+json$|\+text$|\+xml$/i
var extractTypeRegExp = /^\s*([^;\s]*)(?:;|\s|$)/

/**
 * Module exports.
 * @public
 */

module.exports = compressible

/**
 * Checks if a type is compressible.
 *
 * @param {string} type
 * @return {Boolean} compressible
 & @public
 */

function compressible(type) {
  if (!type || typeof type !== 'string') {
    return false
  }

  // strip parameters
  var match = extractTypeRegExp.exec(type)
  var mime = match && match[1].toLowerCase()
  var data = db[mime]

  if ((data && data.compressible) || compressibleTypeRegExp.test(mime)) {
    return true
  }

  return data
    ? data.compressible
    : undefined
}
