/*!
 * basic-auth
 * Copyright(c) 2013 TJ Holowaychuk
 * Copyright(c) 2014 Jonathan Ong
 * Copyright(c) 2015 Douglas Christopher Wilson
 * MIT Licensed
 */

'use strict'

/**
 * Module exports.
 * @public
 */

module.exports = auth

/**
 * RegExp for basic auth credentials
 *
 * credentials = auth-scheme 1*SP token68
 * auth-scheme = "Basic" ; case insensitive
 * token68     = 1*( ALPHA / DIGIT / "-" / "." / "_" / "~" / "+" / "/" ) *"="
 * @private
 */

var credentialsRegExp = /^ *(?:[Bb][Aa][Ss][Ii][Cc]) +([A-Za-z0-9\-\._~\+\/]+=*) *$/

/**
 * RegExp for basic auth user/pass
 *
 * user-pass   = userid ":" password
 * userid      = *<TEXT excluding ":">
 * password    = *TEXT
 * @private
 */

var userPassRegExp = /^([^:]*):(.*)$/

/**
 * Parse the Authorization header field of a request.
 *
 * @param {object} req
 * @return {object} with .name and .pass
 * @public
 */

function auth(req) {
  if (!req) {
    throw new TypeError('argument req is required')
  }

  // get header
  var header = (req.req || req).headers.authorization

  // parse header
  var match = credentialsRegExp.exec(header || '')

  if (!match) {
    return
  }

  // decode user pass
  var userPass = userPassRegExp.exec(decodeBase64(match[1]))

  if (!userPass) {
    return
  }

  // return credentials object
  return new Credentials(userPass[1], userPass[2])
}

/**
 * Decode base64 string.
 * @private
 */

function decodeBase64(str) {
  return new Buffer(str, 'base64').toString()
}

/**
 * Object to represent user credentials.
 * @private
 */

function Credentials(name, pass) {
  this.name = name
  this.pass = pass
}
