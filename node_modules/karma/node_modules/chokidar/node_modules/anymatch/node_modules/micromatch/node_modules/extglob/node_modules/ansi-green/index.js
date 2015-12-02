/*!
 * ansi-green <https://github.com/jonschlinkert/ansi-green>
 *
 * Copyright (c) 2015, Jon Schlinkert.
 * Licensed under the MIT License.
 */

'use strict';

var wrap = require('ansi-wrap');

module.exports = function green(message) {
  return wrap(32, 39, message);
};
