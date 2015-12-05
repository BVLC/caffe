'use strict';

var crypto = require('crypto');

module.exports = function md5(str) {
  return crypto
          .createHash('md5')
          .update(str)
          .digest('hex');
};
