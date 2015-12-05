'use strict';

var url = require('url');

module.exports = function(baseURL) {
  // return undefined if not a string or empty string
  if (typeof baseURL !== 'string') { return; }

  // Makes sure it starts and ends with a slash
  if (baseURL[baseURL.length - 1] !== '/') { baseURL = baseURL + '/'; }

  var parsedURL = url.parse(baseURL);
  if (parsedURL.path[0] !== '/') {
    parsedURL.path = '/' + parsedURL.path;
  }

  if (!parsedURL.host && !parsedURL.protocol) {
    return parsedURL.path;
  } else {
    return parsedURL.href;
  }
};
