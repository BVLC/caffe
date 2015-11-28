/*!
 * is-git-url <https://github.com/jonschlinkert/is-git-url>
 *
 * Copyright (c) 2014-2015, Jon Schlinkert.
 * Licensed under the MIT License.
 */

module.exports = function isGitUrl(str) {
  var re = /(?:git|ssh|https?|git@[\w\.]+):(?:\/\/)?[\w\.@:\/~_-]+\.git(?:\/?|\#[\d\w\.\-_]+?)$/;
  return re.test(str);
};
