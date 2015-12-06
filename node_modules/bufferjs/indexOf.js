/*jshint strict:true node:true es5:true onevar:true laxcomma:true laxbreak:true eqeqeq:true immed:true latedef:true*/
(function () {
  "use strict";

  /**
   * A naiive 'Buffer.indexOf' function. Requires both the
   * needle and haystack to be Buffer instances.
   */
  function indexOf(haystack, needle, i) {
    if (!Buffer.isBuffer(needle)) needle = new Buffer(needle);
    if (typeof i === 'undefined') i = 0;
    var l = haystack.length - needle.length + 1;
    while (i<l) {
      var good = true;
      for (var j=0, n=needle.length; j<n; j++) {
        if (haystack.get(i+j) !== needle.get(j)) {
          good = false;
          break;
        }
      }
      if (good) return i;
      i++;
    }
    return -1;
  }

  if (!Buffer.indexOf) {
    Buffer.indexOf = indexOf;
  }
  if (!Buffer.prototype.indexOf) {
    Buffer.prototype.indexOf = function(needle, i) {
      return Buffer.indexOf(this, needle, i);
    };
  }

})();
