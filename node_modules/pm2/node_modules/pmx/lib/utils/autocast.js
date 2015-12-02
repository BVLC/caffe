(function() {
  /**
   * Common strings to cast
   */
  var common_strings = {
    'true': true,
    'false': false,
    'undefined': undefined,
    'null': null,
    'NaN': NaN
  };

  function process(key,value, o) {
    if (typeof(value) == 'object') return;
    o[key] = _cast(value);
  }

  function traverse(o,func) {
    for (var i in o) {
      func.apply(this,[i,o[i], o]);
      if (o[i] !== null && typeof(o[i])=="object") {
        //going on step down in the object tree!!
        traverse(o[i],func);
      }
    }
  }

  function _cast(s) {
    var key;

    // Don't cast Date objects
    if (s instanceof Date) return s;
    if (typeof s == 'boolean') return s;

    // Try to cast it to a number
    if ((key = +s) == key) return key;

    // Try to make it a common string
    for (key in common_strings) {
      if (s === key) return common_strings[key];
    }

    // Give up
    return s;
  }

  /**
   * Given a value, try and cast it
   */
  function autocast(s) {
    if (typeof(s) == 'object') {
      traverse(s, process);
      return s;
    }

    return _cast(s);
  };

  // export
  module.exports = autocast;
}());
