var bash = exports;

// Simple yet elegant shell escaping. This was proposed by shaver in IRC who
// says he used it in an app that processed millions of commands this way.
// At transloadit we have processed millions of commands this way now as well.
bash.escape = function() {
  return Array.prototype
    .slice.call(arguments)
    .map(function(argument) {
      if (argument === undefined || argument === null) {
        argument = '';
      }

      if (argument === '') {
        return "''";
      }

      // Escape everything that is potentially unsafe with a backslash
      return (argument + '').replace(/([^0-9a-z-])/gi, '\\$1');
    })
    .join(' ');
};

bash.args = function(options, prefix, suffix) {
  var args = [];
  options = [].concat(options);

  options.forEach(function(block) {
    for (var key in block) {
      [].concat(block[key])
        .forEach(function(val) {
          if (val === null || val === true) {
            args.push(prefix + key);
          } else {
            args.push(prefix + key + suffix + bash.escape(val));
          }
        });
    }
  });

  return args.join(' ');
};
