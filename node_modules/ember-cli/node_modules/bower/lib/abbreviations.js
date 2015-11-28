var abbrev = require('abbrev');
var mout = require('mout');

function expandNames(obj, prefix, stack) {
    prefix = prefix || '';
    stack = stack || [];

    mout.object.forOwn(obj, function (value, name) {
        name = prefix + name;

        stack.push(name);

        if (typeof value === 'object' && !value.line) {
            expandNames(value, name + ' ', stack);
        }
    });

    return stack;
}

module.exports = function(commands) {
  var abbreviations = abbrev(expandNames(commands));
  
  abbreviations.i = 'install';
  abbreviations.rm = 'uninstall';
  abbreviations.unlink = 'uninstall';
  abbreviations.ls = 'list';

  return abbreviations;
};
