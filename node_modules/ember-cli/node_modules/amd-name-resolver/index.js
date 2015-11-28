function moduleResolve(child, name) {
  if (child.charAt(0) !== '.') { return child; }

  var parts = child.split('/');
  var nameParts = name.split('/');
  var parentBase = nameParts.slice(0, -1);

  for (var i = 0, l = parts.length; i < l; i++) {
    var part = parts[i];

    if (part === '..') {
      if (parentBase.length === 0) {
        throw new Error('Cannot access parent module of root');
      }
      parentBase.pop();
    } else if (part === '.') {
      
      continue;
    } else { parentBase.push(part); }
  }

  return parentBase.join('/');
}

module.exports = moduleResolve;