var utils = exports;

var uglify = require('uglify-js');

utils.extend = function extend(target, source) {
  Object.keys(source).forEach(function (key) {
    target[key] = source[key];
  });
};

utils.beautify = function beautify(code) {
  var ast = uglify.parser.parse(code);
  return uglify.uglify.gen_code(ast, { beautify: true });
};

utils.expressionify = function expressionify(code) {
  try {
    var ast = uglify.parser.parse('(function(){\n' + code + '\n})');
  } catch(e) {
    console.error(e.message + ' on ' + (e.line - 1) + ':' + e.pos);
    console.error('in');
    console.error(code);
    throw e;
  }

  ast[1] = ast[1][0][1][3];

  function traverse(ast) {
    if (!Array.isArray(ast)) return ast;
    switch (ast[0]) {
      case 'toplevel':
        if (ast[1].length === 1 && ast[1][0][0] !== 'block') {
          return ast;
        } else {
          var children = ast[1][0][0] === 'block' ? ast[1][0][1] : ast[1];

          return ['toplevel', [[
            'call', [
              'dot', [
                'function', null, [],
                children.map(function(child, i, children) {
                  return (i == children.length - 1) ? traverse(child) : child;
                })
              ],
              'call'
            ],
            [ ['name', 'this'] ]
          ]]];
        }
      case 'block':
        // Empty blocks can't be processed
        if (ast[1].length <= 0) return ast;

        var last = ast[1][ast[1].length - 1];
        return [
          ast[0],
          ast[1].slice(0, -1).concat([traverse(last)])
        ];
      case 'while':
      case 'for':
      case 'switch':
        return ast;
      case 'if':
        return [
          'if',
          ast[1],
          traverse(ast[2]),
          traverse(ast[3])
        ];
      case 'stat':
        return [
          'stat',
          traverse(ast[1])
        ];
      default:
        if (ast[0] === 'return') return ast;
        return [
          'return',
          ast
        ]
    }
    return ast;
  }

  return uglify.uglify.gen_code(traverse(ast)).replace(/;$/, '');
};

utils.localify = function localify(code, id) {
  var ast = uglify.parser.parse(code);

  if (ast[1].length !== 1 || ast[1][0][0] !== 'stat') {
    throw new TypeError('Incorrect code for local: ' + code);
  }

  var vars = [],
      set = [],
      unset = [];

  function traverse(node) {
    if (node[0] === 'assign') {
      if (node[1] !== true) {
        throw new TypeError('Incorrect assignment in local');
      }

      if (node[2][0] === 'dot' || node[2][0] === 'sub') {
        var host = ['name', '$l' + id++];
        vars.push(host[1]);

        set.push(['assign', true, host, node[2][1]]);
        node[2][1] = host;

        if (node[2][0] === 'sub') {
          var property = ['name', '$l' + id++];
          vars.push(property[1]);
          set.push(['assign', true, property, node[2][2]]);
          node[2][2] = property;
        }
      }

      var target = ['name', '$l' + id++];

      vars.push(target[1]);

      set.push(['assign', true, target, node[2]]);
      set.push(['assign', true, node[2], node[3]]);
      unset.push(['assign', true, node[2], target]);
    } else if (node[0] === 'seq') {
      traverse(node[1]);
      traverse(node[2]);
    } else {
      throw new TypeError(
        'Incorrect code for local (' + node[0] + '): ' + code
      );
    }
  }
  traverse(ast[1][0][1]);

  function generate(seqs) {
    return uglify.uglify.gen_code(seqs.reduce(function (current, acc) {
      return ['seq', current, acc];
    }));
  }

  return {
    vars: vars,
    before: generate(set.concat([['name', 'true']])),
    afterSuccess: generate(unset.concat([['name', 'true']])),
    afterFail: generate(unset.concat([['name', 'false']]))
  };
};

utils.merge = function merge(a, b) {
  Object.keys(b).forEach(function(key) {
    a[key] = b[key];
  });
};
