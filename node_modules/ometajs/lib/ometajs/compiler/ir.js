var ometajs = require('../../ometajs'),
    util = require('util');

//
// [
//   [
//     'grammar', 'name', 'parent',
//     [ [
//       'rule', 'name', [ 'var1', 'var2', 'var3'],
//       [
//         [ 'atomic', [
//           [ 'match', 'anything' ],
//           [ 'rule', 'anything' ],
//           [ 'rule', 'anything', ['body1', 'body2']],
//           [ 'super', 'anything' ],
//           [ 'any', [ 'rule', 'number' ] ]
//         ]],
//         [ 'choice', [
//           [.. ops ..],
//           [.. ops ..]
//         ]]
//       ]
//     ] ]
//   ],
//   [
//     'code', ['1 + 3']
//   ]
// ]
//

//
// ### function IR ()
// Intermediate representation
//
function IR() {
  ometajs.compiler.ast.call(this);
};
util.inherits(IR, ometajs.compiler.ast);
module.exports = IR;

//
// Merges rules with the same name into one rule (via option, i.e. ||)
//
IR.prototype.mergeRules = function mergeRules() {
  this.result.filter(function (node) {
    return node[0] === 'grammar';
  }).forEach(function (grammar) {
    var rules = {},
        names = [];

    grammar[3].forEach(function (rule) {
      var acc;
      if (rules[rule[1]]) {
        acc = rules[rule[1]];
      } else {
        acc = rules[rule[1]] = [];
        names.push(rule[1]);
      }
      acc.push(rule);
    });

    grammar[3] = names.map(function (name) {
      var acc = rules[name];

      // Not repeating rules
      if (acc.length === 1) return acc[0];

      return ['rule', name, [], [ ['choice', acc.map(function (rule) {
        return [ [ 'atomic', rule[2], rule[3] ] ];
      })] ] ];
    });
  });
};

//
// Optimizes IR
//
IR.prototype.optimize = function optimize() {
  var context = [];
  function traverse(ast) {
    function traverseBody(nodes) {
      return nodes.map(function (ast, i, nodes) {
        switch (ast[0]) {
          case 'choice':
            if (ast[1].length === 1 && ast[1][0].length === 1) {
              var next = ast[1][0][0];

              // choice+atomic
              if (next[0] === 'atomic' && nodes[i + 1] === undefined) {
                // Lift arguments
                context.push.apply(context, next[1]);
                return traverse(['choice', [next[2]]]);
              }
            }
          default:
            return traverse(ast);
        }
      });
    }

    // atomic+atomic = atomic
    // atomic+match = match
    // atomic+list = list
    switch (ast[0]) {
      case 'grammar':
        return [
          'grammar',
          ast[1],
          ast[2],
          ast[3].map(function(rule) {
            return [
              'rule',
              rule[1],
              context = rule[2],
              traverseBody(rule[3])
            ];
          })
        ];
      case 'choice':
        return ['choice', ast[1].map(function(nodes) {
          return nodes.map(traverse);
        })];
      case 'atomic':
        if (ast[2].length === 1) {
          var next = ast[2][0];
          switch (next[0]) {
            // atomic+atomic
            case 'atomic':
              // Merge arguments
              return traverse(['atomic', ast[1].concat(next[1]), next[2]]);
            // atomic+match
            case 'match':
            case 'seq':
            case 're':
              // No args here, definitely
              return traverse(next);
            case 'list':
              // Lift arguments to upper context
              context.push.apply(context, ast[1]);
              return traverse(next);
          }
        }
        return ['atomic', ast[1], traverseBody(ast[2])];
      case 'lookahead':
      case 'not':
      case 'list':
      case 'chars':
      case 'any':
      case 'many':
      case 'optional':
        return [ast[0], traverseBody(ast[1])];
      default:
        return ast;
    }
  }

  this.result = this.result.map(traverse);
};

IR.prototype.render = function render(options) {
  var buf = [];

  function multibody(nodes, op, fn) {
    var flag = false;
    if (nodes.length === 0) {
      buf.push('true');
    } else {
      nodes.forEach(function(node, i) {
        if (i !== 0) buf.push(op);
        fn(node);
      });
    }
  };

  function body(nodes, op) {
    return multibody(nodes, op, traverse);
  };

  function args(list) {
    // Add variables for arguments
    if (list.length > 0) {
      list.forEach(function(arg, i) {
        buf.push(i === 0 ? 'var ' : ', ');
        buf.push(arg);
      });
      buf.push(';');
    }
  };

  var parent;

  function traverse(ast) {
    switch (ast[0]) {
      case 'grammar':
        buf.push('var ', ast[1], ' = function ', ast[1], '(source, opts) {');
        buf.push(ast[2], '.call(this, source, opts)');
        buf.push('};');

        buf.push(ast[1], '.grammarName = ', JSON.stringify(ast[1]), ';');

        buf.push(ast[1], '.match = ', ast[2], '.match;');
        buf.push(ast[1], '.matchAll = ', ast[2], '.matchAll;');

        buf.push('exports.', ast[1], ' = ', ast[1], ';');

        buf.push('require("util").inherits(', ast[1], ', ', ast[2], ');');

        parent = ast[2];

        ast[3].forEach(function(rule) {
          var name = rule[1] || '';
          buf.push(
            ast[1], '.prototype[', JSON.stringify(name), ']',
            ' = function $' + name.replace(/[^\w]+/g, '') + '() {'
          );

          args(rule[2]);

          buf.push('return ');
          body(rule[3], ' && ');
          buf.push('};\n');
        });

        break;
      case 'code':
        ast[1].forEach(function(item) {
          buf.push(item);
        });
        break;
      case 'choice':
        // Each choice should be wrapped in atomic
        buf.push('(');
        multibody(ast[1], ' || ', function(nodes) {
          return body(nodes, ' && ');
        });
        buf.push(')');
        break;
      case 'atomic':
        buf.push('this._atomic(function() {');
        args(ast[1]);
        buf.push('return ');
        body(ast[2], ' && ');
        buf.push('})');
        break;
      case 'store':
        buf.push('((', ast[1], ' = this._getIntermediate()), true)');
        break;
      case 'lookahead':
        buf.push('this._atomic(function() {return ');
        body(ast[1], ' && ');
        buf.push('}, true)');
        break;
      case 'not':
        buf.push('!this._atomic(function() {return ');
        body(ast[1], ' && ');
        buf.push('}, true)');
        break;
      case 'list':
        buf.push('this._list(function() {return ');
        body(ast[1], ' && ');
        buf.push('})');
        break;
      case 'chars':
        buf.push('this._list(function() {return ');
        body(ast[1], ' && ');
        buf.push('}, true)');
        break;
      case 'exec':
        buf.push('this._exec(', ast[1], ')');
        break;
      case 'predicate':
        buf.push('(', ast[1], ')');
        break;
      case 'rule':
      case 'super':
        if (ast[1] === 'anything') {
          buf.push('this._skip()');
          break;
        }

        buf.push(
          'this.', '_rule', '(',
          JSON.stringify(ast[1].replace(/^@/, '')),
          ',',
          (/^@/.test(ast[1]) || ast[1] === 'token') ? 'true': 'false',
          ',['
        );
        if (ast[2]) {
          ast[2].forEach(function(code, i) {
            if (i !== 0) buf.push(',');
            buf.push(code);
          });
        }
        buf.push(
          ']',
          ',',
          ast[0] === 'rule' ? 'null' : parent,
          ',',
          ast[0] === 'rule' ? 'this[' : (parent + '.prototype['),
          JSON.stringify(ast[1].replace(/^@/, '')),
          ']',
          ')'
        );
        break;
      case 'match':
        buf.push('this._match(', ast[1], ')');
        break;
      case 'seq':
      case 're':
        if (/^\/\^/.test(ast[1])) {
          buf.push('this._seq(', ast[1], ')');
        } else {
          var re = ast[1].replace(/^\/(.*)\/(\w*)$/, '/^($1)/$2');
          buf.push('this._seq(', re, ')');
        }
        break;
      case 'optional':
        buf.push('this._', ast[0], '(function() {return ');
        body(ast[1], ' && ');
        buf.push('})');
        break;
      case 'any':
      case 'many':
        buf.push('this._', ast[0], '(function() {');
        buf.push('  return this._atomic(function() { return ');
        body(ast[1], ' && ');
        buf.push('  });');
        buf.push('})');
        break;
      default:
        throw new Error('Unknown IR node type:' + ast[0]);
    }
  };

  this.mergeRules();
  this.optimize();

  this.result.forEach(function(node) {
    traverse(node);
  });

  if (this.options.beautify) {
    return ometajs.utils.beautify(buf.join(''));
  } else {
    return buf.join('');
  }
};
