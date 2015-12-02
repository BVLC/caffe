var common = require('../fixtures/common'),
    assert = require('assert');

suite('Ometajs module', function() {
  function unit(grmr, rule, src, dst) {
    var i = new grmr(src);

    if (!i._rule(rule)) {
      i._getError();
    }
    if (dst) assert.deepEqual(i._getIntermediate(), dst);
  };

  suite('given a simple left recursion grammar', function() {
    var grmr = common.require('lr').LeftRecursion;

    test('should match input successfully', function() {
      unit(
        grmr,
        'expr',
        '123 + 456 - 789.4',
        [
          '-' ,
          [ '+', [ 'number', 123 ], [ 'number', 456 ] ],
          [ 'number', 789.4 ]
        ]
      );
    });
  });

  suite('given a grammar with local statement', function() {
    var grmr = common.require('local').Local;

    test('should match input successfully', function() {
      unit(grmr, 'rule', '+', 1);
      unit(grmr, 'rule', '-', 0);
    });
  });

  suite('given a regr-1 grammar', function() {
    var grmr = common.require('regr-1').Regr1;

    test('should match input successfully', function() {
      unit(grmr, 'run', '123', true);
    });
  });

  suite('given a javascript grammar\'s', function() {
    suite('parser', function() {
      var grmr = common.ometajs.grammars.BSJSParser;

      function js(code, ast) {
        var name;
        if (code.length > 50) {
          name = code.slice(0, 47) + '...';
        } else {
          name = code;
        }

        test('`'+ name + '`', function() {
          unit(grmr, 'topLevel', code, ast);
        });
      }

      suite('should match', function() {
        js('var x', ['begin', ['stmt', ['var', ['x']]]]);
        js('var x = 1.2', ['begin', ['stmt', ['var', ['x', ['number', 1.2]]]]]);
        js('var x = 1e2, y, z;', ['begin',
           ['stmt', ['var', ['x', ['number', 100]], ['y'], ['z']]]
        ]);
        js('x.y', [ 'begin',
           ['stmt', [ 'getp', [ 'string', 'y' ], [ 'get' , 'x' ] ]]
        ]);

        js('function a() {}', ['begin',
           ['stmt', ['func', 'a', [], ['begin']]]
        ]);

        js('function a() {return a()}', ['begin',
           ['stmt',['func','a',[],[
             'begin',['stmt',['return',['call',['get','a']]]]
           ]]]
        ]);

        js('function a() {return a()};"123"+"456"', ['begin',
           ['stmt', ['func', 'a', [], [
             'begin', ['stmt', ['return', ['call', ['get', 'a']]]]
           ]]],
           ['stmt', ['binop', '+', ['string', '123'], ['string', '456']]]
        ]);

        js('/a/', ['begin', ['stmt', ['regExp', '/a/']]]);

        js('{ a: 1 , b: 2 }', [
           'begin',
           ['stmt', [ 'json',
             ['binding','a',['number',1]],
             ['binding','b',['number',2]]
           ]]
        ]);

        js('var a = b || c;x', [
           'begin',
           ['stmt', ['var', ['a',['binop','||',['get','b'],['get','c']]]]],
           ['stmt', ['get','x']]
        ]);

        js('a[b].x().l', ['begin',
           ['stmt', ['getp',
             ['string','l'],
             ['call',['getp',['string','x'],['getp',['get','b'],['get','a']]]]
           ]]
        ]);

        js('a.x = function i() {}', [
           'begin',
           ['stmt',['set',
             ['getp',['string','x'],['get','a']],
             ['func','i',[],['begin']]
           ]]
        ]);

        js('a && b || c && d', [
           'begin',
           ['stmt',['binop','||',
             ['binop','&&',['get','a'],['get','b']],
             ['binop','&&',['get','c'],['get','d']]
           ]]
        ]);

        js('(a)', [
           'begin',
           ['stmt',['parens', ['get', 'a']]]
        ]);

        js('"a\\-b\\-c"', [
           'begin',
           ['stmt',['string', 'a-b-c']]
        ]);

        js('"\\"str\\""', [
           'begin',
           ['stmt',['string', '"str"']]
        ]);
      });

      suite('should parse real javascript like', function() {
        test('jquery', function() {
          assert.doesNotThrow(function() {
            grmr.matchAll(common.readFile('jquery.js'), 'topLevel');
          });
        });
      });
    });

    suite('compiler', function() {
      var grmr = common.ometajs.grammars.BSJSTranslator;

      function unit(name, ast, source) {
        test(name, function() {
          assert.equal(grmr.match(ast, 'trans'), source);
        });
      }

      unit('undefined', ['get', 'undefined'], 'undefined');
      unit('basic name', ['get', 'a'], 'a');
      unit('var declaration', ['var', ['x', ['number', 1]]], 'var x = 1');
      unit('var declaration (with binop)', [
        'var', ['x', ['binop', ',', ['number', 1], ['number', 2]]]
      ], 'var x = (1 , 2)');
      unit('two statements', ['begin',
        ['stmt', ['var', 'x']],
        ['stmt', ['var', 'y']]
      ], '{var x;var y}');
      unit(
        'block with statements',
        [ 'begin',
          ['stmt', ['get', 'x']],
          ['stmt', ['var', ['x', ['number', 1]]]]
        ],
        '{x;var x = 1}'
      );
      unit(
        'binop',
        ['binop', '+', ['get', 'x'], ['get', 'y']],
        'x + y'
      );
      unit(
        'binop and assignment',
        ['binop', '+', ['get', 'x'], ['set', ['get', 'y'], ['number', 1]]],
        'x + (y = 1)'
      );
      unit(
        'complex assignment',
        ['set', ['getp', ['get', 'x'], ['get', 'y']], ['get', 'z']],
        'y[x] = z'
      );
      unit(
        'anonymous call',
        ['call', ['func', null, [], ['begin']]],
        '(function (){})()'
      );
      unit(
        'delete keyword',
        ['unop', 'delete', ['get', 'a']],
        'delete a'
      );
      unit(
        'property lookup (regr)',
        ['getp', ['string', 'a-b'], ['get', 'a']],
        'a["a-b"]'
      );
      unit(
        'property lookup (regr#2)',
        ['getp', ['string', 'for'], ['get', 'a']],
        'a["for"]'
      );
      unit(
        'property lookup (regr#3)',
        ['getp', ['string', 'ABC'], ['get', 'a']],
        'a.ABC'
      );
      unit(
        'typeof plus dot',
        ['getp', ['string', 'b'], ['unop', 'typeof', ['get', 'a']]],
        '(typeof a).b'
      );
      unit(
        '&& + || + &&',
        [ 'binop', '&&',
          ['binop', '&&',
            ['get', 'a'],
            ['binop', '||', ['get', 'b'], ['get', 'c']]
          ],
          ['get', 'd']
        ],
        'a && (b || c) && d'
      );
      unit(
        'array and ,',
        ['arr', ['binop', ',', ['get', 'a'], ['get', 'b']]],
        '[(a , b)]'
      );
      unit(
        'call and ,',
        ['call', ['get', 'a'], ['binop', ',', ['get', 'a'], ['get', 'b']]],
        'a((a , b))'
      );
      unit(
        'json and ,',
        ['json', ['binding', 'a', ['binop', ',', ['get', 'a'], ['get', 'b']]]],
        '{"a": (a , b)}'
      );
      unit(
        'parens',
        ['parens', ['get', 'b']],
        '(b)'
      );
    });
  });

  suite('Various grammars', function() {
    function unit(name, grmr, rule, source, result) {
      grmr = common.require(grmr).Grammar;

      test(name, function() {
        assert.deepEqual(grmr.match(source, rule), result);
      });
    }

    unit('grammar with repeating rule', 'rep-rule', 'rule', false, false);
  });
});
