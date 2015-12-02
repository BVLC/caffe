var common = require('../fixtures/common'),
    assert = require('assert'),
    uglify = require('uglify-js');

var units = [
];

function unit(src, dst, throws) {
  if (throws) {
    assert.throws(function () { common.parse(src) });
  } else if (dst) {
    assert.deepEqual(common.parse(src), dst);
  } else {
    var ast = common.parse(src);
    assert.ok(Array.isArray(ast));
    assert.ok(ast.length > 0);
  }
};

suite('Ometajs language parser', function() {
  suite('should fail on grammar with', function() {
    test('missing closing bracket', function() {
      unit('ometa name {', null, true);
    });

    test('<: but without parent grammar name', function() {
      unit('ometa name {', null, true);
    });
  });

  suite('should generate correct AST for', function() {
    suite('simple grammars like', function() {
      test('empty grammar', function() {
        unit('ometa name {\n}', [ [ 'grammar', 'name', null, [] ] ]);
      });

      test('empty grammars', function() {
        unit(
          'ometa name {\n} ometa name2 <: name {\n}',
          [
            [ 'grammar', 'name', null, [] ],
            [ 'grammar', 'name2', 'name', [] ]
          ]
        );
      });

      test('empty grammar with a host code near it', function() {
        unit(
          'var ometa = 1;\nometa name {\n};\nconsole.log("123");',
          [
            [ 'code', 'var ometa = 1;\n'],
            [ 'grammar', 'name', null, [] ],
            [ 'code', ';\nconsole.log("123");\n']
          ]
        );
      })

      test('grammar with only one empty rule', function() {
        unit(
          'ometa name { ruleName }',
          [ [
            'grammar', 'name', null,
            [ [ 'rule', 'ruleName', [] ] ]
          ] ]
        );
      });
    });

    suite('grammars with basic rule with', function() {
      test('one arg', function() {
        unit(
          'ometa name { rule :a }',
          [ [
            'grammar',
            'name',
            null,
            [ [ 'rule', 'rule', [ [
              'arg',
              [ 'match', null, 'anything' ],
              'a'
            ] ] ] ]
          ] ]
        );
      });

      test('primitives (number, boolean, null)', function() {
        unit(
          'ometa name { rule 123 true false null }',
          [ [
            'grammar',
            'name',
            null,
            [ [ 'rule', 'rule', [
              [ 'number', 123 ],
              [ 'bool', true ],
              [ 'bool', false ],
              [ 'null' ]
            ] ] ]
          ] ]
        );
      });

      test('char sequence', function() {
        unit(
          'ometa name { rule ``abc\'\' }',
          [ [
            'grammar',
            'name',
            null,
            [ [ 'rule', 'rule', [ [ 'seq', 'abc' ] ] ] ]
          ] ]
        );
      });

      test('regexp', function() {
        unit(
          'ometa name { rule /abc/gim }',
          [ [
            'grammar',
            'name',
            null,
            [ [ 'rule', 'rule', [ [ 're', '/abc/gim' ] ] ] ]
          ] ]
        );
      });

      test('two args', function() {
        unit(
          'ometa name { rule :a :b }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule',
              [
                [ 'arg', [ 'match', null, 'anything' ], 'a' ],
                [ 'arg', [ 'match', null, 'anything' ], 'b' ]
              ]
            ] ]
          ] ]
        );
      });

      test('repeating rule', function() {
        unit(
          'ometa name { rule true, rule false }',
          [ [
            'grammar',
            'name',
            null,
            [
              [ 'rule', 'rule', [ [ 'bool', true ] ] ],
              [ 'rule', 'rule', [ [ 'bool', false ] ] ]
            ]
          ] ]
        );
      });

      test('two args (x2)', function() {
        unit(
          'ometa name { rule1 :a :b, rule2 :c :d }',
          [ [
            'grammar',
            'name',
            null,
            [
              [
                'rule', 'rule1',
                [
                  [ 'arg', [ 'match', null, 'anything' ], 'a' ],
                  [ 'arg', [ 'match', null, 'anything' ], 'b' ]
                ]
              ],
              [
                'rule', 'rule2',
                [
                  [ 'arg', [ 'match', null, 'anything' ], 'c' ],
                  [ 'arg', [ 'match', null, 'anything' ], 'd' ]
                ]
              ]
            ]
          ] ]
        );
      });

      test('two args (x2 grammars)', function() {
        unit(
          'ometa name { rule :a :b } ometa name2 { rule :c :d }',
          [
            [
              'grammar',
              'name',
              null,
              [ [
                'rule', 'rule',
                [
                  [ 'arg', [ 'match', null, 'anything' ], 'a' ],
                  [ 'arg', [ 'match', null, 'anything' ], 'b' ]
                ]
              ] ]
            ],
            [
              'grammar',
              'name2',
              null,
              [ [
                'rule', 'rule',
                [
                  [ 'arg', [ 'match', null, 'anything' ], 'c' ],
                  [ 'arg', [ 'match', null, 'anything' ], 'd' ]
                ]
              ] ]
            ]
          ]
        );
      });

      test('left side and right side', function() {
        unit(
          'ometa name { rule :a = :b }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule',
              [
                [ 'arg', [ 'match', null, 'anything' ], 'a' ],
                [ 'choice', [ [ 'arg', [ 'match', null, 'anything' ], 'b' ] ] ]
              ]
            ] ]
          ] ]
        );
      });
    });

    suite('grammars with a complex rule with', function() {
      test('predicate', function() {
        unit(
          'ometa name { ruleName = ?doAnything}',
          [ [
            'grammar', 'name', null,
            [ [ 'rule', 'ruleName', [
              [ 'choice', [ [ 'predicate', common.expressionify('doAnything') ] ] ]
            ] ] ]
          ] ]
        );
      });

      test('local', function() {
        unit(
          'ometa name { ruleName = %(this.a = 1, this.b = 1) }',
          [ [
            'grammar', 'name', null,
            [ [ 'rule', 'ruleName', [
              [ 'choice', [ [ 'local', common.expressionify('this.a = 1, this.b = 1') ] ] ]
            ] ] ]
          ] ]
        );
      });

      test('predicate and code', function() {
        unit(
          'ometa name { ruleName = ?doAnything() { 1 + 1 } }',
          [ [
            'grammar', 'name', null,
            [ [ 'rule', 'ruleName', [
              [ 'choice', [
                [ 'predicate', common.expressionify('doAnything()') ],
                [ 'body', common.expressionify('1 + 1') ]
              ] ]
            ] ] ]
          ] ]
        );
      });

      test('predicate (regr#1)', function() {
        unit(
          'ometa name { ' +
          '  keyword :k = iName:kk isKeyword(kk) ' +
          '  ?(!k || k == kk) -> [#keyword, kk]' +
          '}',
          [ [
            'grammar',
            'name',
            null,
            [ [ 'rule',
                'keyword',
                [ [ 'arg', [ 'match', null, 'anything' ], 'k' ],
                  [ 'choice',
                    [ [ 'arg', [ 'match', null, 'iName' ], 'kk' ],
                      [ 'call', null, 'isKeyword', [ 'kk' ] ],
                      [ 'predicate', '!k||k==kk' ],
                      [ 'result', '["keyword",kk]' ]
                    ]
                  ]
                ]
            ] ]
          ] ]
        );
      });

      test('named invocation', function() {
        unit(
          'ometa name { rule sub:a }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule', [
                [ 'arg', [ 'match', null, 'sub' ], 'a' ]
            ]
            ] ]
          ] ]
        );
      });

      test('named arg + modificator', function() {
        unit(
          'ometa name { rule sub*:a }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule', [
                [ 'arg', [ 'any', [ 'match', null, 'sub' ] ], 'a' ]
            ]
            ] ]
          ] ]
        );
      });

      test('super invocation', function() {
        unit(
          'ometa name { rule ^rule }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule', [
                [ 'super', [ 'match', null, 'rule' ] ]
            ]
            ] ]
          ] ]
        );
      });

      test('lookahead and named match', function() {
        unit(
          'ometa name { rule &a :b }',
          [ [ 'grammar',
            'name',
            null,
            [ [ 'rule',
              'rule',
              [ [ 'lookahead', [ 'match', null, 'a' ] ],
                [ 'arg', [ 'match', null, 'anything' ], 'b' ] ] ] ]
          ] ]
        );
      });

      test('lookahead and predicate match', function() {
        unit(
          'ometa name { rule &(a) ?b }',
          [ [ 'grammar',
            'name',
            null,
            [ [ 'rule',
              'rule',
              [ [ 'lookahead', [ 'choice', [ [ 'match', null, 'a' ] ] ] ],
                [ 'predicate', 'b'] ] ] ]
          ] ]
        );
      });

      test('one left arg and two right choices', function() {
        unit(
          'ometa name { rule :a = :b :c -> b | :d :e -> e }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule',
              [
                [ 'arg', [ 'match', null, 'anything' ], 'a' ],
                [ 'choice',
                  [
                    [ 'arg', [ 'match', null, 'anything' ], 'b' ],
                    [ 'arg', [ 'match', null, 'anything' ], 'c' ],
                    [ 'result', common.expressionify('b ')]
                  ],
                  [
                    [ 'arg', [ 'match', null, 'anything' ], 'd' ],
                    [ 'arg', [ 'match', null, 'anything' ], 'e' ],
                    [ 'result', common.expressionify('e ')]
                  ]
                ]
              ]
            ] ]
          ] ]
        );
      });

      test('one left arg and right arg in parens', function() {
        unit(
          'ometa name { rule :a = (:b :d) }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule',
              [
                [ 'arg', [ 'match', null, 'anything' ], 'a' ],
                [
                  'choice',
                  [
                    [ 'choice',
                      [
                        ['arg', [ 'match', null, 'anything' ], 'b'],
                        ['arg', [ 'match', null, 'anything' ], 'd']
                      ]
                    ]
                  ]
                ]
              ]
            ] ]
          ] ]
        );
      });

      test('three args in parens', function() {
        unit(
          'ometa name { rule (:b | c | :d) }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule',
              [
                [
                  'choice',
                  [ ['arg', [ 'match', null, 'anything' ], 'b'] ],
                  [ ['match', null, 'c'] ],
                  [ ['arg', [ 'match', null, 'anything' ], 'd'] ]
                ]
              ]
            ] ]
          ] ]
        );
      });

      test('list match', function() {
        unit(
          'ometa name { rule [:a [#123 :b] :c] }',
          [ [
            'grammar',
            'name',
            null,
            [ [
              'rule', 'rule',
              [
                [
                  'list',
                  ['arg', [ 'match', null, 'anything' ], 'a'],
                  [
                    'list',
                    ['string', '123'],
                    ['arg', [ 'match', null, 'anything' ], 'b']
                  ],
                  ['arg', [ 'match', null, 'anything' ], 'c']
                ]
              ]
            ] ]
          ] ]
        );
      });
    });

    suite('grammars with a rule with host code in', function() {
      test('body of rule', function() {
        unit(
          'ometa name { rule -> { x = y * x + fn(#1,2,3); } }',
          [ [
            'grammar', 'name', null,
            [ [ 'rule', 'rule', [[
              'result',
              common.expressionify('{ x = y * x + fn("1",2,3); } ')
            ]] ] ]
          ] ]
        );
      });

      test('arg of rule', function() {
        unit(
          'ometa name { rule { x = y * x + fn(1,2,3); } }',
          [ [
            'grammar', 'name', null,
            [ [ 'rule', 'rule', [[
              'body',
              common.expressionify('x = y * x + fn(1,2,3); ')
            ]] ] ]
          ] ]
        );
      });

      test('arg of rule (+named match after)', function() {
        unit(
          'ometa name { rule { x = y * x + fn(1,2,3); } :a }',
          [ [
            'grammar', 'name', null,
            [[
              'rule', 'rule',
              [
                [
                  'body',
                  common.expressionify('x = y * x + fn(1,2,3); ')
                ],
                [ 'arg', [ 'match', null, 'anything' ], 'a' ]
              ]
            ]]
          ] ]
        );
      });

      test('arg of invocation', function() {
        unit(
          'ometa name { rule another(1 + 2, [1,2,3].join(""),3):k -> k }',
          [ [
            'grammar',
            'name',
            null,
            [ [ 'rule',
              'rule',
              [
                [ 'arg',
                  [ 'call',
                    null,
                    'another',
                    [
                      common.expressionify('1 + 2'),
                      common.expressionify('[1,2,3].join("")'),
                      common.expressionify('3')
                    ]
                  ],
                  'k'
                ],
                [ 'result', common.expressionify('k ') ]
              ]
            ] ]
          ] ]
        );
      });
    });

    suite('real word grammars', function() {
      test('bs-ometa-compiler', function() {
        unit(common.loadFile('bs-ometa-compiler'), false);
      });
    });
  });
});
