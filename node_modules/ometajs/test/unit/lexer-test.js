var common = require('../fixtures/common'),
    assert = require('assert');

suite('Ometajs language lexer', function() {
  test('should parse source correctly', function() {
    assert.deepEqual([
      { type: 'name', value: 'abc', offset: 0 },
      { type: 'space', value: ' ', offset: 3 },
      { type: 'string', value: 'abc', offset: 4 },
      { type: 'space', value: ' ', offset: 8 },
      { type: 'string', value: '"a""\'', offset: 9 },
      { type: 'space', value: ' ', offset: 17 },
      { type: 'token', value: 'abc', offset: 18 },
      { type: 'space', value: ' ', offset: 23 },
      { type: 'punc', value: '{', offset: 24 },
      { type: 'punc', value: '}', offset: 25 },
      { type: 'space', value: ' ', offset: 26 },
      { type: 'punc', value: '[', offset: 27 },
      { type: 'string', value: 'a', offset: 28 },
      { type: 'space', value: ' ', offset: 30 },
      { type: 'punc', value: ']', offset: 31 },
      { type: 'space', value: '// 123\n', offset: 32 },
      { type: 'token', value: '123', offset: 39 },
      { type: 'space', value: '/* \n123\r */', offset: 44 },
      { type: 'punc', value: '[', offset: 55 },
      { type: 'string', value: '123', offset: 56 },
      { type: 'space', value: ' ', offset: 60 },
      { type: 'number', value: '123', offset: 61 },
      { type: 'punc', value: ']', offset: 64 },
      { type: 're', value: '/123abc/', offset: 65 },
      { type: 'space', value: ' ', offset: 73 },
      { type: 'punc', value: '/', offset: 74 }
    ], common.lexems('abc `abc \'"a""\\\'\' "abc" ' +
                     '{} [#a ]// 123\n"123"' +
                     '/* \n123\r */[#123 123]/123abc/ /'));
    assert.deepEqual([
      { type: 'punc', value: '(', offset: 0 },
      { type: 'name', value: 'a', offset: 1 },
      { type: 'space', value: ' ', offset: 2 },
      { type: 'punc', value: '/', offset: 3 },
      { type: 'space', value: ' ', offset: 4 },
      { type: 'name', value: 'b', offset: 5 },
      { type: 'punc', value: ')', offset: 6 },
      { type: 'space', value: '\n', offset: 7 },
      { type: 'space', value: '// b', offset: 8 },
    ], common.lexems('(a / b)\n// b'));
  });

});
