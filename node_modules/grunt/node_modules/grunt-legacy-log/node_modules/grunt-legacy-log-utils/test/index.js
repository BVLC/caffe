'use strict';

var logUtils = require('../');

exports['Helpers'] = {
  setUp: function(done) {
    done();
  },
  'uncolor': function(test) {
    test.expect(1);

    test.equal(logUtils.uncolor('a'.red + 'b'.bold.green + 'c'.blue.underline), 'abc');

    test.done();
  },
  'wordlist': function(test) {
    test.expect(2);

    test.equal(logUtils.uncolor(logUtils.wordlist(['a', 'b'])), 'a, b');
    test.equal(logUtils.uncolor(logUtils.wordlist(['a', 'b'], {separator: '-'})), 'a-b');

    test.done();
  },
  'wraptext': function(test) {
    test.expect(8);

    // // I'm not writing out comprehensive unit tests for this right now.
    // function doAll(text) {
    //   console.log('==========');
    //   console.log('==========');
    //   [4, 6, 10, 15, 20, 25, 30, 40, 60].forEach(function(n) {
    //     doOne(n, text);
    //   });
    // }
    // function doOne(n, text) {
    //   console.log(new Array(n + 1).join('-'));
    //   console.log(logUtils.wraptext(n, text));
    // }
    // var text = 'this is '.red + 'a simple'.yellow.inverse + ' test of'.green + ' ' + 'some wrapped'.blue + ' text over '.inverse.magenta + 'many lines'.red;
    // doAll(text);
    // text = 'foolish '.red.inverse + 'monkeys'.yellow + ' eating'.green + ' ' + 'delicious'.inverse.blue + ' bananas '.magenta + 'forever'.red;
    // doAll(text);
    // text = 'foolish monkeys eating delicious bananas forever'.rainbow;
    // doAll(text);

    test.equal(logUtils.wraptext(2, 'aabbc'), 'aa\nbb\nc');
    test.equal(logUtils.wraptext(2, 'aabbcc'), 'aa\nbb\ncc');
    test.equal(logUtils.wraptext(3, 'aaabbbc'), 'aaa\nbbb\nc');
    test.equal(logUtils.wraptext(3, 'aaabbbcc'), 'aaa\nbbb\ncc');
    test.equal(logUtils.wraptext(3, 'aaabbbccc'), 'aaa\nbbb\nccc');
    test.equal(logUtils.uncolor(logUtils.wraptext(3, 'aaa'.blue + 'bbb'.green + 'c'.underline)), 'aaa\nbbb\nc');
    test.equal(logUtils.uncolor(logUtils.wraptext(3, 'aaa'.blue + 'bbb'.green + 'cc'.underline)), 'aaa\nbbb\ncc');
    test.equal(logUtils.uncolor(logUtils.wraptext(3, 'aaa'.blue + 'bbb'.green + 'ccc'.underline)), 'aaa\nbbb\nccc');

    test.done();
  },
  'table': function(test) {
    test.expect(1);

    test.equal(logUtils.table([3, 1, 5, 1, 8, 1, 12, 1, 20], [
      'a aa aaa aaaa aaaaa',
      '|||||||',
      'b bb bbb bbbb bbbbb',
      '|||||||',
      'c cc ccc cccc ccccc',
      '|||||||',
      'd dd ddd dddd ddddd',
      '|||||||',
      'e ee eee eeee eeeee eeeeee',
    ]), 'a  |b bb |c cc ccc|d dd ddd    |e ee eee eeee eeeee \n' +
        'aa |bbb  |cccc    |dddd ddddd  |eeeeee              \n' +
        'aaa|bbbb |ccccc   |            |\n' +
        'aaa|bbbbb|        |            |\n' +
        'a  |     |        |            |\n' +
        'aaa|     |        |            |\n' +
        'aa |     |        |            |');
    test.done();
  },
};
