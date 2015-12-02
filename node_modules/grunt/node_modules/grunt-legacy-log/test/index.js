'use strict';

var legacyLog = require('../');
var Log = legacyLog.Log;

// Helper for testing stdout
var hooker = require('hooker');
function stdoutEqual(test, callback, expected) {
  var actual = '';
  // Hook process.stdout.write
  hooker.hook(process.stdout, 'write', {
    // This gets executed before the original process.stdout.write.
    pre: function(result) {
      // Concatenate uncolored result onto actual.
      actual += result;
      // Prevent the original process.stdout.write from executing.
      return hooker.preempt();
    },
  });
  // Execute the logging code to be tested.
  callback();
  // Restore process.stdout.write to its original value.
  stdoutUnmute();
  // Actually test the actually-logged stdout string to the expected value.
  // test.equal(legacyLog.uncolor(actual), expected);
  test.equal(actual, expected);
}

// Outright mute stdout.
function stdoutMute() {
  hooker.hook(process.stdout, 'write', {
    pre: function() {
      return hooker.preempt();
    },
  });
}

// Unmute stdout.
function stdoutUnmute() {
  hooker.unhook(process.stdout, 'write');
}

// Helper function: repeat('a', 3) -> 'aaa', repeat('a', 3, '-') -> 'a-a-a'
function repeat(str, n, separator) {
  var result = str;
  for (var i = 1; i < n; i++) {
    result += (separator || '') + str;
  }
  return result;
}

var fooBuffer = new Buffer('foo');

exports['Log instance'] = {
  setUp: function(done) {
    this.grunt = {fail: {errorcount: 0}};
    done();
  },
  'write': function(test) {
    test.expect(4);
    var log = new Log();

    stdoutEqual(test, function() { log.write(''); }, '');
    stdoutEqual(test, function() { log.write('foo'); }, 'foo');
    stdoutEqual(test, function() { log.write('%s', 'foo'); }, 'foo');
    stdoutEqual(test, function() { log.write(fooBuffer); }, 'foo');

    test.done();
  },
  'writeln': function(test) {
    test.expect(4);
    var log = new Log();

    stdoutEqual(test, function() { log.writeln(); }, '\n');
    stdoutEqual(test, function() { log.writeln('foo'); }, 'foo\n');
    stdoutEqual(test, function() { log.writeln('%s', 'foo'); }, 'foo\n');
    stdoutEqual(test, function() { log.writeln(fooBuffer); }, 'foo\n');

    test.done();
  },
  'warn': function(test) {
    test.expect(5);
    var log = new Log({grunt: this.grunt});

    stdoutEqual(test, function() { log.warn(); }, 'ERROR'.red + '\n');
    stdoutEqual(test, function() { log.warn('foo'); }, '>> '.red + 'foo\n');
    stdoutEqual(test, function() { log.warn('%s', 'foo'); }, '>> '.red + 'foo\n');
    stdoutEqual(test, function() { log.warn(fooBuffer); }, '>> '.red + 'foo\n');
    test.equal(this.grunt.fail.errorcount, 0);

    test.done();
  },
  'error': function(test) {
    test.expect(5);
    var log = new Log({grunt: this.grunt});

    stdoutEqual(test, function() { log.error(); }, 'ERROR'.red + '\n');
    stdoutEqual(test, function() { log.error('foo'); }, '>> '.red + 'foo\n');
    stdoutEqual(test, function() { log.error('%s', 'foo'); }, '>> '.red + 'foo\n');
    stdoutEqual(test, function() { log.error(fooBuffer); }, '>> '.red + 'foo\n');
    test.equal(this.grunt.fail.errorcount, 4);

    test.done();
  },
  'ok': function(test) {
    test.expect(4);
    var log = new Log({grunt: this.grunt});

    stdoutEqual(test, function() { log.ok(); }, 'OK'.green + '\n');
    stdoutEqual(test, function() { log.ok('foo'); }, '>> '.green + 'foo\n');
    stdoutEqual(test, function() { log.ok('%s', 'foo'); }, '>> '.green + 'foo\n');
    stdoutEqual(test, function() { log.ok(fooBuffer); }, '>> '.green + 'foo\n');

    test.done();
  },
  'errorlns': function(test) {
    test.expect(2);
    var log = new Log({grunt: this.grunt});

    stdoutEqual(test, function() {
      log.errorlns(repeat('foo', 30, ' '));
    }, '>> '.red + repeat('foo', 19, ' ') +
      '\n>> '.red + repeat('foo', 11, ' ') + '\n');
    test.equal(this.grunt.fail.errorcount, 1);

    test.done();
  },
  'oklns': function(test) {
    test.expect(1);
    var log = new Log();

    stdoutEqual(test, function() {
      log.oklns(repeat('foo', 30, ' '));
    }, '>> '.green + repeat('foo', 19, ' ') +
      '\n>> '.green + repeat('foo', 11, ' ') + '\n');

    test.done();
  },
  'success': function(test) {
    test.expect(4);
    var log = new Log();

    stdoutEqual(test, function() { log.success(); }, ''.green + '\n');
    stdoutEqual(test, function() { log.success('foo'); }, 'foo'.green + '\n');
    stdoutEqual(test, function() { log.success('%s', 'foo'); }, 'foo'.green + '\n');
    stdoutEqual(test, function() { log.success(fooBuffer); }, 'foo'.green + '\n');

    test.done();
  },
  'fail': function(test) {
    test.expect(4);
    var log = new Log();

    stdoutEqual(test, function() { log.fail(); }, ''.red + '\n');
    stdoutEqual(test, function() { log.fail('foo'); }, 'foo'.red + '\n');
    stdoutEqual(test, function() { log.fail('%s', 'foo'); }, 'foo'.red + '\n');
    stdoutEqual(test, function() { log.fail(fooBuffer); }, 'foo'.red + '\n');

    test.done();
  },
  'header': function(test) {
    test.expect(5);
    var log = new Log();

    stdoutEqual(test, function() { log.header(); }, ''.underline + '\n');
    stdoutEqual(test, function() { log.header(); }, '\n' + ''.underline + '\n');
    stdoutEqual(test, function() { log.header('foo'); }, '\n' + 'foo'.underline + '\n');
    stdoutEqual(test, function() { log.header('%s', 'foo'); }, '\n' + 'foo'.underline + '\n');
    stdoutEqual(test, function() { log.header(fooBuffer); }, '\n' + 'foo'.underline + '\n');

    test.done();
  },
  'subhead': function(test) {
    test.expect(5);
    var log = new Log();

    stdoutEqual(test, function() { log.subhead(); }, ''.bold + '\n');
    stdoutEqual(test, function() { log.subhead(); }, '\n' + ''.bold + '\n');
    stdoutEqual(test, function() { log.subhead('foo'); }, '\n' + 'foo'.bold + '\n');
    stdoutEqual(test, function() { log.subhead('%s', 'foo'); }, '\n' + 'foo'.bold + '\n');
    stdoutEqual(test, function() { log.subhead(fooBuffer); }, '\n' + 'foo'.bold + '\n');

    test.done();
  },
  'writetableln': function(test) {
    test.expect(1);
    var log = new Log();

    stdoutEqual(test, function() {
      log.writetableln([10], [repeat('foo', 10)]);
    }, 'foofoofoof\noofoofoofo\nofoofoofoo\n');

    test.done();
  },
  'writelns': function(test) {
    test.expect(1);
    var log = new Log();

    stdoutEqual(test, function() {
      log.writelns(repeat('foo', 30, ' '));
    }, repeat('foo', 20, ' ') + '\n' +
      repeat('foo', 10, ' ') + '\n');

    test.done();
  },
  'writeflags': function(test) {
    test.expect(3);
    var log = new Log();

    stdoutEqual(test, function() {
      log.writeflags(['a', 'b']);
    }, 'Flags: ' + 'a'.cyan + ', ' + 'b'.cyan + '\n');
    stdoutEqual(test, function() {
      log.writeflags(['a', 'b'], 'Prefix');
    }, 'Prefix: ' + 'a'.cyan + ', ' + 'b'.cyan + '\n');
    stdoutEqual(test, function() {
      log.writeflags({a: true, b: false, c: 0, d: null}, 'Prefix');
    }, 'Prefix: ' + 'a'.cyan + ', ' + 'b=false'.cyan + ', ' + 'c=0'.cyan + ', ' + 'd=null'.cyan + '\n');

    test.done();
  },
  'always': function(test) {
    test.expect(3);
    var log = new Log();

    test.strictEqual(log.always, log);
    test.strictEqual(log.verbose.always, log);
    test.strictEqual(log.notverbose.always, log);

    test.done();
  },
  'or': function(test) {
    test.expect(2);
    var log = new Log();

    test.strictEqual(log.verbose.or, log.notverbose);
    test.strictEqual(log.notverbose.or, log.verbose);

    test.done();
  },
  'hasLogged': function(test) {
    // Should only be true if output has been written!
    test.expect(24);
    var log = new Log();
    test.equal(log.hasLogged, false);
    test.equal(log.verbose.hasLogged, false);
    test.equal(log.notverbose.hasLogged, false);
    log.write('');
    test.equal(log.hasLogged, true);
    test.equal(log.verbose.hasLogged, true);
    test.equal(log.notverbose.hasLogged, true);

    log = new Log({verbose: true});
    log.verbose.write('');
    test.equal(log.hasLogged, true);
    test.equal(log.verbose.hasLogged, true);
    test.equal(log.notverbose.hasLogged, true);

    log = new Log();
    log.notverbose.write('');
    test.equal(log.hasLogged, true);
    test.equal(log.verbose.hasLogged, true);
    test.equal(log.notverbose.hasLogged, true);

    stdoutMute();
    log = new Log({debug: true});
    log.debug('');
    test.equal(log.hasLogged, true);
    test.equal(log.verbose.hasLogged, true);
    test.equal(log.notverbose.hasLogged, true);
    stdoutUnmute();

    // The following should be false since there's a verbose mismatch!
    log = new Log();
    log.verbose.write('');
    test.equal(log.hasLogged, false);
    test.equal(log.verbose.hasLogged, false);
    test.equal(log.notverbose.hasLogged, false);

    log = new Log({verbose: true});
    log.notverbose.write('');
    test.equal(log.hasLogged, false);
    test.equal(log.verbose.hasLogged, false);
    test.equal(log.notverbose.hasLogged, false);

    // The following should be false since there's a debug mismatch!
    log = new Log();
    log.debug('');
    test.equal(log.hasLogged, false);
    test.equal(log.verbose.hasLogged, false);
    test.equal(log.notverbose.hasLogged, false);

    test.done();
  },
  'muted': function(test) {
    test.expect(30);
    var log = new Log();

    test.equal(log.muted, false);
    test.equal(log.verbose.muted, false);
    test.equal(log.notverbose.muted, false);
    test.equal(log.options.muted, false);
    test.equal(log.verbose.options.muted, false);
    test.equal(log.notverbose.options.muted, false);

    log.muted = true;
    test.equal(log.muted, true);
    test.equal(log.verbose.muted, true);
    test.equal(log.notverbose.muted, true);
    test.equal(log.options.muted, true);
    test.equal(log.verbose.options.muted, true);
    test.equal(log.notverbose.options.muted, true);

    log.muted = false;
    test.equal(log.muted, false);
    test.equal(log.verbose.muted, false);
    test.equal(log.notverbose.muted, false);
    test.equal(log.options.muted, false);
    test.equal(log.verbose.options.muted, false);
    test.equal(log.notverbose.options.muted, false);

    log.options.muted = true;
    test.equal(log.muted, true);
    test.equal(log.verbose.muted, true);
    test.equal(log.notverbose.muted, true);
    test.equal(log.options.muted, true);
    test.equal(log.verbose.options.muted, true);
    test.equal(log.notverbose.options.muted, true);

    log.options.muted = false;
    test.equal(log.muted, false);
    test.equal(log.verbose.muted, false);
    test.equal(log.notverbose.muted, false);
    test.equal(log.options.muted, false);
    test.equal(log.verbose.options.muted, false);
    test.equal(log.notverbose.options.muted, false);

    test.done();
  },
  'verbose': function(test) {
    test.expect(15);
    var log = new Log();
    log.muted = true;

    // Test verbose methods to make sure they always return the verbose object.
    test.strictEqual(log.verbose.write(''), log.verbose);
    test.strictEqual(log.verbose.writeln(''), log.verbose);
    test.strictEqual(log.verbose.warn(''), log.verbose);
    test.strictEqual(log.verbose.error(''), log.verbose);
    test.strictEqual(log.verbose.ok(''), log.verbose);
    test.strictEqual(log.verbose.errorlns(''), log.verbose);
    test.strictEqual(log.verbose.oklns(''), log.verbose);
    test.strictEqual(log.verbose.success(''), log.verbose);
    test.strictEqual(log.verbose.fail(''), log.verbose);
    test.strictEqual(log.verbose.header(''), log.verbose);
    test.strictEqual(log.verbose.subhead(''), log.verbose);
    test.strictEqual(log.verbose.debug(''), log.verbose);
    test.strictEqual(log.verbose.writetableln([]), log.verbose);
    test.strictEqual(log.verbose.writelns(''), log.verbose);
    test.strictEqual(log.verbose.writeflags([]), log.verbose);

    test.done();
  },
  'notverbose': function(test) {
    test.expect(15);
    var log = new Log();
    log.muted = true;

    // Test notverbose methods to make sure they always return the notverbose object.
    test.strictEqual(log.notverbose.write(''), log.notverbose);
    test.strictEqual(log.notverbose.writeln(''), log.notverbose);
    test.strictEqual(log.notverbose.warn(''), log.notverbose);
    test.strictEqual(log.notverbose.error(''), log.notverbose);
    test.strictEqual(log.notverbose.ok(''), log.notverbose);
    test.strictEqual(log.notverbose.errorlns(''), log.notverbose);
    test.strictEqual(log.notverbose.oklns(''), log.notverbose);
    test.strictEqual(log.notverbose.success(''), log.notverbose);
    test.strictEqual(log.notverbose.fail(''), log.notverbose);
    test.strictEqual(log.notverbose.header(''), log.notverbose);
    test.strictEqual(log.notverbose.subhead(''), log.notverbose);
    test.strictEqual(log.notverbose.debug(''), log.notverbose);
    test.strictEqual(log.notverbose.writetableln([]), log.notverbose);
    test.strictEqual(log.notverbose.writelns(''), log.notverbose);
    test.strictEqual(log.notverbose.writeflags([]), log.notverbose);

    test.done();
  },
  'options.debug = true': function(test) {
    test.expect(4);
    var log = new Log({debug: true});

    stdoutEqual(test, function() { log.debug(); }, '[D] ' + ''.magenta + '\n');
    stdoutEqual(test, function() { log.debug('foo'); }, '[D] ' + 'foo'.magenta + '\n');
    stdoutEqual(test, function() { log.debug('%s', 'foo'); }, '[D] ' + 'foo'.magenta + '\n');
    stdoutEqual(test, function() { log.debug(fooBuffer); }, '[D] ' + 'foo'.magenta + '\n');

    test.done();
  },
  'options.verbose = false': function(test) {
    test.expect(7);
    var log = new Log({verbose: false});

    stdoutEqual(test, function() { log.notverbose.write('foo'); }, 'foo');
    stdoutEqual(test, function() { log.notverbose.write('%s', 'foo'); }, 'foo');
    stdoutEqual(test, function() { log.notverbose.write(fooBuffer); }, 'foo');
    stdoutEqual(test, function() { log.verbose.write('foo'); }, '');
    stdoutEqual(test, function() { log.verbose.write('%s', 'foo'); }, '');
    stdoutEqual(test, function() { log.verbose.write(fooBuffer); }, '');
    stdoutEqual(test, function() { log.verbose.write('a').or.write('b'); }, 'b');

    test.done();
  },
  'options.verbose = true': function(test) {
    test.expect(7);
    var log = new Log({verbose: true});

    stdoutEqual(test, function() { log.verbose.write('foo'); }, 'foo');
    stdoutEqual(test, function() { log.verbose.write('%s', 'foo'); }, 'foo');
    stdoutEqual(test, function() { log.verbose.write(fooBuffer); }, 'foo');
    stdoutEqual(test, function() { log.notverbose.write('foo'); }, '');
    stdoutEqual(test, function() { log.notverbose.write('%s', 'foo'); }, '');
    stdoutEqual(test, function() { log.notverbose.write(fooBuffer); }, '');
    stdoutEqual(test, function() { log.notverbose.write('a').or.write('b'); }, 'b');

    test.done();
  },
  'options.debug = false': function(test) {
    test.expect(1);
    var log = new Log({debug: false});

    stdoutEqual(test, function() { log.debug('foo'); }, '');

    test.done();
  },
  'options.color = true': function(test) {
    test.expect(1);
    var log = new Log({color: true});

    stdoutEqual(test, function() { log.write('foo'.blue + 'bar'.underline); }, 'foo'.blue + 'bar'.underline);

    test.done();
  },
  'options.color = false': function(test) {
    test.expect(1);
    var log = new Log({color: false});

    stdoutEqual(test, function() { log.write('foo'.blue + 'bar'.underline); }, 'foobar');

    test.done();
  },
  'perma-bind this when passing grunt in (backcompat)': function(test) {
    test.expect(43);
    var log = new Log({grunt: this.grunt});
    stdoutMute();
    [
      'write',
      'writeln',
      'warn',
      'error',
      'ok',
      'errorlns',
      'oklns',
      'success',
      'fail',
      'header',
      'subhead',
      'debug',
    ].forEach(function(method) {
      var fn = log[method];
      var verboseFn = log.verbose[method];
      var notVerboseFn = log.notverbose[method];
      test.equal(fn(), log, 'Should return log if invoked in a way where this is not log.');
      test.equal(verboseFn(), log.verbose, 'Should return log.verbose if invoked in a way where this is not log.');
      test.equal(notVerboseFn(), log.notverbose, 'Should return log.notverbose if invoked in a way where this is not log.');
    });

    test.doesNotThrow(function() { var fn = log.writetableln; fn([]); }, 'Should not throw if invoked in a way where this is not log.');
    test.doesNotThrow(function() { var fn = log.writelns; fn([]); }, 'Should not throw if invoked in a way where this is not log.');
    test.doesNotThrow(function() { var fn = log.writeflags; fn([]); }, 'Should not throw if invoked in a way where this is not log.');
    test.doesNotThrow(function() { var fn = log.wordlist; fn([]); }, 'Should not throw if invoked in a way where this is not log.');
    test.doesNotThrow(function() { var fn = log.uncolor; fn(''); }, 'Should not throw if invoked in a way where this is not log.');
    test.doesNotThrow(function() { var fn = log.wraptext; fn(1,''); }, 'Should not throw if invoked in a way where this is not log.');
    test.doesNotThrow(function() { var fn = log.table; fn([],''); }, 'Should not throw if invoked in a way where this is not log.');
    stdoutUnmute();

    test.done();
  },
};

exports['Helpers'] = {
  'uncolor': function(test) {
    test.expect(2);
    var log = new Log();
    test.ok(log.uncolor);
    test.strictEqual(log.uncolor, legacyLog.uncolor);
    test.done();
  },
  'wordlist': function(test) {
    test.expect(2);
    var log = new Log();
    test.ok(log.wordlist);
    test.strictEqual(log.wordlist, legacyLog.wordlist);
    test.done();
  },
  'wraptext': function(test) {
    test.expect(2);
    var log = new Log();
    test.ok(log.wraptext);
    test.strictEqual(log.wraptext, legacyLog.wraptext);
    test.done();
  },
  'table': function(test) {
    test.expect(2);
    var log = new Log();
    test.ok(log.table);
    test.strictEqual(log.table, legacyLog.table);
    test.done();
  },
};
