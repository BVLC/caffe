var prettyjson = process.env.EXPRESS_COV ? require('../lib-cov/prettyjson') : require('../lib/prettyjson');
var should = require('should');

describe('prettyjson general tests', function() {

  it("should output a string exactly equal as the input", function() {

    var input = 'This is a string';
    var output = prettyjson.render(input);

    output.should.equal(input);
  });

  it("should output a string with indentation", function() {

    var input = 'This is a string';
    var output = prettyjson.render(input, {}, 4);

    output.should.equal('    ' + input);
  });

  it("should output a multiline string with indentation", function() {

    var input = 'multiple\nlines'
    var output = prettyjson.render(input, {}, 4);

    output.should.equal('    """\n      multiple\n      lines\n    """');
  });

  it("should output an array of strings", function() {

    var input = ['first string', 'second string'];
    var output = prettyjson.render(input);

    output.should.equal([
      '- '.green + input[0],
      '- '.green + input[1]
    ].join('\n'));
  });

  it("should output an array of arrays", function() {

    var input = ['first string', ['nested 1', 'nested 2'], 'second string'];
    var output = prettyjson.render(input);

    output.should.equal([
      '- '.green + input[0],
      '- '.green,
      '  ' + '- '.green + input[1][0],
      '  ' + '- '.green + input[1][1],
      '- '.green + input[2]
    ].join('\n'));
  });

  it("should output a hash of strings", function() {

    var input = {param1: 'first string', param2: 'second string'};
    var output = prettyjson.render(input);

    output.should.equal([
      'param1: '.green + 'first string',
      'param2: '.green + 'second string'
    ].join('\n'));
  });

  it("should output a hash of hashes", function() {

    var input = {first_param: {subparam: 'first string', subparam2: 'another string'}, second_param: 'second string'};
    var output = prettyjson.render(input);

    output.should.equal([
      'first_param: '.green,
      '  ' + 'subparam: '.green + ' first string',
      '  ' + 'subparam2: '.green + 'another string',
      'second_param: '.green + 'second string'
    ].join('\n'));
  });

  it("should indent correctly the hashes keys", function() {

    var input = {very_large_param: 'first string', param: 'second string'};
    var output = prettyjson.render(input);

    output.should.equal([
      'very_large_param: '.green + 'first string',
      'param: '.green + '           second string'
    ].join('\n'));
  });

  it("should output a really nested object", function() {

    var input = {
      first_param: {
        subparam: 'first string',
        subparam2: 'another string',
        subparam3: ["different", "values", "in an array"]
      },
      second_param: 'second string',
      an_array: [{
        param3: 'value',
        param10: 'other value'
      }],
      empty_array: []
    };

    var output = prettyjson.render(input);

    output.should.equal([
      'first_param: '.green,
      '  ' + 'subparam: '.green + ' first string',
      '  ' + 'subparam2: '.green + 'another string',
      '  ' + 'subparam3: '.green,
      '    ' + '- '.green + 'different',
      '    ' + '- '.green + 'values',
      '    ' + '- '.green + 'in an array',
      'second_param: '.green + 'second string',
      'an_array: '.green,
      '  ' + '- '.green,
      '    ' + 'param3: '.green + ' value',
      '    ' + 'param10: '.green + 'other value',
      'empty_array: '.green,
      '  (empty array)'
    ].join('\n'));
  });

  it("should allow to configure colors for hash keys", function() {
    var input = {param1: 'first string', param2: 'second string'};
    var output = prettyjson.render(input, {keysColor: 'blue'});

    output.should.equal([
      'param1: '.blue + 'first string',
      'param2: '.blue + 'second string'
    ].join('\n'));
  });

  it("should allow to configure colors for numbers", function() {
    var input = {param1: 17, param2: 22.3};
    var output = prettyjson.render(input, {numberColor: 'red'});

    output.should.equal([
      'param1: '.green + '17'.red,
      'param2: '.green + '22.3'.red
    ].join('\n'));
  });

  it("should allow to configure rainbow as color", function() {
    var input = {param_long: 'first string', param2: 'second string'};
    var output = prettyjson.render(input, {keysColor: 'rainbow'});

    output.should.equal([
      'param_long: '.rainbow + 'first string',
      'param2: '.rainbow + '    second string'
    ].join('\n'));
  });

  it("should allow to configure the default indentation", function() {
    var input = {param: ['first string', "second string"]};
    var output = prettyjson.render(input, {defaultIndentation: 4});

    output.should.equal([
      'param: '.green,
      '    ' + '- '.green + 'first string',
      '    ' + '- '.green + 'second string'
    ].join('\n'));
  });

  it("should allow to configure the empty message for arrays", function() {
    var input = [];
    var output = prettyjson.render(input, {emptyArrayMsg: '(empty)'});

    output.should.equal([
      '(empty)'
    ].join('\n'));
  });

  it("should allow to configure colors for strings", function() {
    var input = {param1: 'first string', param2: 'second string'};
    var output = prettyjson.render(input, {keysColor: 'blue', stringColor: 'red'});

    output.should.equal([
      'param1: '.blue + 'first string'.red,
      'param2: '.blue + 'second string'.red
    ].join('\n'));
  });

  it("should allow to not use colors", function() {
    var input = {param1: 'first string', param2: ['second string']};
    var output = prettyjson.render(input, {noColor: true});

    output.should.equal([
      'param1: first string',
      'param2: ',
      '  - second string'
    ].join('\n'));
  });

  it("should allow to print simple arrays inline", function() {
    var input = {installs: ['first string', 'second string', false, 13]};
    var output = prettyjson.render(input, {inlineArrays: true});

    output.should.equal(
      'installs: '.green + 'first string, second string, false, 13');

    input = {installs: [ ['first string', 'second string'], 'third string']};
    output = prettyjson.render(input, {inlineArrays: true});

    output.should.equal([
      'installs: '.green,
      '  ' + '- '.green + 'first string, second string',
      '  ' + '- '.green + 'third string'
      ].join('\n'));
  });

  it("should not print an object prototype", function() {
    var Input = function() {
      this.param1 = 'first string';
      this.param2 = 'second string';
    };
    Input.prototype = {randomProperty: 'idontcare'};

    var output = prettyjson.render(new Input);

    output.should.equal([
      'param1: '.green + 'first string',
      'param2: '.green + 'second string'
    ].join('\n'));
  });
});

describe('Printing numbers, booleans and other objects', function() {
  it("should print numbers correctly ", function() {
    var input = 12345;
    var output = prettyjson.render(input, {}, 4);

    output.should.equal('    ' + '12345'.blue);
  });

  it("should print booleans correctly ", function() {
    var input = true;
    var output = prettyjson.render(input, {}, 4);

    output.should.equal('    ' + 'true'.green);

    input = false;
    output = prettyjson.render(input, {}, 4);

    output.should.equal('    ' + 'false'.red);
  });

  it("should print a null object correctly ", function() {
    var input = null;
    var output = prettyjson.render(input, {}, 4);

    output.should.equal('    ' + 'null'.grey);
  });

  it("should print an Error correctly ", function() {
    Error.stackTraceLimit = 1;
    var input = new Error('foo');
    var stack = input.stack.split('\n');
    var output = prettyjson.render(input, {}, 4);

    output.should.equal([
      '    ' + 'stack: '.green,
      '      ' + '- '.green + stack[0],
      '      ' + '- '.green + stack[1],
      '    ' + 'message: '.green + 'foo'
    ].join('\n'));
  });

  it('should print serializable items in an array inline', function() {
	var dt = new Date();
    var output = prettyjson.render([ 'a', 3, null, true, false, dt]);

    output.should.equal([
      '- '.green + 'a',
      '- '.green + '3'.blue,
      '- '.green + 'null'.grey,
      '- '.green + 'true'.green,
      '- '.green + 'false'.red,
	  '- '.green + dt
    ].join('\n'));
  });

  it('should print dates correctly', function() {
	var input = new Date();
	var expected = input.toString();
	var output = prettyjson.render(input, {}, 4);

	output.should.equal('    ' + expected);
  });

  it('should print dates in objects correctly', function() {
	var dt1 = new Date();
	var dt2 = new Date();

	var input = {
		dt1: dt2,
		dt2: dt2
	};

	var output = prettyjson.render(input, {}, 4);

	output.should.equal([
		'    ' + 'dt1: '.green + dt1.toString(),
		'    ' + 'dt2: '.green + dt2.toString()].join('\n'));
  });
});

describe('prettyjson.renderString() method', function(){
  it('should return an empty string if input is empty', function(){
    var input = '';

    var output = prettyjson.renderString(input);

    output.should.equal('');
  });

  it('should return an empty string if input is not a string', function(){
    var output = prettyjson.renderString({});
    output.should.equal('');
  });

  it('should return an error message if the input is an invalid JSON string', function(){
    var output = prettyjson.renderString('not valid!!');
    output.should.equal('Error:'.red + ' Not valid JSON!');
  });

  it('should return the prettyfied string if it is a valid JSON string', function(){
    var output = prettyjson.renderString('{"test": "OK"}');
    output.should.equal('test: '.green + 'OK');
  });

  it('should dismiss trailing characters which are not JSON', function(){
    var output = prettyjson.renderString('characters that are not JSON at all... {"test": "OK"}');
    output.should.equal("characters that are not JSON at all... \n" + 'test: '.green + 'OK');
  });

  it('should dismiss trailing characters which are not JSON with an array', function(){
    var output = prettyjson.renderString('characters that are not JSON at all... ["test"]');
    output.should.equal("characters that are not JSON at all... \n" + '- '.green + 'test');
  });

  it('should be able to accept the options parameter', function(){
    var output = prettyjson.renderString('{"test": "OK"}', {stringColor: 'red'});
    output.should.equal('test: '.green + 'OK'.red);
  });
});
