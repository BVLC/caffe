'use strict';

var expect     = require('chai').expect;
var writeError = require('../../../lib/ui/write-error');
var MockUI     = require('../../helpers/mock-ui');
var BuildError = require('../../helpers/build-error');
var EOL        = require('os').EOL;
var chalk      = require('chalk');

describe('writeError', function() {
  var ui;

  beforeEach(function() {
    ui = new MockUI();
  });

  it('no error', function() {
    writeError(ui);

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal('');
  });

  it('error with message', function() {
    writeError(ui, new BuildError({
      message: 'build error'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('build error') + EOL);
  });

  it('error with stack', function() {
    writeError(ui, new BuildError({
      stack: 'the stack'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('Error') + EOL + 'the stack' + EOL);
  });

  it('error with file', function() {
    writeError(ui, new BuildError({
      file: 'the file'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('File: the file') + EOL + chalk.red('Error') + EOL);
  });

  it('error with filename (as from Uglify)', function() {
    writeError(ui, new BuildError({
      filename: 'the file'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('File: the file') + EOL + chalk.red('Error') + EOL);
  });

  it('error with file + line', function() {
    writeError(ui, new BuildError({
      file: 'the file',
      line: 'the line'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('File: the file (the line)') + EOL + chalk.red('Error') + EOL);
  });

  it('error with file + col', function() {
    writeError(ui, new BuildError({
      file: 'the file',
      col: 'the col'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('File: the file') + EOL + chalk.red('Error') + EOL);
  });

  it('error with file + line + col', function() {
    writeError(ui, new BuildError({
      file: 'the file',
      line: 'the line',
      col:  'the col'
    }));

    expect(ui.output).to.equal('');
    expect(ui.errors).to.equal(chalk.red('File: the file (the line:the col)') + EOL + chalk.red('Error') + EOL);
  });
});
