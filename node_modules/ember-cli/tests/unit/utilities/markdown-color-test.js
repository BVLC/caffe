'use strict';

var MarkdownColor = require('../../../lib/utilities/markdown-color');
var expect        = require('chai').expect;
var path          = require('path');
var chalk         = require('chalk');

function isAnsiSupported() {
  // when ansi is supported this should be '\u001b[0m\u001b[31ma\u001b[39m\u001b[0m\n\n'
  return chalk.red('a') !== 'a';
}

(isAnsiSupported() ? describe : describe.skip)('MarkdownColor', function() {
  var mc;

  beforeEach(function() {
    /*
    // check to make sure ansi is supported
    // can use this.skip() after Mocha 2.1.1
    // see https://github.com/mochajs/mocha/pull/946
    if (isAnsiSupported()) {
      this.skip();
    }
    */
    mc = new MarkdownColor();
  });

  it('parses default markdown', function() {

    // console.log(mc.render('# foo\n__bold__ **words**\n* un\n* ordered\n* list'))
    expect(mc.render('# foo\n__bold__ words\n* un\n* ordered\n* list')).to.equal(
      '\u001b[35m\u001b[4m\u001b[1m\nfoo\u001b[22m\u001b[24m\u001b[39m\n\u001b[0m'+
      '\u001b[1mbold\u001b[22m words\u001b[0m\n\n   \u001b[0m* un\u001b[0m\n   '+
      '\u001b[0m* ordered\u001b[0m\n   \u001b[0m* list\u001b[0m\n\n');
  });

  it('parses color tokens', function() {
    expect(mc.render('<red>red</red>')).to.equal('\u001b[0m\u001b[31mred\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<green>green</green>')).to.equal('\u001b[0m\u001b[32mgreen\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<blue>blue</blue>')).to.equal('\u001b[0m\u001b[34mblue\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<cyan>cyan</cyan>')).to.equal('\u001b[0m\u001b[36mcyan\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<magenta>magenta</magenta>')).to.equal('\u001b[0m\u001b[35mmagenta\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<yellow>yellow</yellow>')).to.equal('\u001b[0m\u001b[33myellow\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<black>black</black>')).to.equal('\u001b[0m\u001b[30mblack\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<gray>gray</gray>')).to.equal('\u001b[0m\u001b[90mgray\u001b[39m\u001b[0m\n\n');
    expect(mc.render('<grey>grey</grey>')).to.equal('\u001b[0m\u001b[90mgrey\u001b[39m\u001b[0m\n\n');

    expect(mc.render('<bgRed>bgRed</bgRed>')).to.equal('\u001b[0m\u001b[41mbgRed\u001b[49m\u001b[0m\n\n');
    expect(mc.render('<bgGreen>bgGreen</bgGreen>')).to.equal('\u001b[0m\u001b[42mbgGreen\u001b[49m\u001b[0m\n\n');
    expect(mc.render('<bgBlue>bgBlue</bgBlue>')).to.equal('\u001b[0m\u001b[44mbgBlue\u001b[49m\u001b[0m\n\n');
    expect(mc.render('<bgCyan>bgCyan</bgCyan>')).to.equal('\u001b[0m\u001b[46mbgCyan\u001b[49m\u001b[0m\n\n');
    expect(mc.render('<bgMagenta>bgMagenta</bgMagenta>')).to.equal('\u001b[0m\u001b[45mbgMagenta\u001b[49m\u001b[0m\n\n');
    expect(mc.render('<bgYellow>bgYellow</bgYellow>')).to.equal('\u001b[0m\u001b[43mbgYellow\u001b[49m\u001b[0m\n\n');
    expect(mc.render('<bgBlack>bgBlack</bgBlack>')).to.equal('\u001b[0m\u001b[40mbgBlack\u001b[49m\u001b[0m\n\n');
  });

  it('parses custom tokens', function() {
    expect(mc.render('--option')).to.equal('\u001b[0m\u001b[36m--option\u001b[39m\u001b[0m\n\n');
    expect(mc.render('(Default: value)')).to.equal('\u001b[0m\u001b[36m(Default: value)\u001b[39m\u001b[0m\n\n');
    expect(mc.render('(Required)')).to.equal('\u001b[0m\u001b[36m(Required)\u001b[39m\u001b[0m\n\n');
  });

  it('accepts tokens on instantiation', function() {
    var mctemp = new MarkdownColor({
      tokens: {
        foo: {
          token: '^foo^',
          pattern: /(?:\^foo\^)(.*?)(?:\^foo\^)/g,
          render: MarkdownColor.prototype.renderStylesFactory(chalk, ['blue','bgWhite'])
        }
      }
    });
    expect(mctemp.render('^foo^foo^foo^')).to.equal('\u001b[0m\u001b[34m\u001b[47mfoo\u001b[49m\u001b[39m\u001b[0m\n\n');
  });

  it('parses markdown files', function() {
    // console.log(mc.renderFile(path.join(__dirname,'../../../tests/fixtures/markdown/foo.md')))
    expect(mc.renderFile(path.join(__dirname,'../../../tests/fixtures/markdown/foo.md'))).to
      .equal('\u001b[0m\u001b[36mtacos are \u001b[33mdelicious\u001b[36m \u001b[34mand I\u001b[39m enjoy eating them\u001b[39m\u001b[0m\n\n');
  });

  it('allows tokens inside other token bounds', function() {
    // console.log(mc.render('<cyan>tacos are <yellow>delicious</yellow> and I enjoy eating them</cyan>'))
    expect(mc.render('<cyan>tacos are <yellow>delicious</yellow> and I enjoy eating them</cyan>'))
      .to.equal('\u001b[0m\u001b[36mtacos are \u001b[33mdelicious\u001b[36m and I enjoy eating'+
      ' them\u001b[39m\u001b[0m\n\n');
  });
});
/* Chalk supported styles -
styles:
   { reset: { open: '\u001b[0m', close: '\u001b[0m', closeRe: /[0m/g },
     bold: { open: '\u001b[1m', close: '\u001b[22m', closeRe: /[22m/g },
     dim: { open: '\u001b[2m', close: '\u001b[22m', closeRe: /[22m/g },
     italic: { open: '\u001b[3m', close: '\u001b[23m', closeRe: /[23m/g },
     underline: { open: '\u001b[4m', close: '\u001b[24m', closeRe: /[24m/g },
     inverse: { open: '\u001b[7m', close: '\u001b[27m', closeRe: /[27m/g },
     hidden: { open: '\u001b[8m', close: '\u001b[28m', closeRe: /[28m/g },
     strikethrough: { open: '\u001b[9m', close: '\u001b[29m', closeRe: /[29m/g },
     black: { open: '\u001b[30m', close: '\u001b[39m', closeRe: /[39m/g },
     red: { open: '\u001b[31m', close: '\u001b[39m', closeRe: /[39m/g },
     green: { open: '\u001b[32m', close: '\u001b[39m', closeRe: /[39m/g },
     yellow: { open: '\u001b[33m', close: '\u001b[39m', closeRe: /[39m/g },
     blue: { open: '\u001b[34m', close: '\u001b[39m', closeRe: /[39m/g },
     magenta: { open: '\u001b[35m', close: '\u001b[39m', closeRe: /[39m/g },
     cyan: { open: '\u001b[36m', close: '\u001b[39m', closeRe: /[39m/g },
     white: { open: '\u001b[37m', close: '\u001b[39m', closeRe: /[39m/g },
     gray: { open: '\u001b[90m', close: '\u001b[39m', closeRe: /[39m/g },
     bgBlack: { open: '\u001b[40m', close: '\u001b[49m', closeRe: /[49m/g },
     bgRed: { open: '\u001b[41m', close: '\u001b[49m', closeRe: /[49m/g },
     bgGreen: { open: '\u001b[42m', close: '\u001b[49m', closeRe: /[49m/g },
     bgYellow: { open: '\u001b[43m', close: '\u001b[49m', closeRe: /[49m/g },
     bgBlue: { open: '\u001b[44m', close: '\u001b[49m', closeRe: /[49m/g },
     bgMagenta: { open: '\u001b[45m', close: '\u001b[49m', closeRe: /[49m/g },
     bgCyan: { open: '\u001b[46m', close: '\u001b[49m', closeRe: /[49m/g },
     bgWhite: { open: '\u001b[47m', close: '\u001b[49m', closeRe: /[49m/g },
     grey: { open: '\u001b[90m', close: '\u001b[39m', closeRe: /[39m/g } },

Use strip-ansi to check content in environments that don't support it?
*/
