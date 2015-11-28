'use strict';

var styles = require('ansi-styles');
var path     = require('path');
var expect   = require('chai').expect;

/*eslint-env mocha*/

describe('markdown-it-terminal', function () {
  var md;
  beforeEach(function () {
    md = require('markdown-it')().use(require('../'));
  });
  
  it('renders basic markdown', function() {
    expect(md.render('# foo\n__bold__ **words**\n* un\n* ordered\n* list'))
      .to.equal('\u001b[35m\u001b[4m\u001b[1m\nfoo\u001b[22m\u001b[24m\u001b[39m\n'+
                '\u001b[0m\u001b[1mbold\u001b[22m \u001b[1mwords\u001b[22m\u001b[0m'+
                '\n\n   \u001b[0m* un\u001b[0m\n   \u001b[0m* ordered\u001b[0m\n   '+
                '\u001b[0m* list\u001b[0m\n\n');
  });
  
  it('renders nested styles', function() {
    expect(md.render('# foo __bold ~~del~~ and strong__'))
      .to.equal('\u001b[35m\u001b[4m\u001b[1m\nfoo \u001b[1mbold \u001b[2m\u001b[90m'+
                '\u001b[9mdel\u001b[29m\u001b[39m\u001b[22m and strong\u001b[22m'+
                '\u001b[22m\u001b[24m\u001b[39m\n');
  });
  
  it('renders html', function() {
    expect(md.render('<div>one</div>')).to.equal('\u001b[0m<div>one</div>\u001b[0m\n\n');
  });
  
  it('renders hr', function(){
    expect(md.render('---'))
      .to.have.string('\u001b[0m-----')
      .to.have.string('-----\u001b[0m\n');
    expect(md.render('***'))
      .to.have.string('\u001b[0m-----')
      .to.have.string('-----\u001b[0m\n');
    expect(md.render('___'))      
      .to.have.string('\u001b[0m-----')
      .to.have.string('-----\u001b[0m\n');
  });
  
  it('renders headers', function() {
    expect(md.render('# h1\n## h2\n### h3\n#### h4\n##### h5\n###### h6'))
      .to.equal('\u001b[35m\u001b[4m\u001b[1m\nh1\u001b[22m\u001b[24m\u001b[39m\n'+
                '\u001b[32m\u001b[1m\nh2\u001b[22m\u001b[39m\n\u001b[32m\u001b[1m'+
                '\nh3\u001b[22m\u001b[39m\n\u001b[32m\u001b[1m\nh4\u001b[22m'+
                '\u001b[39m\n\u001b[32m\u001b[1m\nh5\u001b[22m\u001b[39m\n'+
                '\u001b[32m\u001b[1m\nh6\u001b[22m\u001b[39m\n');
  });
//TODO: nested list test
  it('renders ordered list', function() {
    expect(md.render('1. Item 1\n2. Item 2\n3. Item 3'))
      .to.equal('   \u001b[0m1 Item 1\u001b[0m\n   \u001b[0m2 Item 2\u001b[0m\n'+
                '   \u001b[0m3 Item 3\u001b[0m\n\u001b[0m\n');
  });
  
  it('renders links', function(){
    expect(md.render('[test](http://www.test.com)')).to.equal('\u001b[0m\u001b[34mtest\u001b[39m\u001b[0m\n\n');
  });
  
  it('renders image links', function(){
    expect(md.render('![test](http://www.test.com/test.jpg)'))
      .to.equal('\u001b[0m\u001b[34m![http://www.test.com/test.jpg](http://www.test'+
                '.com/test.jpg) \u001b[39m\n\u001b[0m\n\n');
  });
  
  it('renders strong', function(){
    expect(md.render('**strong**')).to.equal('\u001b[0m\u001b[1mstrong\u001b[22m\u001b[0m\n\n');
    expect(md.render('__strong__')).to.equal('\u001b[0m\u001b[1mstrong\u001b[22m\u001b[0m\n\n');
  });
  
  it('renders em', function(){
    expect(md.render('*em*')).to.equal('\u001b[0m\u001b[3mem\u001b[23m\u001b[0m\n\n');
    expect(md.render('_em_')).to.equal('\u001b[0m\u001b[3mem\u001b[23m\u001b[0m\n\n');
  });
  
  it('renders s (del)', function(){     
    expect(md.render('~~strike~~'))
      .to.equal('\u001b[0m\u001b[2m\u001b[90m\u001b[9mstrike\u001b[29m\u001b[39m\u001b[22m\u001b[0m\n\n');
  });
  
  it('renders code inline', function(){
    expect(md.render('`var foo = blah;`')).to.equal('\u001b[0m\u001b[33mvar foo = blah;\u001b[39m\u001b[0m\n\n');
  });
  
  it('renders code blocks', function(){
    expect(md.render('```\nvar foo = blah;\nfunction bar() {\n   return "bar";\n}\n```'))
      .to.equal('\n\u001b[33mvar foo = blah;\nfunction bar() {\n   return "bar";\n}\n\u001b[39m\n\n');
  });
  
  it('renders code highlighting', function(){    
    expect(md.render('```js\nvar foo = blah;\nfunction bar() {\n   return "bar";\n}\n```'))
      .to.equal('\n\u001b[33m\u001b[32mvar\u001b[39m \u001b[37mfoo\u001b[39m \u001b[93m=\u001b[39m'+
      ' \u001b[37mblah\u001b[39m\u001b[90m;\u001b[39m\n\u001b[94mfunction\u001b[39m \u001b[37m'+
      'bar\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m\n   \u001b[31m'+
      'return\u001b[39m \u001b[92m"bar"\u001b[39m\u001b[90m;\u001b[39m\n\u001b[33m}\u001b[39m\n\u001b[39m\n\n');
  });
  
  it('allows overrides of basic styles', function() {
    var markdown = require('markdown-it')().use(require('../'),{styleOptions:{code:styles.green}});
    // console.log(markdown.render('`code should be green`'))
    expect(markdown.render('`code should be green`'))
      .to.equal('\u001b[0m\u001b[32mcode should be green\u001b[39m\u001b[0m\n\n');
  });
  
  it.skip('renders blue', function(){   
    console.log(md.render('<blue>content is blue and <red>this should be red</red> but this is blue</blue>'))  
    expect(md.render('<blue>blue</blue>'))
      .to.equal('\u001b[0m\u001b[2m\u001b[90m\u001b[9mstrike\u001b[29m\u001b[39m\u001b[22m\u001b[0m\n\n');
  });
});