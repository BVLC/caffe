'use strict';

var fs     = require('fs-extra');
var path   = require('path');
var expect = require('chai').expect;
var exists = require('./');
var root   = process.cwd();
var tmpdir = path.join(root, 'tmp');

describe('exists-sync', function() {
  var tmp;
  beforeEach(function() {
      fs.mkdirsSync('./tmp');
      fs.writeFileSync('./tmp/taco.js', 'TACO!');
      fs.symlinkSync('./tmp/taco.js', './tmp/link-to-taco.js');
    });
    
    afterEach(function() {
      fs.deleteSync('./tmp');
    });
  it('verifies files exist', function() {
    expect(exists('./tmp/taco.js')).to.be.true;
    expect(exists('./taco.js')).to.be.false;
  });
  
  it('works with symlinks', function() {
    expect(exists('./tmp/link-to-taco.js'), 'symlink').to.be.true;
  });
  
});

describe('exists-sync symlinks', function(){ 
  var tmp;
  beforeEach(function() {
      fs.mkdirsSync('./tmp');
      fs.writeFileSync('./tmp/taco.js', 'TACO!');
      fs.writeFileSync('./tmp/burrito.js', 'BURRITO!');
      fs.symlinkSync('./tmp/taco.js', './tmp/link-to-taco.js');
      fs.symlinkSync('./tmp/burrito.js', './tmp/link-to-burrito.js');
    });
    
    afterEach(function() {
      fs.deleteSync('./tmp');
    });
  
  it('verifies symlink targets', function() {
    expect(exists('./tmp/link-to-burrito.js'), 'symlink').to.be.true;
    fs.deleteSync('./tmp/burrito.js');
    expect(exists('./tmp/link-to-burrito.js'), 'dead symlink').to.be.false;
  });
  
  it('verifies symlinked symlinks', function() {
    fs.symlinkSync('./tmp/link-to-taco.js', './tmp/link-to-taco-link.js');
    process.chdir(tmpdir);
    fs.symlinkSync('../link-to-taco.js', '../tmp/rel-link-to-taco.js');
    process.chdir(root);
    fs.mkdirSync('./tmp/symlinks');
    fs.symlinkSync('./tmp/link-to-taco-link.js', './tmp/symlinks/link-to-taco-link.js');
    
    expect(exists('./tmp/link-to-taco-link.js'), 'symlinked symlink').to.be.true;
    expect(exists('./tmp/rel-link-to-taco.js'), 'I heard you like relative symlinks').to.be.true;
    // symlink made from dir other than root
    expect(exists('./tmp/symlinks/link-to-taco-link.js'), 'I heard you like bad symlinks').to.be.true;
  });
  
  it('guards against cyclic symlinks', function() {
    fs.symlinkSync('./tmp/link-to-taco.js', './tmp/link-to-taco-back.js');
    fs.unlinkSync('./tmp/link-to-taco.js');
    fs.symlinkSync('./tmp/link-to-taco-back.js', './tmp/link-to-taco.js');
    expect(exists.bind(this,'./tmp/link-to-taco.js'), 'cyclic hell').to.throw(Error);//(/Circular symlink detected/);
  });
});