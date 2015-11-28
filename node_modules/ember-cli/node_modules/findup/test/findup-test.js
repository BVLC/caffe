var assert = require('chai').assert,
  Path = require('path'),
  fs = require('fs'),
  findup = require('..');

describe('find-up', function(){
  var fixtureDir = Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c', 'b', 'a'),
    fsExists = fs.exists ? fs.exists : Path.exists;
  it('accept a function', function(done){
    findup(fixtureDir, function(dir, cb){
      return fsExists(Path.join(dir, 'config.json'), cb);
    }, function(err, file){
      assert.ifError(err);
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c'));
      done();
    });
  });

  it('accept a string', function(done){
    findup(fixtureDir, 'config.json', function(err, file){
      assert.ifError(err);
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c'));
      done();
    });
  });

  it('is usable with the Object syntax', function(done) {
    new findup.FindUp(fixtureDir, 'config.json', function(err, file){
      assert.ifError(err);
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c'));
      done();
    });
  });

  it('find several files when using with the EventEmitter syntax', function(done){
    var ee = new findup.FindUp(fixtureDir, 'config.json');
    ee.once('found', function(file){
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c'));

      ee.once('found', function(file){
        assert.equal(file, Path.join(__dirname, 'fixture'));

        ee.once('end', function(){
          done();
        });
      });
    });
  });

  it('return files in top dir', function(done){
    findup(fixtureDir, 'top.json', function(err, file){
      assert.ifError(err);
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c', 'b', 'a'));
      done();
    });
  });

  it('return files in root dir', function(done){
    findup(fixtureDir, 'dev', function(err, file){
      assert.ifError(err);
      assert.equal(file, '/');
      done();
    });
  });

  it('return an error when looking for non existiong files', function(done){
    findup(fixtureDir, 'toto.json', function(err, file){
      assert.isNotNull(err);
      done();
    });
  });

  it('return an error when looking in a non existing directory', function(done){
    findup('dsqkjfnqsdkjghq', 'toto.json', function(err, file){
      assert.isNotNull(err);
      done();
    });
  });

  it('trigger an error event when looking in a non existing directory', function(done){
    findup('dsqkjfnqsdkjghq', 'toto.json').on('error', function(err, files){
      assert.isNotNull(err);
      done();
    });
  });

  describe('Sync API', function(){
    it('accept a string', function(){
      var file = findup.sync(fixtureDir, 'config.json');
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c'));
    });

    it('return a file in top dir', function(){
      var file = findup.sync(fixtureDir, 'top.json');
      assert.equal(file, Path.join(__dirname, 'fixture', 'f', 'e', 'd', 'c', 'b', 'a'));
    });

    it('throw error when looking for a non existing file', function(){
      assert.throw(function(){
        findup.sync(fixtureDir, 'toto.json');
      });
    });

    it('throw error when looking for in a non existing directory', function(){
      assert.throw(function(){
        findup.sync('uhjhbjkg,nfg', 'toto.json');
      });
    });
  });
});
