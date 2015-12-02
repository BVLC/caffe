
var assert = require("assert");
var shell  = require("shelljs");
var vizion = require("..");

/*
 To enable a sample test suite, remove the _disabled
 and fill in the strings.  One way to fetch these values is to
 create the sample directory, enter it as the directory,
 and then run this test suite (npm test).
 The test will return the expected value (a blank string),
 and the actual value, which can be then used as the string to
 test.
 */
var sample = {
	svn: {
		directory: "./test/fixtures/test_svn/",
		url: "https://github.com/jshkurti/vizionar_test",
		revision: "r3",
		comment: "dat commit though",
		branch: "vizionar_test",
		update_time: "2014-10-21T12:29:21.289Z"
	},
	hg: {
		directory: "./test/fixtures/test_hg/",
		url: "https://jshkurti@bitbucket.org/jshkurti/vizionar_test",
		revision: "0:a070c08854c3",
		comment: "Initial commit with contributors",
		branch: "default",
		update_time: "2014-10-21T12:42:31.017Z"
	}
};

describe("vizion.analyze()", function() {
  if (shell.which('svn')) {
	  it.skip("Pulling from Subversion", function(done) {
      this.timeout(5000);
		  vizion.analyze({folder: sample.svn.directory}, function(err, metadata) {
			  assert.equal(err, null);
			  assert.equal(metadata.url, sample.svn.url);
			  assert.equal(metadata.revision, sample.svn.revision);
			  assert.equal(metadata.comment, sample.svn.comment);
			  assert.equal(metadata.branch, sample.svn.branch);
			  done();
		  });
	  });
  }
  if (shell.which('hg')) {
	  it("Pulling from Mercurial", function(done) {
      this.timeout(5000);
		  vizion.analyze({folder: sample.hg.directory}, function(err, metadata) {
			  assert.equal(err, null);
			  assert.equal(metadata.url, sample.hg.url);
			  assert.equal(metadata.revision, sample.hg.revision);
			  assert.equal(metadata.comment, sample.hg.comment);
			  assert.equal(metadata.branch, sample.hg.branch);
			  done();
		  });
	  });
  }
});
