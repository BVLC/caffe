var temp = require('temp');
temp.track();

console.log('Doing something');

describe('temp will create dir that will remain after the process exits', function() {
  it('creates a dir', function() {
    var p = temp.mkdirSync("shouldBeDeletedOnExit");
    console.log('created dir ' + p);
  });
});
