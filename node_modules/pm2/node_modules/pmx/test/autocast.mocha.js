
var autocast = require('../lib/utils/autocast.js');

describe('Autocast', function() {
  it('should cast', function() {
    autocast('2').should.be.Number;
    autocast('toto').should.be.String;
    autocast('true').should.be.Boolean;
    autocast(true).should.be.Boolean;
    autocast('{ val : "aight" }').should.be.String;
    autocast({ val : "aight" }).should.be.Object;
  });

  it('should cast object', function(done) {
    autocast({ val : '2' }).val.should.be.Number;

    var test_obj = {
      a : {
        val : {
          e : '2',
          b : 'test',
          bool : 'false',
          d : {
            a : 2
          },
          bool_raw : false
        }
      }
    };

    autocast(test_obj).a.val.e.should.be.Number;
    autocast(test_obj).a.val.b.should.be.String;
    autocast(test_obj).a.val.bool.should.be.Boolean;
    autocast(test_obj).a.val.bool_raw.should.be.Boolean;
    autocast(test_obj).a.val.d.a.should.be.Number;
    done();
  });

});
