var test = require('../');

test('Harness async test support', function(t) {
  t.plan(3);

  t.ok(true, 'sync child A');

  t.test('sync child B', function(tt) {
    tt.plan(2);

    setTimeout(function(){
      tt.test('async grandchild A', function(ttt) {
        ttt.plan(1);
        ttt.ok(true);
      });
    }, 50);

    setTimeout(function() {
      tt.test('async grandchild B', function(ttt) {
        ttt.plan(1);
        ttt.ok(true);
      });
    }, 100);
  });

  setTimeout(function() {
    t.test('async child', function(tt) {
      tt.plan(2);
      tt.ok(true, 'sync grandchild in async child A');
      tt.test('sync grandchild in async child B', function(ttt) {
        ttt.plan(1);
        ttt.ok(true);
      });
    });
  }, 200);
});
