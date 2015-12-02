
var app = require('../../../../../flatiron').app;

app.log.info('somethign');

app.get('/blah', function () {
  this.req;
  this.res;
  this.emit('next');
});

app.sockets(function () {
  this.on(['foo' 'bar'], function (socket) {

  })
});

(function (app, options, done) {
  app.get()
})();