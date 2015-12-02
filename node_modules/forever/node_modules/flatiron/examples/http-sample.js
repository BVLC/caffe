var util = require('util'),
    flatiron = require('../lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.http);

app.router.get('/', function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.end('Hello world!\n');
});

app.router.post('/', function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.write('Hey, you posted some cool data!\n');
  this.res.end(util.inspect(this.req.body, true, 2, true) + '\n');
});

app.router.get('/sandwich/:type', function (type) {
  if (~['bacon', 'burger'].indexOf(type)) {
    this.res.writeHead(200, { 'Content-Type': 'text/plain' });
    this.res.end('Serving ' + type + ' sandwich!\n');
  }
  else {
    this.res.writeHead(404, { 'Content-Type': 'text/plain' });
    this.res.end('No such sandwich, sorry!\n');
  }
});

app.start(8080);

