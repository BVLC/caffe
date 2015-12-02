var http = require('http'),
    director = require('../lib/director');

var router = new director.http.Router();

var server = http.createServer(function (req, res) {
  req.chunks = [];
  req.on('data', function (chunk) {
    req.chunks.push(chunk.toString());
  });

  router.dispatch(req, res, function (err) {
    if (err) {
      res.writeHead(404);
      res.end();
    }

    console.log('Served ' + req.url);
  });
});

router.get(/foo/, function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.end('hello world\n');
});

router.post(/foo/, function () {
  this.res.writeHead(200, { 'Content-Type': 'application/json' });
  this.res.end(JSON.stringify(this.req.body));
});

server.listen(8080);
console.log('vanilla http server with director running on 8080');
