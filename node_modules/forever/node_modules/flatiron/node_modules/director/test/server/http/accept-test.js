/*
 * accept-test.js: Tests for `content-type`-based routing
 *
 * (C) 2012, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    apiEasy = require('api-easy'),
    director = require('../../../lib/director'),
    helpers = require('../helpers'),
    macros = helpers.macros,
    handlers = helpers.handlers;

var PORT = 9067;

apiEasy.describe('director/http/accept')
  .addBatch({
    "An instance of `director.http.Router`": {
      "with routes set up": {
        topic: function () {
          var router = new director.http.Router();
          router.get('/json', { accept: 'application/json' }, handlers.respondWithOk());
          router.get('/txt', { accept: 'text/plain' }, handlers.respondWithOk());
          router.get('/both', { accept: ['text/plain', 'application/json'] }, handlers.respondWithOk());
          router.get('/regex', { accept: /.+\/x\-.+/ }, handlers.respondWithOk());

          router.get('/weird', { accept: 'application/json' }, function () {
            this.res.writeHead(400);
            this.res.end();
          });

          router.get('/weird', handlers.respondWithOk());

          helpers.createServer(router).listen(PORT, this.callback);
        },
        "should be created": function (err) {
          assert(!err);
        }
      }
    }
  })
  .use('localhost', PORT)
  .discuss('with `content-type: application/json`')
  .setHeader('content-type', 'application/json')
    .get('/json')
      .expect(200)
    .get('/txt')
      .expect(404)
    .get('/both')
      .expect(200)
    .get('/regex')
      .expect(404)
    .get('/weird')
      .expect(400)
  .undiscuss()
  .next()
  .discuss('with `content-type: text/plain`')
  .setHeader('content-type', 'text/plain')
    .get('/json')
      .expect(404)
    .get('/txt')
      .expect(200)
    .get('/both')
      .expect(200)
    .get('/regex')
      .expect(404)
    .get('/weird')
      .expect(200)
  .undiscuss()
  .next()
  .discuss('with `content-type: application/x-tar-gz`')
  .setHeader('content-type', 'application/x-tar-gz`')
    .get('/json')
    .get('/json')
      .expect(404)
    .get('/txt')
      .expect(404)
    .get('/both')
      .expect(404)
    .get('/regex')
      .expect(200)
    .get('/weird')
      .expect(200)
  .undiscuss()
  .export(module);

