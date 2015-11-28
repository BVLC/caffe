'use strict';

var expect            = require('chai').expect;
var ExpressServer     = require('../../../../lib/tasks/server/express-server');
var Promise           = require('../../../../lib/ext/promise');
var MockUI            = require('../../../helpers/mock-ui');
var MockProject       = require('../../../helpers/mock-project');
var MockWatcher       = require('../../../helpers/mock-watcher');
var MockServerWatcher = require('../../../helpers/mock-server-watcher');
var ProxyServer       = require('../../../helpers/proxy-server');
var chalk             = require('chalk');
var request           = require('supertest');
var net               = require('net');
var EOL               = require('os').EOL;
var nock              = require('nock');
var express           = require('express');

describe('express-server', function() {
  var subject, ui, project, proxy, nockProxy;
  nock.enableNetConnect();

  beforeEach(function() {
    this.timeout(10000);
    ui = new MockUI();
    project = new MockProject();
    proxy = new ProxyServer();
    subject = new ExpressServer({
      ui: ui,
      project: project,
      watcher: new MockWatcher(),
      serverWatcher: new MockServerWatcher(),
      serverRestartDelayTime: 100,
      serverRoot: './server',
      proxyMiddleware: function() {
        return proxy.handler.bind(proxy);
      },
      environment: 'development'
    });
  });

  afterEach(function() {
    try {
      subject.httpServer.close();
    } catch(err) { }
    try {
      proxy.httpServer.close();
    } catch(err) { }
  });

  describe('displayHost', function() {
    it('should use the specified host if specified', function() {
      expect(subject.displayHost('1.2.3.4')).to.equal('1.2.3.4');
    });

    it('should use the use localhost if host is not specified', function() {
      expect(subject.displayHost(undefined)).to.equal('localhost');
    });
  });

  describe('processAppMiddlewares', function() {
    it('has a good error message if a file exists, but does not export a function', function() {
      subject.project = {
        has:     function() { return true; },
        require: function() { return {};   }
      };

      expect(function() {
        subject.processAppMiddlewares();
      }).to.throw(TypeError, 'ember-cli expected ./server/index.js to be the entry for your mock or proxy server');
    });

    it('returns values returned by server/index', function(){
      subject.project = {
        has: function() { return true; },
        require: function() {
          return function(){ return 'foo'; };
        }
      };

      expect(subject.processAppMiddlewares()).to.equal('foo');
    });
  });

  describe('output', function() {
    this.timeout(40000);

    it('with ssl', function() {
      return subject.start({
        host: undefined,
        port: '1337',
        ssl: true,
        sslCert: 'tests/fixtures/ssl/server.crt',
        sslKey: 'tests/fixtures/ssl/server.key',
        baseURL: '/'
      }).then(function() {
        var output = ui.output.trim().split(EOL);
        expect(output[0]).to.equal('Serving on https://localhost:1337/');
      });
    });

    it('with proxy', function() {
      return subject.start({
        proxy: 'http://localhost:3001/',
        host: undefined,
        port: '1337',
        baseURL: '/'
      }).then(function() {
        var output = ui.output.trim().split(EOL);
        expect(output[1]).to.equal('Serving on http://localhost:1337/');
        expect(output[0]).to.equal('Proxying to http://localhost:3001/');
        expect(output.length).to.equal(2, 'expected only two lines of output');
      });
    });

    it('without proxy', function() {
      return subject.start({
        host: undefined,
        port: '1337',
        baseURL: '/'
      }).then(function() {
        var output = ui.output.trim().split(EOL);
        expect(output[0]).to.equal('Serving on http://localhost:1337/');
        expect(output.length).to.equal(1, 'expected only one line of output');
      });
    });

    it('with baseURL', function() {
      return subject.start({
        host: undefined,
        port: '1337',
        baseURL: '/foo'
      }).then(function() {
        var output = ui.output.trim().split(EOL);
        expect(output[0]).to.equal('Serving on http://localhost:1337/foo/');
        expect(output.length).to.equal(1, 'expected only one line of output');
      });
    });

    it('address in use', function() {
      var preexistingServer = net.createServer();
      preexistingServer.listen(1337);

      return subject.start({
        host: undefined,
        port: '1337'
      })
        .then(function() {
          expect(false, 'should have rejected');
        })
        .catch(function(reason) {
          expect(reason.message).to.equal('Could not serve on http://localhost:1337. It is either in use or you do not have permission.');
        })
        .finally(function() {
          preexistingServer.close();
        });
    });
  });

  describe('behaviour', function() {
    it('starts with ssl if ssl option is passed', function() {

      return subject.start({
        host: 'localhost',
        port: '1337',
        ssl: true,
        sslCert: 'tests/fixtures/ssl/server.crt',
        sslKey: 'tests/fixtures/ssl/server.key',
        baseURL: '/'
      })
        .then(function() {
          return new Promise(function(resolve, reject) {
            process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
            request('https://localhost:1337', {strictSSL: false}).
              get('/').expect(200, function(err, value) {
                process.env.NODE_TLS_REJECT_UNAUTHORIZED = '1';
                if(err) { reject(err);    }
                else    { resolve(value); }
              });
          });
        });
    }),


    it('app middlewares are processed before the proxy', function(done) {
      var expected = '/foo was hit';

      project.require = function() {
        return function(app) {
          app.use('/foo', function(req,res) {
            res.send(expected);
          });
        };
      };

      subject.start({
        proxy: 'http://localhost:3001/',
        host: undefined,
        port: '1337',
        baseURL: '/'
      })
        .then(function() {
          request(subject.app)
            .get('/foo')
            .set('accept', 'application/json, */*')
            .expect(function(res) {
              expect(res.text).to.equal(expected);
            })
            .end(function(err) {
              if (err) {
                return done(err);
              }
              expect(proxy.called).to.equal(false);
              done();
            });
        });
    });
    it('works with a regular express app', function(done) {
      var expected = '/foo was hit';

      project.require = function() {
        var app = express();
        app.use('/foo', function(req,res) {
          res.send(expected);
        });
        return app;
      };

      subject.start({
        proxy: 'http://localhost:3001/',
        host: undefined,
        port: '1337',
        baseURL: '/'
      })
        .then(function() {
          request(subject.app)
            .get('/foo')
            .set('accept', 'application/json, */*')
            .expect(function(res) {
              expect(res.text).to.equal(expected);
            })
            .end(function(err) {
              if (err) {
                return done(err);
              }
              expect(proxy.called).to.equal(false);
              done();
            });
        });
    });
    describe('with proxy', function() {
      beforeEach(function() {
        return subject.start({
          proxy: 'http://localhost:3001/',
          host: undefined,
          port: '1337',
          baseURL: '/'
        });
      });

      function bypassTest(app, url, done, responseCallback) {
        request(app)
          .get(url)
          .set('accept', 'text/html')
          .end(function(err, response) {
            if (err) {
              return done(err);
            }
            expect(proxy.called).to.equal(false);
            if (responseCallback) { responseCallback(response); }
            done();
          });
      }

      it('bypasses proxy for /', function(done) {
        bypassTest(subject.app, '/', done);
      });

      it('bypasses proxy for files that exist', function(done) {
        bypassTest(subject.app, '/test-file.txt', done, function(response) {
          expect(response.text.trim()).to.equal('some contents');
        });
      });

      function apiTest(app, method, url, done) {
        var req = request(app);
        return req[method].call(req, url)
          .set('accept', 'text/json')
          .end(function(err) {
            if (err) {
              return done(err);
            }

            expect(proxy.called, 'proxy receives the request');
            expect(proxy.lastReq.method).to.equal(method.toUpperCase());
            expect(proxy.lastReq.url).to.equal(url);
            done();
          });
      }
      it('proxies GET', function(done) {
        apiTest(subject.app, 'get', '/api/get', done);
      });
      it('proxies PUT', function(done) {
        apiTest(subject.app, 'put', '/api/put', done);
      });
      it('proxies POST', function(done) {
        apiTest(subject.app, 'post', '/api/post', done);
      });
      it('proxies DELETE', function(done) {
        apiTest(subject.app, 'delete', '/api/delete', done);
      });
      // test for #1263
      it('proxies when accept contains */*', function(done) {
        request(subject.app)
          .get('/api/get')
          .set('accept', 'application/json, */*')
          .end(function(err) {
            if (err) {
              return done(err);
            }
            expect(proxy.called, 'proxy receives the request');
            done();
          });
      });
    });

    describe('proxy with subdomain', function() {
      beforeEach(function() {
        nockProxy = {
          called: null,
          method: null,
          url: null
        };

        return subject.start({
          proxy: 'http://api.lvh.me',
          host: undefined,
          port: '1337',
          baseURL: '/'
        });
      });

      function apiTest(app, method, url, done) {
        var req = request(app);
        return req[method].call(req, url)
          .set('accept', 'text/json')
          .end(function(err) {
            if (err) {
              return done(err);
            }
            expect(nockProxy.called, 'proxy receives the request');
            expect(nockProxy.method).to.equal(method.toUpperCase());
            expect(nockProxy.url).to.equal(url);
            done();
          });
      }

      it('proxies GET', function(done) {
        nock('http://api.lvh.me', {
          reqheaders: {
            'host': 'api.lvh.me'
          }
        }).get('/api/get')
          .reply(200, function() {
            nockProxy.called = true;
            nockProxy.method = 'GET';
            nockProxy.url = '/api/get';

            return '';
          });

        apiTest(subject.app, 'get', '/api/get', done);
      });
      it('proxies PUT', function(done) {
        nock('http://api.lvh.me', {
          reqheaders: {
            'host': 'api.lvh.me'
          }
        }).put('/api/put')
          .reply(204, function() {
            nockProxy.called = true;
            nockProxy.method = 'PUT';
            nockProxy.url = '/api/put';

            return '';
          });

        apiTest(subject.app, 'put', '/api/put', done);
      });
      it('proxies POST', function(done) {
        nock('http://api.lvh.me', {
          reqheaders: {
            'host': 'api.lvh.me'
          }
        }).post('/api/post')
          .reply(201, function() {
            nockProxy.called = true;
            nockProxy.method = 'POST';
            nockProxy.url = '/api/post';

            return '';
          });

        apiTest(subject.app, 'post', '/api/post', done);
      });
      it('proxies DELETE', function(done) {
        nock('http://api.lvh.me', {
          reqheaders: {
            'host': 'api.lvh.me'
          }
        }).delete('/api/delete')
          .reply(204, function() {
            nockProxy.called = true;
            nockProxy.method = 'DELETE';
            nockProxy.url = '/api/delete';

            return '';
          });

        apiTest(subject.app, 'delete', '/api/delete', done);
      });
      // test for #1263
      it('proxies when accept contains */*', function(done) {
        nock('http://api.lvh.me')
          .get('/api/get')
          .reply(200, function() {
            nockProxy.called = true;
            nockProxy.method = 'GET';
            nockProxy.url = '/api/get';

            return '';
          });

        request(subject.app)
          .get('/api/get')
          .set('accept', 'application/json, */*')
          .end(function(err) {
            if (err) {
              return done(err);
            }
            expect(nockProxy.called, 'proxy receives the request');
            done();
          });
      });
    });

    describe('without proxy', function() {
      function startServer(baseURL) {
        return subject.start({
          host: undefined,
          port: '1337',
          baseURL: baseURL || '/'
        });
      }

      it('serves index.html when file not found with auto/history location', function(done) {
        return startServer()
          .then(function() {
            request(subject.app)
              .get('/someurl.withperiod')
              .set('accept', 'text/html')
              .expect(200)
              .expect('Content-Type', /html/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('GET /tests serves tests/index.html for mime of */* (hash location)', function(done) {
        project._config = {
          baseURL: '/',
          locationType: 'hash'
        };

        return startServer()
          .then(function() {
            request(subject.app)
              .get('/tests')
              .set('accept', '*/*')
              .expect(200)
              .expect('Content-Type', /html/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('GET /tests serves tests/index.html for mime of */* (auto location)', function(done) {
        return startServer()
          .then(function() {
            request(subject.app)
              .get('/tests')
              .set('accept', '*/*')
              .expect(200)
              .expect('Content-Type', /html/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('GET /tests/whatever serves tests/index.html when file not found', function(done) {
        return startServer()
          .then(function() {
            request(subject.app)
              .get('/tests/whatever')
              .set('accept', 'text/html')
              .expect(200)
              .expect('Content-Type', /html/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('GET /tests/an-existing-file.tla serves tests/an-existing-file.tla if it is found', function(done) {
        return startServer()
          .then(function() {
            request(subject.app)
              .get('/tests/test-file.txt')
              .set('accept', 'text/html')
              .expect(200)
              .expect(/some contents/)
              .expect('Content-Type', /text/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('serves index.html when file not found (with baseURL) with auto/history location', function(done) {
        return startServer('/foo')
          .then(function() {
            request(subject.app)
              .get('/foo/someurl')
              .set('accept', 'text/html')
              .expect(200)
              .expect('Content-Type', /html/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('serves index.html when file not found (with baseURL) with custom history location', function(done) {
        project._config = {
          baseURL: '/',
          locationType: 'blahr',
          historySupportMiddleware: true
        };

        return startServer('/foo')
          .then(function() {
            request(subject.app)
              .get('/foo/someurl')
              .set('accept', 'text/html')
              .expect(200)
              .expect('Content-Type', /html/)
              .end(function(err) {
                if (err) {
                  return done(err);
                }
                done();
              });
          });
      });

      it('returns a 404 when file not found with hash location', function(done) {
        project._config = {
          baseURL: '/',
          locationType: 'hash'
        };

        return startServer()
          .then(function() {
            request(subject.app)
              .get('/someurl.withperiod')
              .set('accept', 'text/html')
              .expect(404)
              .end(done);
          });
      });

      it('files that exist in broccoli directory are served up', function(done) {
        return startServer()
          .then(function() {
            request(subject.app)
            .get('/test-file.txt')
            .end(function(err, response) {
              expect(response.text.trim()).to.equal('some contents');
              done();
            });
          });
      });

      it('serves static asset up from build output without a period in name', function(done) {
        return startServer()
          .then(function() {
            request(subject.app)
              .get('/someurl-without-period')
              .expect(200)
              .end(function(err, response) {
                if (err) {
                  return done(err);
                }

                expect(response.text.trim()).to.equal('some other content');

                done();
              });
          });
      });

      it('serves static asset up from build output without a period in name (with baseURL)', function(done) {
        return startServer('/foo')
          .then(function() {
            request(subject.app)
              .get('/foo/someurl-without-period')
              .expect(200)
              .end(function(err, response) {
                if (err) {
                  return done(err);
                }

                expect(response.text.trim()).to.equal('some other content');

                done();
              });
          });
      });
    });

    describe('addons', function() {
      var calls;
      beforeEach(function() {
        calls = 0;

        subject.processAddonMiddlewares = function() {
          calls++;
        };
      });

      it('calls processAddonMiddlewares upon start', function() {
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          expect(calls).to.equal(1);
        });
      });
    });

    describe('addon middleware', function() {
      var firstCalls;
      var secondCalls;
      beforeEach(function() {
        firstCalls = 0;
        secondCalls = 0;

        project.initializeAddons = function() { };
        project.addons = [{
            serverMiddleware: function() {
              firstCalls++;
            }
          }, {
            serverMiddleware: function() {
              secondCalls++;
            }
          }, {
            doesntGoBoom: null
          }];

      });

      it('calls serverMiddleware on the addons on start', function() {
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          expect(firstCalls).to.equal(1);
          expect(secondCalls).to.equal(1);
        });
      });

      it('calls serverMiddleware on the addons on restart', function() {
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          subject.changedFiles = ['bar.js'];
          return subject.restartHttpServer();
        }).then(function() {
          expect(firstCalls).to.equal(2);
          expect(secondCalls).to.equal(2);
        });
      });
    });

    describe('addon middleware is async', function(){
      var order = [];
      beforeEach(function() {
        project.initializeAddons = function() { };
        project.addons = [
          {
            serverMiddleware: function () {
              order.push('first');
            }
          },
          {
            serverMiddleware: function() {
              return new Promise(function(resolve) {
                setTimeout(function(){
                  order.push('second');
                  resolve();
                }, 50);
              });
            }
          }, {
            serverMiddleware: function() {
              order.push('third');
            }
          }
        ];
      });

      it('waits for async middleware to complete before the next middleware', function(){
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          expect(order[0]).to.equal('first');
          expect(order[1]).to.equal('second');
          expect(order[2]).to.equal('third');
        });
      });
    });

    describe('addon middleware bubble errors', function(){
      beforeEach(function() {
        project.initializeAddons = function() { };
        project.addons = [{
          serverMiddleware: function() {
            return Promise.reject('addon middleware fail');
          }
        }
        ];
      });
      it('up to server start', function(){
        return subject.start({
          host: undefined,
          port: '1337'
        })
          .catch(function(reason){
            expect(reason).to.equal('addon middleware fail');
          });
      });
    });

    describe('app middleware', function() {
      var passedOptions;
      var calls;

      beforeEach(function() {
        passedOptions = null;
        calls = 0;

        subject.processAppMiddlewares = function(options) {
          passedOptions = options;
          calls++;
        };
      });

      it('calls processAppMiddlewares upon start', function() {
        var realOptions = {
          host: undefined,
          port: '1337'
        };

        return subject.start(realOptions).then(function() {
          expect(passedOptions === realOptions).to.equal(true);
          expect(calls).to.equal(1);
        });
      });

      it('calls processAppMiddlewares upon restart', function() {
        var realOptions = {
          host: undefined,
          port: '1337'
        };

        var originalApp;

        return subject.start(realOptions)
          .then(function() {
            originalApp = subject.app;
            subject.changedFiles = ['bar.js'];
            return subject.restartHttpServer();
          })
          .then(function() {
            expect(subject.app);
            expect(originalApp).to.not.equal(subject.app);
            expect(passedOptions === realOptions).to.equal(true);
            expect(calls).to.equal(2);
          });
      });

      it('includes httpServer instance in options', function() {
        var passedOptions;

        subject.processAppMiddlewares = function(options) {
          passedOptions = options;
        };

        var realOptions = {
          host: undefined,
          port: '1337'
        };

        return subject.start(realOptions).then(function() {
          expect(!!passedOptions.httpServer.listen);
        });
      });
    });

    describe('serverWatcherDidChange', function() {
      it('is called on file change', function() {
        var calls = 0;
        subject.serverWatcherDidChange = function() {
          calls++;
        };

        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          subject.serverWatcher.emit('change', 'foo.txt');
          expect(calls).to.equal(1);
        });
      });

      it('schedules a server restart', function() {
        var calls = 0;
        subject.scheduleServerRestart = function() {
          calls++;
        };

        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          subject.serverWatcher.emit('change', 'foo.txt');
          subject.serverWatcher.emit('change', 'bar.txt');
          expect(calls).to.equal(2);
        });
      });
    });

    describe('scheduleServerRestart', function() {
      it('schedules exactly one call of restartHttpServer', function(done) {
        var calls = 0;
        subject.restartHttpServer = function() {
          calls++;
        };

        subject.scheduleServerRestart();
        expect(calls).to.equal(0);
        setTimeout(function() {
          expect(calls).to.equal(0);
          subject.scheduleServerRestart();
        }, 50);
        setTimeout(function() {
          expect(calls).to.equal(1);
          done();
        }, 175);
      });
    });

    describe('restartHttpServer', function() {
      it('restarts the server', function() {
        var originalHttpServer;
        var originalApp;
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          ui.output = '';
          originalHttpServer = subject.httpServer;
          originalApp = subject.app;
          subject.changedFiles = ['bar.js'];
          return subject.restartHttpServer();
        }).then(function() {
          expect(ui.output).to.equal(EOL + chalk.green('Server restarted.') + EOL + EOL);
          expect(subject.httpServer, 'HTTP server exists');
          expect(subject.httpServer).to.not.equal(originalHttpServer, 'HTTP server has changed');
          expect(!!subject.app).to.equal(true, 'App exists');
          expect(subject.app).to.not.equal(originalApp, 'App has changed');
        });
      });

      it('restarts the server again if one or more files change during a previous restart', function() {
        var originalHttpServer;
        var originalApp;
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          originalHttpServer = subject.httpServer;
          originalApp = subject.app;
          subject.serverRestartPromise = new Promise(function(resolve) {
            setTimeout(function () {
              subject.serverRestartPromise = null;
              resolve();
            }, 20);
          });
          subject.changedFiles = ['bar.js'];
          return subject.restartHttpServer();
        }).then(function() {
          expect(!!subject.httpServer).to.equal(true, 'HTTP server exists');
          expect(subject.httpServer).to.not.equal(originalHttpServer, 'HTTP server has changed');
          expect(!!subject.app).to.equal(true, 'App exists');
          expect(subject.app).to.not.equal(originalApp, 'App has changed');
        });
      });

      it('emits the restart event', function() {
        var calls = 0;
        subject.on('restart', function() {
          calls++;
        });
        return subject.start({
          host: undefined,
          port: '1337'
        }).then(function() {
          subject.changedFiles = ['bar.js'];
          return subject.restartHttpServer();
        }).then(function() {
          expect(calls).to.equal(1);
        });
      });
    });
  });
});
