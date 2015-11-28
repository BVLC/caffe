/*

server.js
=========

Testem's server. Serves up the HTML, JS, and CSS required for
running the tests in a browser.

*/

var express = require('express')
var socketIO = require('socket.io-pure')
var fs = require('fs')
var path = require('path')
var async = require('async')
var log = require('npmlog')
var EventEmitter = require('events').EventEmitter
var path = require('path')
var Mustache = require('consolidate').mustache
var http = require('http')
var https = require('https')
var httpProxy = require('http-proxy')


function Server(config){
  this.config = config
  this.ieCompatMode = null

  // Maintain a hash of all connected sockets to close them manually
  // Workaround https://github.com/joyent/node/issues/9066
  this.sockets = {}
  this.nextSocketId = 0
}
Server.prototype = {
  __proto__: EventEmitter.prototype,
  start: function(callback){
    callback = callback || function(){}
    this.createExpress()

    var self = this
    // Start the server!
    // Create socket.io sockets
    this.server.on('listening', function() {
      self.config.set('port', self.server.address().port)
      callback(null)
      self.emit('server-start')
    });
    this.server.on('error', function(e) {
      self.stopped = true
      callback(e)
      self.emit('server-error', e)
    });
    this.server.on('connection', function (socket) {
      var socketId = self.nextSocketId++
      self.sockets[socketId] = socket
      socket.on('close', function () {
        delete self.sockets[socketId];
      });
    });
    this.server.listen(this.config.get('port'))
  },
  stop: function(callback){
    if (this.server && !this.stopped) {
      this.stopped = true
      this.server.close(callback)
      // Destroy all open sockets
      for (var socketId in this.sockets) {
        this.sockets[socketId].destroy();
      }
    }
    else {
      callback()
    }
  },
  createExpress: function(){
    var self = this
    var app = this.express = express()

    if(this.config.get('key') || this.config.get('pfx')) {
      var options = {}
      if (this.config.get('key')) {
        options.key = fs.readFileSync(this.config.get('key'))
        options.cert = fs.readFileSync(this.config.get('cert'))
      }
      else {
        options.pfx = fs.readFileSync(this.config.get('pfx'))
      }
      this.server = https.createServer(options, this.express)
    }
    else {
      this.server = http.createServer(this.express)
    }
    this.io = socketIO(this.server)

    this.io.on('connection', this.onClientConnected.bind(this))

    this.configureExpress(app)

    this.injectMiddleware(app)

    this.configureProxy(app)

    app.get('/', function(req, res){
      self.serveHomePage(req, res)
    })

    app.get(/\/([0-9]+)$/, function(req, res){
      self.serveHomePage(req, res)
    })

    app.get('/testem.js', function(req, res){
      self.serveTestemClientJs(req, res)
    })

    app.get(/^\/(?:[0-9]+)(\/.+)$/, serveStaticFile)
    app.post(/^\/(?:[0-9]+)(\/.+)$/, serveStaticFile)
    app.get(/^(.+)$/, serveStaticFile)
    app.post(/^(.+)$/, serveStaticFile)

    app.use(function(err, req, res, next){
      if (err){
        log.error(err.message)
        if (err.code === 'ENOENT'){
          res.status(404).send('Not found: ' + req.url)
        }else{
          res.status(500).send(err.message)
        }
      }else{
        next()
      }
    })

    function serveStaticFile(req, res){
      self.serveStaticFile(req.params[0], req, res)
    }
  },
  configureExpress: function(app){
    var self = this

    app.engine('mustache', Mustache)
    app.set("view options", {layout: false})
    app.use(function(req, res, next){
      if (self.ieCompatMode)
        res.setHeader('X-UA-Compatible', 'IE=' + self.ieCompatMode)
      next()
    })
    app.use(express.static(__dirname + '/../../public'))
  },
  injectMiddleware: function(app) {
    var middlewares = this.config.get('middleware')
    if (middlewares) {
      middlewares.forEach(function(middleware) {
        middleware(app);
      })
    }
  },
  shouldProxy: function(req, opts) {
    var accepts,
      acceptCheck = [
        'html',
        'css',
        'javascript',
      ];

    //Only apply filtering logic if 'onlyContentTypes' key is present
    if (!('onlyContentTypes' in opts)) {
      return true;
    }

    acceptCheck = acceptCheck.concat(opts.onlyContentTypes)
    acceptCheck.push('text')
    accepts = req.accepts(acceptCheck);
    if (accepts.indexOf(opts.onlyContentTypes) !== -1) {
      return true;
    }
    return false;
  },
  configureProxy: function(app) {
    var proxies = this.config.get('proxies');
    var self = this;
    if (proxies) {
      self.proxy = new httpProxy.createProxyServer({changeOrigin: true})

      Object.keys(proxies).forEach(function(url) {
        app.all(url + '*', function(req, res, next) {
          var opts = proxies[url];
          if (self.shouldProxy(req, opts)) {
            if (opts.host) {
              opts.target = 'http://' + opts.host + ':' + opts.port
              delete opts.host
              delete opts.port
            }
            self.proxy.web(req, res, opts)
          } else {
            next()
          }
        })
      });
    }
  },

  renderRunnerPage: function(err, files, res){
    var config = this.config
    var framework = config.get('framework') || 'jasmine'
    var css_files = config.get('css_files')
    var templateFile = {
      jasmine: 'jasminerunner',
      jasmine2: 'jasmine2runner',
      qunit: 'qunitrunner',
      mocha: 'mocharunner',
      'mocha+chai': 'mochachairunner',
      buster: 'busterrunner',
      custom: 'customrunner',
      tap: 'taprunner'
    }[framework] + '.mustache'
    res.render(__dirname + '/../../views/' + templateFile, {
      scripts: files,
      styles: css_files
    })
  },
  renderDefaultTestPage: function(req, res){
    res.header('Cache-Control', 'No-cache')
    res.header('Pragma', 'No-cache')


    var self = this
    var config = this.config
    var test_page = config.get('test_page')

    if (test_page){
      if (test_page[0] === "/") {
        test_page = encodeURIComponent(test_page)
      }
      var base = req.path === '/' ?
        req.path : req.path + '/'
      var url = base + test_page
      res.redirect(url)
    } else {
      config.getServeFiles(function(err, files){
        self.renderRunnerPage(err, files, res)
      })
    }
  },
  serveHomePage: function(req, res){
    var config = this.config
    var routes = config.get('routes') || config.get('route') || {}
    if (routes['/']){
      this.serveStaticFile('/', req, res)
    }else{
      this.renderDefaultTestPage(req, res)
    }
  },
  serveTestemClientJs: function(req, res){
    res.setHeader('Content-Type', 'text/javascript')

    res.write(';(function(){')
    var files = [
      'decycle.js',
      'jasmine_adapter.js',
      'jasmine2_adapter.js',
      'qunit_adapter.js',
      'mocha_adapter.js',
      'buster_adapter.js',
      'testem_client.js'
    ]
    async.forEachSeries(files, function(file, done){
      if (file.indexOf(path.sep) === -1) {
        file = __dirname + '/../../public/testem/' + file
      }
      fs.readFile(file, function(err, data){
        if (err){
          res.write('// Error reading ' + file + ': ' + err)
        }else{
          res.write('\n//============== ' + path.basename(file) + ' ==================\n\n')
          res.write(data)
        }
        done()
      })
    }, function(){
      res.write('}());')
      res.end()
    })

  },
  killTheCache: function killTheCache(req, res){
    res.setHeader('Cache-Control', 'No-cache')
    res.setHeader('Pragma', 'No-cache')
    delete req.headers['if-modified-since']
    delete req.headers['if-none-match']
  },
  route: function route(uri){
    var config = this.config
    var routes = config.get('routes') || config.get('route') || {}
    var bestMatchLength = 0
    var bestMatch = null
    for (var prefix in routes){
      if (uri.substring(0, prefix.length) === prefix){
        if (bestMatchLength < prefix.length){
          if (routes[prefix] instanceof Array) {
            routes[prefix].some(function(folder) {
              bestMatch = folder + '/' + uri.substring(prefix.length)
              return fs.existsSync(config.resolvePath(bestMatch))
            })
          } else {
            bestMatch = routes[prefix] + '/' + uri.substring(prefix.length)
          }
          bestMatchLength = prefix.length
        }
      }
    }
    return {
      routed: !!bestMatch,
      uri: bestMatch || uri.substring(1)
    }
  },
  serveStaticFile: function(uri, req, res){
    var self = this
    var config = this.config
    var routeRes = this.route(uri)
    uri = routeRes.uri
    var wasRouted = routeRes.routed
    this.killTheCache(req, res)
    var allowUnsafeDirs = config.get('unsafe_file_serving')
    var filePath = path.resolve(config.resolvePath(uri))
    var ext = path.extname(filePath)
    var isPathPermitted = filePath.indexOf(config.cwd()) !== -1
    if (!wasRouted && !allowUnsafeDirs && !isPathPermitted) {
      res.status(403)
      res.write('403 Forbidden')
      res.end()
    } else if (ext === '.mustache') {
      config.getTemplateData(function(err, data){
        res.render(filePath, data)
        self.emit('file-requested', filePath)
      })
    } else {
      fs.stat(filePath, function(err, stat){
        self.emit('file-requested', filePath)
        if (err) return res.sendFile(filePath)
        if (stat.isDirectory()){
          fs.readdir(filePath, function(err, files){
            var dirListingPage = __dirname + '/../../views/directorylisting.mustache'
            res.render(dirListingPage, {files: files})
          })
        }else{
          res.sendFile(filePath)
        }
      })
    }
  },
  onClientConnected: function(client){
    var self = this
    client.once('browser-login', function(browserName, id){
      log.info('New client connected: ' + browserName + ' ' + id)
      self.emit('browser-login', browserName, id, client)
    })
  }
}

module.exports = Server
