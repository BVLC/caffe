# morgan

[![NPM Version][npm-image]][npm-url]
[![NPM Downloads][downloads-image]][downloads-url]
[![Build Status][travis-image]][travis-url]
[![Test Coverage][coveralls-image]][coveralls-url]
[![Gratipay][gratipay-image]][gratipay-url]

HTTP request logger middleware for node.js

> Named after [Dexter](http://en.wikipedia.org/wiki/Dexter_Morgan), a show you should not watch until completion.

## API

```js
var morgan = require('morgan')
```

### morgan(format, options)

Create a new morgan logger middleware function using the given `format` and `options`.
The `format` argument may be a string of a predefined name (see below for the names),
a string of a format string, or a function that will produce a log entry.

#### Options

Morgan accepts these properties in the options object.

#### immediate

Write log line on request instead of response. This means that a requests will
be logged even if the server crashes, _but data from the response (like the
response code, content length, etc.) cannot be logged_.

##### skip

Function to determine if logging is skipped, defaults to `false`. This function
will be called as `skip(req, res)`.

```js
// EXAMPLE: only log error responses
morgan('combined', {
  skip: function (req, res) { return res.statusCode < 400 }
})
```

##### stream

Output stream for writing log lines, defaults to `process.stdout`.

#### Predefined Formats

There are various pre-defined formats provided:

##### combined

Standard Apache combined log output.

```
:remote-addr - :remote-user [:date[clf]] ":method :url HTTP/:http-version" :status :res[content-length] ":referrer" ":user-agent"
```

##### common

Standard Apache common log output.

```
:remote-addr - :remote-user [:date[clf]] ":method :url HTTP/:http-version" :status :res[content-length]
```

##### dev

Concise output colored by response status for development use. The `:status`
token will be colored red for server error codes, yellow for client error
codes, cyan for redirection codes, and uncolored for all other codes.

```
:method :url :status :response-time ms - :res[content-length]
```

##### short

Shorter than default, also including response time.

```
:remote-addr :remote-user :method :url HTTP/:http-version :status :res[content-length] - :response-time ms
```

##### tiny

The minimal output.

```
:method :url :status :res[content-length] - :response-time ms
```

#### Tokens

##### Creating new tokens

To define a token, simply invoke `morgan.token()` with the name and a callback function. This callback function is expected to return a string value. The value returned is then available as ":type" in this case:
```js
morgan.token('type', function(req, res){ return req.headers['content-type']; })
```

Calling `morgan.token()` using the same name as an existing token will overwrite that token definition.

##### :date[format]

The current date and time in UTC. The available formats are:

  - `clf` for the common log format (`"10/Oct/2000:13:55:36 +0000"`)
  - `iso` for the common ISO 8601 date time format (`2000-10-10T13:55:36.000Z`)
  - `web` for the common RFC 1123 date time format (`Tue, 10 Oct 2000 13:55:36 GMT`)

If no format is given, then the default is `web`.

##### :http-version

The HTTP version of the request.

##### :method

The HTTP version of the request.

##### :referrer

The Referrer header of the request. This will use the standard mis-spelled Referer header if exists, otherwise Referrer.

##### :remote-addr

The remote address of the request. This will use `req.ip`, otherwise the standard `req.connection.remoteAddress` value (socket address).

##### :remote-user

The user authenticated as part of Basic auth for the request.

##### :req[header]

The given `header` of the request.

##### :res[header]

The given `header` of the response.

##### :response-time

The time between the request coming into `morgan` and when the response headers are written, in milliseconds.

##### :status

The status code of the response.

##### :url

The URL of the request. This will use `req.originalUrl` if exists, otherwise `req.url`.

##### :user-agent

The contents of the User-Agent header of the request.

### morgan.compile(format)

Compile a format string into a function for use by `morgan`. A format string
is a string that represents a single log line and can utilize token syntax.
Tokens are references by `:token-name`. If tokens accept arguments, they can
be passed using `[]`, for example: `:token-name[pretty]` would pass the string
`'pretty'` as an argument to the token `token-name`.

Normally formats are defined using `morgan.format(name, format)`, but for certain
advanced uses, this compile function is directly available.

## Examples

### express/connect

Simple app that will log all request in the Apache combined format to STDOUT

```js
var express = require('express')
var morgan = require('morgan')

var app = express()

app.use(morgan('combined'))

app.get('/', function (req, res) {
  res.send('hello, world!')
})
```

### vanilla http server

Simple app that will log all request in the Apache combined format to STDOUT

```js
var finalhandler = require('finalhandler')
var http = require('http')
var morgan = require('morgan')

// create "middleware"
var logger = morgan('combined')

http.createServer(function (req, res) {
  var done = finalhandler(req, res)
  logger(req, res, function (err) {
    if (err) return done(err)

    // respond to request
    res.setHeader('content-type', 'text/plain')
    res.end('hello, world!')
  })
})
```

### write logs to a file

#### single file

Simple app that will log all requests in the Apache combined format to the file
`access.log`.

```js
var express = require('express')
var fs = require('fs')
var morgan = require('morgan')

var app = express()

// create a write stream (in append mode)
var accessLogStream = fs.createWriteStream(__dirname + '/access.log', {flags: 'a'})

// setup the logger
app.use(morgan('combined', {stream: accessLogStream}))

app.get('/', function (req, res) {
  res.send('hello, world!')
})
```

#### log file rotation

Simple app that will log all requests in the Apache combined format to one log
file per date in the `log/` directory using the
[file-stream-rotator module](https://www.npmjs.com/package/file-stream-rotator).

```js
var FileStreamRotator = require('file-stream-rotator')
var express = require('express')
var fs = require('fs')
var morgan = require('morgan')

var app = express()
var logDirectory = __dirname + '/log'

// ensure log directory exists
fs.existsSync(logDirectory) || fs.mkdirSync(logDirectory)

// create a rotating write stream
var accessLogStream = FileStreamRotator.getStream({
  filename: logDirectory + '/access-%DATE%.log',
  frequency: 'daily',
  verbose: false
})

// setup the logger
app.use(morgan('combined', {stream: accessLogStream}))

app.get('/', function (req, res) {
  res.send('hello, world!')
})
```

### use custom token formats

Sample app that will use custom token formats. This adds an ID to all requests and displays it using the `:id` token.

```js
var express = require('express')
var morgan = require('morgan')
var uuid = require('node-uuid')

morgan.token('id', function getId(req) {
  return req.id
})

var app = express()

app.use(assignId)
app.use(morgan(':id :method :url :response-time'))

app.get('/', function (req, res) {
  res.send('hello, world!')
})

function assignId(req, res, next) {
  req.id = uuid.v4()
  next()
}
```

## License

[MIT](LICENSE)

[npm-image]: https://img.shields.io/npm/v/morgan.svg
[npm-url]: https://npmjs.org/package/morgan
[travis-image]: https://img.shields.io/travis/expressjs/morgan/master.svg
[travis-url]: https://travis-ci.org/expressjs/morgan
[coveralls-image]: https://img.shields.io/coveralls/expressjs/morgan/master.svg
[coveralls-url]: https://coveralls.io/r/expressjs/morgan?branch=master
[downloads-image]: https://img.shields.io/npm/dm/morgan.svg
[downloads-url]: https://npmjs.org/package/morgan
[gratipay-image]: https://img.shields.io/gratipay/dougwilson.svg
[gratipay-url]: https://www.gratipay.com/dougwilson/
