var Buffer = require('buffer').Buffer
var fs = require('fs')
var test = require('tape')
var UAParser = require('ua-parser-js')

var http = require('../..')

test('headers', function (t) {
	http.get({
		path: '/testHeaders?Response-Header=bar&Response-Header-2=BAR2',
		headers: {
			'Test-Request-Header': 'foo',
			'Test-Request-Header-2': 'FOO2'
		}
	}, function (res) {
		var rawHeaders = []
		for (var i = 0; i < res.rawHeaders.length; i += 2) {
			var lowerKey = res.rawHeaders[i].toLowerCase()
			if (lowerKey.indexOf('test-') === 0)
				rawHeaders.push(lowerKey, res.rawHeaders[i + 1])
		}
		var header1Pos = rawHeaders.indexOf('test-response-header')
		t.ok(header1Pos >= 0, 'raw response header 1 present')
		t.equal(rawHeaders[header1Pos + 1], 'bar', 'raw response header value 1')
		var header2Pos = rawHeaders.indexOf('test-response-header-2')
		t.ok(header2Pos >= 0, 'raw response header 2 present')
		t.equal(rawHeaders[header2Pos + 1], 'BAR2', 'raw response header value 2')
		t.equal(rawHeaders.length, 4, 'correct number of raw headers')

		t.equal(res.headers['test-response-header'], 'bar', 'response header 1')
		t.equal(res.headers['test-response-header-2'], 'BAR2', 'response header 2')

		var buffers = []

		res.on('end', function () {
			var body = JSON.parse(Buffer.concat(buffers).toString())
			t.equal(body['test-request-header'], 'foo', 'request header 1')
			t.equal(body['test-request-header-2'], 'FOO2', 'request header 2')
			t.equal(Object.keys(body).length, 2, 'correct number of request headers')
			t.end()
		})

		res.on('data', function (data) {
			buffers.push(data)
		})
	})
})

test('content-type response header', function (t) {
	http.get('/testHeaders', function (res) {
		t.equal(res.headers['content-type'], 'application/json', 'content-type preserved')
		t.end()
	})
})

var browser = (new UAParser()).setUA(navigator.userAgent).getBrowser()
var browserName = browser.name
var browserVersion = browser.major
// The content-type header is broken when 'prefer-streaming' or 'allow-wrong-content-type'
// is passed in browsers that rely on xhr.overrideMimeType(), namely older chrome and newer safari
var wrongMimeType = ((browserName === 'Chrome' && browserVersion <= 42) ||
	((browserName === 'Safari' || browserName === 'Mobile Safari') && browserVersion >= 6))

test('content-type response header with forced streaming', function (t) {
	http.get({
		path: '/testHeaders',
		mode: 'prefer-streaming'
	}, function (res) {
		if (wrongMimeType)
			t.equal(res.headers['content-type'], 'text/plain; charset=x-user-defined', 'content-type overridden')
		else
			t.equal(res.headers['content-type'], 'application/json', 'content-type preserved')
		t.end()
	})
})