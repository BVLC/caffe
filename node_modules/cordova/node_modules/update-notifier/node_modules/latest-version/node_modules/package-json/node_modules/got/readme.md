<h1 align="center">
	<br>
	<img width="360" src="https://rawgit.com/sindresorhus/got/master/media/logo.svg" alt="got">
	<br>
	<br>
	<br>
</h1>

> Simplified HTTP/HTTPS requests

[![Build Status](https://travis-ci.org/sindresorhus/got.svg?branch=master)](https://travis-ci.org/sindresorhus/got)

A nicer interface to the built-in [`http`](http://nodejs.org/api/http.html) module.

It supports following redirects, streams, automagically handling gzip/deflate and some convenience options.

Created because [`request`](https://github.com/mikeal/request) is bloated *(several megabytes!)* and slow.


## Install

```
$ npm install --save got
```


## Usage

```js
var got = require('got');

// Callback mode
got('todomvc.com', function (err, data, res) {
	console.log(data);
	//=> <!doctype html> ...
});


// Stream mode
got('todomvc.com').pipe(fs.createWriteStream('index.html'));

// For POST, PUT and PATCH methods got returns a WritableStream
fs.createReadStream('index.html').pipe(got.post('todomvc.com'));
```

### API

It's a `GET` request by default, but can be changed in `options`.

#### got(url, [options], [callback])

##### url

*Required*  
Type: `string`, `object`

The URL to request or bare [http.request options](https://nodejs.org/api/http.html#http_http_request_options_callback) object.

Properties from `options` will override properties in the parsed `url`.

##### options

Type: `object`

Any of the [`http.request`](http://nodejs.org/api/http.html#http_http_request_options_callback) options.

###### body

Type: `string`, `Buffer`, `ReadableStream`  

*This option and stream mode are mutually exclusive.*

Body that will be sent with a `POST` request. If present in `options` and `options.method` is not set - `options.method` will be set to `POST`.

If `content-length` or `transfer-encoding` is not set in `options.headers` and `body` is a string or buffer, `content-length` will be set to the body length.

###### encoding

Type: `string`, `null`  
Default: `'utf8'`

Encoding to be used on `setEncoding` of the response data. If `null`, the body is returned as a Buffer.

###### json

Type: `boolean`  
Default: `false`

*This option and stream mode are mutually exclusive.*

Parse response body with `JSON.parse` and set `accept` header to `application/json`.

###### query

Type: `string`, `object`  

Query string object that will be added to the request URL. This will override the query string in `url`.

###### timeout

Type: `number`

Milliseconds after which the request will be aborted and an error event with `ETIMEDOUT` code will be emitted.

###### agent

[http.Agent](http://nodejs.org/api/http.html#http_class_http_agent) instance.

If `undefined` - [`infinity-agent`](https://github.com/floatdrop/infinity-agent) will be used to backport Agent class from Node.js core.

To use default [globalAgent](http://nodejs.org/api/http.html#http_http_globalagent) just pass `null`.

##### callback(error, data, response)

###### error

`Error` object with HTTP status code as `code` property.

###### data

The data you requested.

###### response

The [response object](http://nodejs.org/api/http.html#http_http_incomingmessage).

When in stream mode, you can listen for events:

##### .on('response', response)

`response` event to get the response object of the final request.

##### .on('redirect', response, nextOptions)

`redirect` event to get the response object of a redirect. Second argument is options for the next request to the redirect location.

##### .on('error', error, body, response)

`error` event emitted in case of protocol error (like `ENOTFOUND` etc.) or status error (4xx or 5xx). Second argument is body of server response in case of status error. Third argument is response object.

###### response

The [response object](http://nodejs.org/api/http.html#http_http_incomingmessage).

#### got.get(url, [options], [callback])
#### got.post(url, [options], [callback])
#### got.put(url, [options], [callback])
#### got.patch(url, [options], [callback])
#### got.head(url, [options], [callback])
#### got.delete(url, [options], [callback])

Sets `options.method` to the method name and makes a request.


## Proxy

You can use the [`tunnel`](https://github.com/koichik/node-tunnel) module with the `agent` option to work with proxies:

```js
var got = require('got');
var tunnel = require('tunnel');

got('todomvc.com', {
	agent: tunnel.httpOverHttp({
		proxy: {
			host: 'localhost'
		}
	})
}, function () {});
```


## Tip

It's a good idea to set the `'user-agent'` header so the provider can more easily see how their resource is used. By default it's the URL to this repo.

```js
var got = require('got');

got('todomvc.com', {
	headers: {
		'user-agent': 'https://github.com/your-username/repo-name'
	}
}, function () {});
```


## Related

- [gh-got](https://github.com/sindresorhus/gh-got) - Convenience wrapper for interacting with the GitHub API
- [got-promise](https://github.com/floatdrop/got-promise) - Promise wrapper


## Created by

[![Sindre Sorhus](https://avatars.githubusercontent.com/u/170270?v=3&s=100)](http://sindresorhus.com) | [![Vsevolod Strukchinsky](https://avatars.githubusercontent.com/u/365089?v=3&s=100)](https://github.com/floatdrop)
---|---
[Sindre Sorhus](http://sindresorhus.com) | [Vsevolod Strukchinsky](https://github.com/floatdrop)


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
