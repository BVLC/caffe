## Changelog

### 1.4.0

* Enforce content-length header if set in response

### 1.3.0

* node v0.10 compatibility

### 1.2.0

* Make HTTP agent configurable.

### 1.1.10

* Re-tagging due to a problem with 1.1.9 being previously tagged

### 1.1.9

* More fixes to proxy support

### 1.1.8

* Fix proxy support when not using auth

### 1.1.7

* Automatic base64 encoding of proxy.auth into headers['proxy-authorization']
* Now properly sets headers on requests
* Moved tests to mocha

### 1.1.6

* Now returns 504 errors
* Only uses setTimeout if timeout value is > 0

### 1.1.5

* Added max_length setting (assumes bytes) that cancels
  the download if the file is growing too big

### 1.1.4

* Retain node v0.4.x compatibility.

### 1.1.3

* Now using 10 second timeout - tests using mocha

### 1.1.2

* Better error handling around invalid URLs

### 1.1.1

* Node 0.6.3 compatibility without warnings

### 1.1.0

* Returns Get instance as last parameter of `toDisk`, which
  assists with filetype-guessing

### 1.0.0

* Switched from deprecated `createClient()` API to new
  `http.request` API from node.
* Stronger support for HTTPS
* No longer supports node versions below 0.3.6

### 0.4.0

* Added `asBuffer()` method
* Streamlined `asDisk` to use node's native `.pipe()` function
* Added `encoding` option to constructor

### 0.4.0

* `.asBuffer()` added
* `get()` can now be used without `new`

### 0.3.0

* `get` now supports HTTP SOCKS proxies by setting `HTTP_PROXY` in `ENV`

### 0.2.0

* `node-get` is now `get`.

### 0.1.1

* [laplatrem](https://github.com/leplatrem): Fixed HTTPS support

### 0.1.0

* `max_redirs`, `headers` options in node-get constructor
* The API changes in 0.1.x - Get should never be expected to throw an exception.
* Handling of invalid URLs on redirect.
* Handling of file-level errors.

### 0.0.3

* Handling of DNS-level exceptions.

### 0.0.2

* Enhanced URL validation.
