
2.0.1 / 2014-09-09
==================

 * fix Floating-point durations to setTimeout may cause infinite loop

2.0.0 / 2014-02-25
==================

 * refactor to use the AMP protocol. Closes #577
 * remove old codec support

1.0.0 / 2013-08-30
==================

* change Socket#connect() to use inaddr_any as well

0.6.1 / 2013-04-13
==================

  * fix Socket#close() callback support
  * add callback to reply() when peer is gone

0.6.0 / 2013-04-13
==================

  * add optional reply() callback. Closes #95
  * add support for optional req.send() callback. Closes #89

0.5.2 / 2013-04-09
==================

  * add `sock.queue` array for logging / debugging etc
  * fix connection queue flush which may drop messages on connection

0.5.1 / 2013-03-01
==================

  * add exit() to HWM example
  * add better HWM example
  * fix: ignore closed sockets on reply(). fixes #82

0.5.0 / 2013-01-01
==================

  * add HWM support. Closes #19
  * add ability to pass a callback in to the Socket.close method.
  * update benchmarks. Closes #72
  * remove batching

0.4.6 / 2012-11-15
==================

  * fix round-robin write to unwritable socket

0.4.5 / 2012-10-30
==================

  * add more network errors to be ignored
  * refactor `SubEmitter`
  * refactor `PubEmitter`
  * fix exponential backoff

0.4.4 / 2012-10-29
==================

  * fix round-robin global var leak for fallback function. Closes #66

0.4.3 / 2012-10-27
==================

  * add 30% throughput increase for sub-emitter by removing some indirection
  * fix `PubSocket#flushBatch()` in order to avoid writing to not writable sockets [AlexeyKupershtokh]

0.4.2 / 2012-10-18
==================

  * add 30% throughput increase for sub-emitter by removing some indirection
  * add escaping of regexp chars for `SubSocket#subscribe()`
  * fix non-multipart `SubEmitterSocket` logic

0.4.1 / 2012-10-16
==================

  * add removal of sockets on error
  * add handling of __ECONNRESET__, __ECONNREFUSED__, and __EPIPE__. Closes #17
  * add immediate closing of sockets on `.close()`
  * fix "bind" event. Closes #53
  * fix 'close' event for server sockets
  * remove "stream" socket type for now

0.4.0 / 2012-10-12
==================

  * add emitter wildcard support
  * add sub socket subscription support
  * add `pub-emitter`
  * add `sub-emitter`
  * perf: remove `.concat()` usage, ~10% gain
  * remove greetings

0.3.2 / 2012-10-08
==================

  * change prefix fix to `reply()` only

0.3.1 / 2012-10-08
==================

  * add fix for reply(undefined)

0.3.0 / 2012-10-05
==================

  * add `Socket#address()` to help with ephemeral port binding. Closes #39
  * add default identity of __PID__. Closes #35
  * remove examples for router/dealer

0.2.0 / 2012-09-27
==================

  * add default random `identity`
  * add `req.send()` callback support
  * remove router / dealer
  * change `ReqSocket` to round-robin send()s

0.1.0 / 2012-09-24
==================

  * add router socket [gjohnson]
  * add dealer socket [gjohnson]
  * add req socket [gjohnson]
  * add rep socket [gjohnson]
  * add multipart support [gjohnson]
  * add `.set()` / `.get()` configuration methods
  * add tcp://hostname:port support to .bind() and .connect(). Closes #16
  * add `make bm`
  * add Batch#empty()
  * remove Socket#option()

0.0.3 / 2012-07-14
==================

  * add resize example
  * add `debug()` instrumentation
  * add `PullSocket` bind support
  * add `Parser`
