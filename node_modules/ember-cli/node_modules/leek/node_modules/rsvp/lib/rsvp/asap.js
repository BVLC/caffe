var len = 0;
var toString = {}.toString;
var vertxNext;
export default function asap(callback, arg) {
  queue[len] = callback;
  queue[len + 1] = arg;
  len += 2;
  if (len === 2) {
    // If len is 1, that means that we need to schedule an async flush.
    // If additional callbacks are queued before the queue is flushed, they
    // will be processed by this flush that we are scheduling.
    scheduleFlush();
  }
}

var browserWindow = (typeof window !== 'undefined') ? window : undefined;
var browserGlobal = browserWindow || {};
var BrowserMutationObserver = browserGlobal.MutationObserver || browserGlobal.WebKitMutationObserver;
var isNode = typeof process !== 'undefined' && {}.toString.call(process) === '[object process]';

// test for web worker but not in IE10
var isWorker = typeof Uint8ClampedArray !== 'undefined' &&
  typeof importScripts !== 'undefined' &&
  typeof MessageChannel !== 'undefined';

// node
function useNextTick() {
  var nextTick = process.nextTick;
  // node version 0.10.x displays a deprecation warning when nextTick is used recursively
  // setImmediate should be used instead instead
  var version = process.versions.node.match(/^(?:(\d+)\.)?(?:(\d+)\.)?(\*|\d+)$/);
  if (Array.isArray(version) && version[1] === '0' && version[2] === '10') {
    nextTick = setImmediate;
  }
  return function() {
    nextTick(flush);
  };
}

// vertx
function useVertxTimer() {
  return function() {
    vertxNext(flush);
  };
}

function useMutationObserver() {
  var iterations = 0;
  var observer = new BrowserMutationObserver(flush);
  var node = document.createTextNode('');
  observer.observe(node, { characterData: true });

  return function() {
    node.data = (iterations = ++iterations % 2);
  };
}

// web worker
function useMessageChannel() {
  var channel = new MessageChannel();
  channel.port1.onmessage = flush;
  return function () {
    channel.port2.postMessage(0);
  };
}

function useSetTimeout() {
  return function() {
    setTimeout(flush, 1);
  };
}

var queue = new Array(1000);
function flush() {
  for (var i = 0; i < len; i+=2) {
    var callback = queue[i];
    var arg = queue[i+1];

    callback(arg);

    queue[i] = undefined;
    queue[i+1] = undefined;
  }

  len = 0;
}

function attemptVertex() {
  try {
    var r = require;
    var vertx = r('vertx');
    vertxNext = vertx.runOnLoop || vertx.runOnContext;
    return useVertxTimer();
  } catch(e) {
    return useSetTimeout();
  }
}

var scheduleFlush;
// Decide what async method to use to triggering processing of queued callbacks:
if (isNode) {
  scheduleFlush = useNextTick();
} else if (BrowserMutationObserver) {
  scheduleFlush = useMutationObserver();
} else if (isWorker) {
  scheduleFlush = useMessageChannel();
} else if (browserWindow === undefined && typeof require === 'function') {
  scheduleFlush = attemptVertex();
} else {
  scheduleFlush = useSetTimeout();
}
