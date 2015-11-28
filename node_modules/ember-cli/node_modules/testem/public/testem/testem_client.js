/*

testem_client.js
================

The client-side script that reports results back to the Testem server via Socket.IO.
It also restarts the tests by refreshing the page when instructed by the server to do so.

*/

(function appendTestemIframeOnLoad() {
  var iframeAppended = false;

  var appendIframe = function() {
    if (iframeAppended) {
      return;
    }
    iframeAppended = true;
    var iframe = document.createElement('iframe');
    iframe.style.border = 'none';
    iframe.style.position = 'fixed';
    iframe.style.right = '5px';
    iframe.style.bottom = '5px';
    iframe.frameBorder = '0';
    iframe.allowTransparency = 'true';
    iframe.src = '/testem/connection.html';
    document.body.appendChild(iframe);
  };

  var DOMContentLoaded = function() {
    if (document.addEventListener) {
      document.removeEventListener('DOMContentLoaded', DOMContentLoaded, false);
    }
    else {
      document.detachEvent('onreadystatechange', DOMContentLoaded);
    }
    DOMReady();
  };

  var DOMReady = function() {
    if ( !document.body ) {
      return setTimeout( DOMReady, 1 );
    }
    appendIframe();
  };

  if (document.addEventListener) {
    document.addEventListener('DOMContentLoaded', DOMContentLoaded, false);
    window.addEventListener('load', DOMContentLoaded, false);
  } else if ( document.attachEvent ) {
    document.attachEvent('onreadystatechange', DOMContentLoaded);
    window.attachEvent('onload', DOMContentLoaded);
  }

  if (document.readyState !== 'loading') {
    DOMReady();
  }
})();

function initTestFrameworkHooks(socket){
  if (typeof getJasmineRequireObj === 'function'){
    jasmine2Adapter(socket)
  }else if (typeof jasmine === 'object'){
    jasmineAdapter(socket)
  }else if ((typeof mocha).match(/function|object/)){
    mochaAdapter(socket)
  }else if (typeof QUnit === 'object'){
    qunitAdapter(socket)
  }else if (typeof buster !== 'undefined'){
    busterAdapter(socket)
  }
}

function init(){
  takeOverConsole()
  interceptWindowOnError()
  initTestFrameworkHooks(Testem)
  setupTestStats()
}

function setupTestStats(){
  var originalTitle = document.title
  var total = 0
  var passed = 0
  Testem.on('test-result', function(test){
    total++
    if (test.failed === 0) passed++
    updateTitle()
  })

  function updateTitle(){
    if (!total) return
    document.title = originalTitle + ' (' + passed + '/' + total + ')'
  }
}

function takeOverConsole(){
  function intercept(method){
    var original = console[method]
    console[method] = function(){
      var doDefault, message
      var args = Array.prototype.slice.apply(arguments)
      if (Testem.handleConsoleMessage) {
        message = decycle(args).join(' ');
        doDefault = Testem.handleConsoleMessage(message);
      }
      if (doDefault !== false) {
        args.unshift(method);
        emit.apply(console, args);
        if (original && original.apply) {
          // Do this for normal browsers
          original.apply(console, arguments)
        }else if (original) {
          // Do this for IE
          if (!message) {
            message = decycle(args).join(' ');
          }
          original(message)
        }
      }
    }
  }
  var methods = ['log', 'warn', 'error', 'info']
  for (var i = 0; i < methods.length; i++) {
    if (window.console && console[methods[i]]) {
      intercept(methods[i])
    }
  }
}

function interceptWindowOnError(){
  var orginalOnError = window.onerror;
  window.onerror = function(msg, url, line){
    if (typeof msg === 'string' && typeof url === 'string' && typeof line === 'number'){
      emit('top-level-error', msg, url, line)
    }
    if (orginalOnError) {
      orginalOnError.apply(window, arguments)
    }
  };
}

function emit(){
  Testem.emit.apply(Testem, arguments)
}

window.Testem = {
  emitConnectionQueue: [],
  useCustomAdapter: function(adapter) {
    adapter(this);
  },
  emitConnection: function() {
    var args = Array.prototype.slice.call(arguments);
    Testem.emitConnectionQueue.push(args);
  },
  emit: function(evt) {
    var argsWithoutFirst = Array.prototype.slice.call(arguments, 1);

    if (this.evtHandlers && this.evtHandlers[evt]) {
      var handlers = this.evtHandlers[evt];
      for (var i = 0; i < handlers.length; i++) {
        var handler = handlers[i];
        handler.apply(this, argsWithoutFirst);
      }
    }
    Testem.emitConnection.apply(Testem, arguments);
  },
  on: function(evt, callback){
    if (!this.evtHandlers){
      this.evtHandlers = {}
    }
    if (!this.evtHandlers[evt]){
      this.evtHandlers[evt] = []
    }
    this.evtHandlers[evt].push(callback)
  },
  handleConsoleMessage: null
}

init()
