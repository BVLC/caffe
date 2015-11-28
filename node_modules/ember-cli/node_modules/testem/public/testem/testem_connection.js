
var socket, connectStatus = 'disconnected'

function syncConnectStatus(){
  var elm = document.getElementById('__testem_ui__')
  if (elm) elm.className = connectStatus
}

function startTests(){
  socket.disconnect()
  parent.window.location.reload()
}

function initUI(){
  var markup = 'TEST\u0027EM \u0027SCRIPTS!'
  var elm = document.createElement('div')
  elm.id = '__testem_ui__'
  elm.className = connectStatus
  elm.innerHTML = markup
  document.body.appendChild(elm)
}

function getBrowserName(userAgent){
  var regexs = [
    /MS(?:(IE) (1?[0-9]\.[0-9]))/,
    [/Trident\/.* rv:(1?[0-9]\.[0-9])/, function(m) {
      return ['IE', m[1]].join(' ')
    }],
    [/(OPR)\/([0-9]+\.[0-9]+)/, function(m){
      return ['Opera', m[2]].join(' ')
    }],
    /(Opera).*Version\/([0-9]+\.[0-9]+)/,
    /(Edge)\/([0-9]+\.[0-9]+)/,
    /(Chrome)\/([0-9]+\.[0-9]+)/,
    /(Firefox)\/([0-9a-z]+\.[0-9a-z]+)/,
    /(PhantomJS)\/([0-9]+\.[0-9]+)/,
    [/(Android).*Version\/([0-9]+\.[0-9]+).*(Safari)/, function(m){
      return [m[1], m[3], m[2]].join(' ')
    }],
    [/(iPhone).*Version\/([0-9]+\.[0-9]+).*(Safari)/, function(m){
      return [m[1], m[3], m[2]].join(' ')
    }],
    [/(iPad).*Version\/([0-9]+\.[0-9]+).*(Safari)/, function(m){
      return [m[1], m[3], m[2]].join(' ')
    }],
    [/Version\/([0-9]+\.[0-9]+).*(Safari)/, function(m){
      return [m[2], m[1]].join(' ')
    }]
  ]
  var defaultPick = function(m) {
    return m.slice(1).join(' ')
  }

  for (var i = 0; i < regexs.length; i++){
    var regex = regexs[i]
    var pick = defaultPick
    if (regex instanceof Array) {
      pick = regex[1]
      regex = regex[0]
    }
    var match = userAgent.match(regex)
    if (match){
      return pick(match)
    }
  }
  return userAgent
}

function getId(){
  var m = parent.location.pathname.match(/^\/([0-9]+)/)
  return m ? m[1] : null
}


var addListener = window.addEventListener ?
  function(obj, evt, cb){ obj.addEventListener(evt, cb, false) } :
  function(obj, evt, cb){ obj.attachEvent('on' + evt, cb) }

function init(){
  socket = io.connect({ reconnectionDelayMax: 1000, randomizationFactor: 0 })
  var id = getId()
  socket.emit('browser-login',
    getBrowserName(navigator.userAgent),
    id)
  socket.on('connect', function(){
    connectStatus = 'connected'
    syncConnectStatus()
  })
  socket.on('disconnect', function(){
    connectStatus = 'disconnected'
    syncConnectStatus()
  })
  socket.on('start-tests', startTests)
  addListener(window, 'load', initUI)

  while (parent.Testem.emitConnectionQueue.length > 0) {
    TestemConnection.emit.apply(this, parent.Testem.emitConnectionQueue.shift());
  }
  parent.Testem.emitConnection = TestemConnection.emit;
}

window.TestemConnection = {
  emit: function() {
    var args = Array.prototype.slice.apply(arguments);
    // Workaround IE 8 max instructions
    setTimeout(function() {
      var decycled = decycle(args);
      setTimeout(function() {
        socket.emit.apply(socket, decycled);
      }, 0);
    }, 0);
  }
}

init()
