/*

patchcharm.js
=============

A patch to charm to add text scrolling capability
and also a fix to `erase()`.

*/

var charm = require('charm')
var Charm = charm.Charm

var encode = function (xs) {
  function bytes (s) {
    if (typeof s === 'string') {
      return s.split('').map(ord);
    }
    else if (Array.isArray(s)) {
      return s.reduce(function (acc, c) {
        return acc.concat(bytes(c));
      }, []);
    }
  }
    
  return new Buffer([ 0x1b ].concat(bytes(xs)));
};

var ord = function ord (c) {
  return c.charCodeAt(0)
};


Charm.prototype.erase = function (s) {
  if (s === 'end' || s === '$') {
    this.write(encode('[K'));
  }
  else if (s === 'start' || s === '^') {
    this.write(encode('[1K'));
  }
  else if (s === 'line') {
    this.write(encode('[2K'));
  }
  else if (s === 'down') {
    this.write(encode('[J'));
  }
  else if (s === 'up') {
    this.write(encode('[1J'));
  }
  else if (s === 'screen') {
    this.write(encode('[2J'));
  }
  else {
    this.emit('error', new Error('Unknown erase type: ' + s));
  }
  return this;
};

Charm.prototype.enableScroll = function(){
  if (arguments.length === 0 || arguments[0] === 'screen')
    this.write(encode('[r'));
  else{
    var start = arguments[0],
      end = arguments[1];
    this.write(encode('[' + start + ';' + end + 'r'));
  }
  return this;
};

Charm.prototype.scrollUp = function(){
  this.write(encode('M'));
};

Charm.prototype.scrollDown = function(){
  this.write(encode('D'));
};
