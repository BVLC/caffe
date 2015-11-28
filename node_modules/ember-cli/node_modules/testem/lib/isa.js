/*

isa.js
======

This is an `isa()` function that does simple and elegant type checking. 
Examples:

  isa([], Array) // true
  isa(function(){}, Function) // true
  isa("abc", String) // true

*/

function typeName(ctr){
  return ctr.name || String(ctr).match(/function (.{1,})\(/)[1]
}
var toString = Object.prototype.toString
function objTypeName(obj){
  return toString.call(obj).match(/^\[object (.*)\]$/)[1]
}

function isa(obj, type){
  if (obj === null || obj === undefined) return false
  return obj instanceof type || // the straight-forward case
    obj.constructor === type || // .constructor check to catch the primitives case
    objTypeName(obj) === typeName(type) // name-based check for the cross window case
    
}
module.exports = isa
