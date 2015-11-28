var vlq = require('./vlq');
var fields = ['generatedColumn', 'source', 'originalLine', 'originalColumn', 'name'];

module.exports = Coder;
function Coder() {}

Coder.prototype.decode = function(mapping) {
  var value = this.rawDecode(mapping);
  var output = {};

  for (var i=0; i<fields.length;i++) {
    var field = fields[i];
    var prevField = 'prev_' + field;
    if (value.hasOwnProperty(field)) {
      output[field] = value[field];
      if (typeof this[prevField] !== 'undefined') {
        output[field] += this[prevField];
      }
      this[prevField] = output[field];
    }
  }
  return output;
};

Coder.prototype.encode = function(value) {
  var output = '';
  for (var i=0; i<fields.length;i++) {
    var field = fields[i];
    if (value.hasOwnProperty(field)){
      var prevField = 'prev_' + field;
      var valueField = value[field];
      if (typeof this[prevField] !== 'undefined') {
        output += vlq.encode(valueField-this[prevField]);
      } else {
        output += vlq.encode(valueField);
      }
      this[prevField] = valueField;
    }
  }
  return output;
};

Coder.prototype.resetColumn = function() {
  this.prev_generatedColumn = 0;
};

Coder.prototype.adjustLine = function(n) {
  this.prev_originalLine += n;
};

Coder.prototype.rawDecode = function(mapping) {
  var buf = {rest: 0};
  var output = {};
  var fieldIndex = 0;
  while (fieldIndex < fields.length && buf.rest < mapping.length) {
    vlq.decode(mapping, buf.rest, buf);
    output[fields[fieldIndex]] = buf.value;
    fieldIndex++;
  }
  return output;
};


Coder.prototype.copy = function() {
  var c = new Coder();
  var key;
  for (var i = 0; i < fields.length; i++) {
    key = 'prev_' + fields[i];
    c[key] = this[key];
  }
  return c;
};

Coder.prototype.serialize = function() {
  var output = '';
  for (var i=0; i<fields.length;i++) {
    var valueField = this['prev_' + fields[i]] || 0;
    output += vlq.encode(valueField);
  }
  return output;
};

Coder.prototype.add = function(other) {
  this._combine(other, function(a,b){return a + b; });
};

Coder.prototype.subtract = function(other) {
  this._combine(other, function(a,b){return a - b; });
};

Coder.prototype._combine = function(other, operation) {
  var key;
  for (var i = 0; i < fields.length; i++) {
    key = 'prev_' + fields[i];
    this[key] = operation((this[key] || 0), other[key] || 0);
  }
};

Coder.prototype.debug = function(mapping) {
  var buf = {rest: 0};
  var output = [];
  var fieldIndex = 0;
  while (fieldIndex < fields.length && buf.rest < mapping.length) {
    vlq.decode(mapping, buf.rest, buf);
    output.push(buf.value);
    fieldIndex++;
  }
  return output.join('.');
};
