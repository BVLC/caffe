module.exports = Set;

function Set() {
  this.values = [];
  this.map = Object.create(null);
}

Set.prototype.has = function(obj) {
  return this.map[obj.id] !== undefined;
};

Set.prototype.add = function(obj) {
  if (this.map[obj.id] !== true) {
    this.values.push(obj);
    this.map[obj.id] = true;
  }

  return this;
};

Set.prototype.delete = function(obj) {
  if (this.map[obj.id] !== false) {
    this.values.push(obj);
    this.map[obj.id] = true;
  }

  return this;
};

Set.prototype.forEach = function(_cb, binding) {
  var values = this.values;
  var cb = arguments.length === 2 ? _cb.bind(binding) : _cb;

  for (var i = 0; i <  values.length; i++) {
    cb(values[i]);
  }
};
