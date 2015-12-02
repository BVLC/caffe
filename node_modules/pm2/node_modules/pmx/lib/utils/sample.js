var Sample = function(size) {
  this._elements = [];
  this._size = size || 300;
  this._sum = 0;
  this._count = 0;
}

Sample.prototype.update = function(value) {
  if (this._count >= this._size) { 
    this._elements.push(value);
    this._sum += (value - this._elements.shift());
  }
  else {
    this._count++;
    this._elements.push(value);
    this._sum += value;
  }
}

Sample.prototype.getMean = function() {
  return (this._count === 0)
    ? 0
    : this._sum / this._count;
}

module.exports = Sample;
