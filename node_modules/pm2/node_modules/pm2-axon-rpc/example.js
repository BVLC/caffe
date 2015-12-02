
exports.add = function(a, b, fn){
  fn(null, a + b);
};

exports.sub = function(a, b, fn){
  fn(null, a - b);
};

exports.uppercase = function(str, fn){
  fn(null, str.toUpperCase());
};

exports.lowercase = function(str, fn){
  fn(null, str.toLowerCase());
};