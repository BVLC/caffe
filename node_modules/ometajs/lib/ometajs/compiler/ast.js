//
// ### function Ast ()
// Abstract Syntax Tree constructor
//
function Ast() {
  this.result = [];
  this.current = this.result;
  this.stack = [this.result];
};
module.exports = Ast;

//
// ### function create ()
// Creates Ast instance
//
Ast.create = function create() {
  return new Ast();
};

//
// ### function push (node)
// #### @node {Array} AST Node
// Pushes node into AST tree and stack
//
Ast.prototype.push = function push(node) {
  this.current.push(node);
  this.stack.push(node);
  this.current = node;
};

//
// ### function insert (...nodes...)
// Inserts nodes in current node without changing stack
//
Ast.prototype.insert = function insert() {
  this.current.push.apply(this.current, arguments);
};

//
// ### function wrap (node)
// #### @node {Array} AST Node
// Wraps curent node into passed one
//
Ast.prototype.wrap = function wrap(node) {
  this.pop();
  var current = node.concat([this.current.pop()]);
  this.push(current);
};

//
// ### function enter (node, body)
// #### @node {String} AST Node type
// #### @body {Function} a callback
// API Wrapper
//
Ast.prototype.enter = function enter(node, body) {
  if (Array.isArray(node)) {
    this.push(node);
  } else {
    this.push([node]);
  }
    body.call(this);
    this.pop();
};

//
// ### function pop ()
// Pops node from stack
//
Ast.prototype.pop = function pop() {
  var result = this.stack.pop();

  this.current = this.stack[this.stack.length - 1];

  return result;
};
