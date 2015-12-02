var parser = require("./");
var editor = parser.createEditor();

editor.set("ok", "hi");
editor.set("hi", "ok");

console.log(editor.toString());

editor.unset("hi");

console.log("===================");
console.log(editor.toString());

editor.unset("ok");

console.log("===================");
console.log(editor.toString());
