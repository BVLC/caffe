// Copyright 2014, 2015 Simon Lydell
// X11 (“MIT”) Licensed. (See LICENSE.)

var fs     = require("fs")
var assert = require("assert")

var lineNumbers = require("../")

var sumJS = [
  "/**",
  " * Sums two numbers.",
  " *",
  " * @param a Number",
  " * @param b Number",
  " * @returns Number",
  " */",
  "",
  "function sum(a, b) {",
  "  return a + b",
  "}"
]

var defaultOutput = [
  "  1 | /**",
  "  2 |  * Sums two numbers.",
  "  3 |  *",
  "  4 |  * @param a Number",
  "  5 |  * @param b Number",
  "  6 |  * @returns Number",
  "  7 |  */",
  "  8 | ",
  "  9 | function sum(a, b) {",
  " 10 |   return a + b",
  " 11 | }"
]

suite("lineNumbers", function() {

  test("is a function", function() {
    assert.equal(typeof lineNumbers, "function")
  })


  test("defaults", function() {
    assert.equal(lineNumbers(sumJS.join("\n")), defaultOutput.join("\n"))
  })


  test("array of lines", function() {
    assert.deepEqual(lineNumbers(sumJS), defaultOutput)
  })


  test("options", function() {
    assert.equal(lineNumbers(sumJS.join("\n"), {
      start: 5,
      padding: "0",
      before: "  ",
      after: ": ",
      transform: function(params) {
        if (params.number === 13) {
          params.line = params.line + "\n" + params.before +
                        Array(params.width + 1).join(" ") + params.after +
                        Array(params.line.indexOf("(") + 1).join(" ") + "^"
          params.before = params.before.replace(/^./, ">")
        }
      }
    }), [
      "  05: /**",
      "  06:  * Sums two numbers.",
      "  07:  *",
      "  08:  * @param a Number",
      "  09:  * @param b Number",
      "  10:  * @returns Number",
      "  11:  */",
      "  12: ",
      "> 13: function sum(a, b) {",
      "    :             ^",
      "  14:   return a + b",
      "  15: }"
    ].join("\n"))
  })


  test("Windows-style newlines", function() {
    assert.equal(lineNumbers("a\r\nb"), " 1 | a\r\n 2 | b")
  })


  test("trailing newline", function() {
    assert.equal(lineNumbers("single line\n"), " 1 | single line\n 2 | ")
  })


  test("no newline", function() {
    assert.equal(lineNumbers("single line"), " 1 | single line")
  })


  test("one more digit", function() {
    assert.equal(lineNumbers("a", {start: 9}), " 9 | a")
    assert.equal(lineNumbers("a\nb", {start: 9}), "  9 | a\n 10 | b")
  })

})
