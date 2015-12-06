# Sorted Array #

An implementation of John von Neumann's sorted arrays in JavaScript. Implements insertion sort and binary search for fast insertion and deletion.

## Installation ##

Sorted arrays may be installed on [node.js](http://nodejs.org/ "node.js") via the [node package manager](https://npmjs.org/ "npm") using the command `npm install sorted-array`.

You may also install it on [RingoJS](http://ringojs.org/ "Home - RingoJS") using the command `ringo-admin install javascript/sorted-array`.

You may install it as a [component](https://github.com/component/component "component/component") for web apps using the command `component install javascript/sorted-array`.

## Usage ##

The six line tutorial on sorted arrays:

```javascript
var SortedArray = require("sorted-array");
var sorted = new SortedArray(3, 1, 5, 2, 4);
console.dir(sorted.array);                   // [1, 2, 3, 4, 5]
sorted.search(3);                            // 2
sorted.remove(3);                            // [1, 2, 4, 5]
sorted.insert(3);                            // [1, 2, 3, 4, 5]
```
