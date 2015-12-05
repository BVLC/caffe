var temp = require('temp').track();

var p = temp.mkdirSync("shouldBeDeletedOnExitNotJasmine");
console.log('created dir ' + p);
