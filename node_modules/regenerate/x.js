var regenerate = require('./');
var jsesc = require('jsesc');

console.log(jsesc(

regenerate()
  .addRange(0x000000, 0x10FFFF) // add all Unicode code points
  .removeRange(0x0041, 0x007A) // remove all code points from U+0041 to U+007A
  .toString()

));

// '\uD834\uDF06'

// [\uD800-\uDFFF](?![\uDC00-\uDFFF])|(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF]
