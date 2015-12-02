'use strict'

var fs = require('fs')
var statusCodes = require('./')

var code = 'module.exports = ' + JSON.stringify(statusCodes, null, 2)

fs.writeFileSync('browser.js', code)
