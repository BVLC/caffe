var ometajs = require('../../ometajs');

// Include various utils
ometajs.utils.extend(exports, require('./utils'));

// Include grammars
ometajs.utils.extend(exports, require('./core'));
ometajs.utils.extend(exports, require('./parsers'));
