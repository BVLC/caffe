/**
 * express-chromeframe - middleware
 * Copyright(c) 2010 Michael Hemesath <mike.hemesath@gmail.com>
 * MIT Licensed
 */

/**
 * Return connect chromeframe middleware for the given version.
 * @param {int} version
 * @return Function
 * @api public
 */
module.exports = function(version) {
  return function(req, res, next){
    res.setHeader('X-UA-Compatible','IE=Edge,chrome=' + (version || 1));
    next();
  }
}