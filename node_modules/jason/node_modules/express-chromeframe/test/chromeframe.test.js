/**
 * Module Dependencies
 */
 
var express = require('express'),
    assert = require('assert'),
    should = require('should'),
    chromeframe = require('../');
    
function create(version) {
  var app = express.createServer();
  app.configure(function() {
    app.use(chromeframe(version));
    app.use(function(req, res, next) {
      res.end('');
    });
  });
  
  return app;
}
 
module.exports = {
  'test default configuration': function(){
    assert.response(create(), 
      { url: '/' }, 
      function(res){
        res.header('X-UA-Compatible').should.equal('IE=Edge,chrome=1');
      });
  },

  'test configured version': function(){
    assert.response(create("IE7"), 
      { url: '/' }, 
      function(res){
        res.header('X-UA-Compatible').should.equal('IE=Edge,chrome=IE7');
    });
  }
};