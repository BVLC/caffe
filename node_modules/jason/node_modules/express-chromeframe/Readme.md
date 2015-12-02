# Express ChromeFrame

  Dead simple middleware to enable chrome frame on connect/express applications.
  
      var app = express.createServer();
      
      app.configure(function() {
        app.use(chromeframe());
      })
      
      app.get('/', function(req, res){
        res.send('I want to render with chrome!');
      });
      
      app.listen(3000);
      
See http://www.chromium.org/developers/how-tos/chrome-frame-getting-started

## IMPORTANT: Before You Get Started

If the application uses connect session middleware, be sure to change the default fingerprint function to return a string that is NOT based off the user agent string.
Chrome frame is known to manipulate the user agent String across requests, which will cause users to lose their session if their session token is generated using the default fingerprint function.
    
See https://github.com/senchalabs/connect/issues/305 
      
## Installation

    $ npm install express-chromeframe
    
## Features

  * Defaults to enable chromeframe for ALL versions of Internet Explorer
  * Includes IE=Edge configuration
  * Configurable to target specific versions of Internet Explorer
    
## Examples

### Target All Versions of Internet Explorer

    var express = require('express'),
        chromeframe = require('express-chromeframe');

    var app = express.createServer();
    
    app.configure(function() {
      app.use(chromeframe());
    });
    
### Target Internet Explorer 7 and Lower
    
    var express = require('express'),
        chromeframe = require('express-chromeframe');

    var app = express.createServer();
    
    app.configure(function() {
      app.use(chromeframe("IE7"));
    });
    
### Activate Chrome Frame for Specific Routes

    var express = require('express'),
        chromeframe = require('express-chromeframe');
    
    var app = express.createServer();
    
    app.get('/foo', chromeframe(), function(req, res){
      res.send('I want to render with chrome!');
    });
    
    app.get('/bar', function(req, res){
      res.send('I want to render with the original user agent!');
    });

## License

Copyright (C) 2011 by Michael Hemesath <mike.hemesath@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.