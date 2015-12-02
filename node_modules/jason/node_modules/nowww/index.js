/*jshint strict:true node:true es5:true onevar:true laxcomma:true laxbreak:true eqeqeq:true immed:true latedef:true undef:true unused:true*/
(function () {
  "use strict";

  function nowww(req, res, next) {
    var host = (req.headers.host||'').replace(/^www\./, '')
      , hostname = host.split(':')[0]
      , protocol = 'http' + (req.connection.encrypted ? 's' : '') + '://'
      , href = protocol + host + req.url
      ;

    if (host === req.headers.host) {
      return next();
    }

    // Permanent Redirect
    res.statusCode = 301;
    res.setHeader('Location', href);
    // TODO set token (cookie, header, something) to notify browser to notify user about www
    res.write(
        'Quit with the www already!!! It\'s not 1990 anymore!'
      + '<br/>'
      + '<a href="' + href + '">' + hostname + '</a>'
      + '<br/>NOT www.' + hostname
      + '<br/>NOT ' + protocol + hostname
      + '<br/>just <a href="' + href + '">' + hostname + '</a> !!!'
      + '<br/>'
      + ';-P'
    );
    res.end();

  }

  function create() {
    return nowww;
  }

  module.exports = create;
}());
