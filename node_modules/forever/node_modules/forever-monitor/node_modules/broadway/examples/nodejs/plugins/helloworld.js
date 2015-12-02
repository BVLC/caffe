
var HelloWorld = exports;

//
// `exports.attach` gets called by broadway on `app.use`
//
HelloWorld.attach = function (options) {

  this.hello = function (world) {
    console.log("Hello "+ world + options.delimiter || ".");
  }
};

//
// `exports.init` gets called by broadway on `app.init`.
//
HelloWorld.init = function (done) {

  //
  // This plugin doesn't require any initialization step.
  //
  return done();
};
