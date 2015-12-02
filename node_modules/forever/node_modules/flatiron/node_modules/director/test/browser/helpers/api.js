module("Director.js", {
  setup: function() {
    window.location.hash = "";
    shared = {};
    // Init needed keys earlier because of in HTML5 mode the route handler
    // is executed upon Router.init() and due to that setting shared.fired
    // in the param test of createTest is too late
    if (HTML5TEST) {
      shared.fired       = [];
      shared.fired_count = 0;
    }
  },
  teardown: function() {
    window.location.hash = "";
    shared = {};
  }
});

var shared;

function createTest(name, config, use, test, initialRoute) {
  // We rename to `RouterAlias` for the browserify tests, since we want to be
  // sure that no code is depending on `window.Router` being available.
  var Router = window.Router || window.RouterAlias;

  if (typeof use === 'function') {
    test = use;
    use = undefined;
  }

  if (HTML5TEST) {
    if (use === undefined) {
      use = {};
    }

    if (use.run_handler_in_init === undefined) {
      use.run_handler_in_init = false;
    }
    use.html5history        = true;
  }

  // Because of the use of setTimeout when defining onpopstate
  var innerTimeout = HTML5TEST === true ? 500 : 0;

  asyncTest(name, function() {
    setTimeout(function() {
      var router = new Router(config),
          context;

      if (use !== undefined) {
        router.configure(use);
      }

      router.init(initialRoute);

      setTimeout(function() {
        test.call(context = {
          router: router,
          navigate: function(url, callback) {
            if (HTML5TEST) {
              router.setRoute(url);
            } else {
              window.location.hash = url;
            }
            setTimeout(function() {
              callback.call(context);
            }, 14);
          },
          finish: function() {
            router.destroy();
            start();
          }
        })
      }, innerTimeout);
    }, 14);
  });
};
