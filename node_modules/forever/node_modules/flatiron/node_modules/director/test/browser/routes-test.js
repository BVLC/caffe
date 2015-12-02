
createTest('Nested route with the many children as a tokens, callbacks should yield historic params', {
  '/a': {
    '/:id': {
      '/:id': function(a, b) {
        shared.fired.push(location.hash, a, b);
      }
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['#/a/b/c', 'b', 'c']);
    this.finish();
  });
});

createTest('Nested route with the first child as a token, callback should yield a param', {
  '/foo': {
    '/:id': {
      on: function(id) {
        shared.fired.push(location.hash, id);
      }
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/foo/a', function() {
    this.navigate('/foo/b/c', function() {
      deepEqual(shared.fired, ['#/foo/a', 'a']);
      this.finish();
    });
  });
});

createTest('Nested route with the first child as a regexp, callback should yield a param', {
  '/foo': {
    '/(\\w+)': {
      on: function(value) {
        shared.fired.push(location.hash, value);
      }
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/foo/a', function() {
    this.navigate('/foo/b/c', function() {
      deepEqual(shared.fired, ['#/foo/a', 'a']);
      this.finish();
    });
  });
});

createTest('Nested route with the several regular expressions, callback should yield a param', {
  '/a': {
    '/(\\w+)': {
      '/(\\w+)': function(a, b) {
        shared.fired.push(a, b);
      }
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['b', 'c']);
    this.finish();
  });
});



createTest('Single nested route with on member containing function value', {
  '/a': {
    '/b': {
      on: function() {
        shared.fired.push(location.hash);
      }
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/b', function() {
      deepEqual(shared.fired, ['#/a/b']);
      this.finish();
  });
});

createTest('Single non-nested route with on member containing function value', {
  '/a/b': {
    on: function() {
      shared.fired.push(location.hash);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/b', function() {
    deepEqual(shared.fired, ['#/a/b']);
    this.finish();
  });
});

createTest('Single nested route with on member containing array of function values', {
  '/a': {
    '/b': {
      on: [function() { shared.fired.push(location.hash); },
        function() { shared.fired.push(location.hash); }]
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/b', function() {
      deepEqual(shared.fired, ['#/a/b', '#/a/b']);
      this.finish();
  });
});

createTest('method should only fire once on the route.', {
  '/a': {
    '/b': {
      once: function() {
        shared.fired++;
      }
    }
  }
}, function() {
  shared.fired = 0;
  this.navigate('/a/b', function() {
    this.navigate('/a/b', function() {
      this.navigate('/a/b', function() {
        deepEqual(shared.fired, 1);
        this.finish();
      });
    });
  });
});

createTest('method should only fire once on the route, multiple nesting.', {
  '/a': {
    on: function() { shared.fired++; },
    once: function() { shared.fired++; }
  },
  '/b': {
    on: function() { shared.fired++; },
    once: function() { shared.fired++; }
  }
}, function() {
  shared.fired = 0;
  this.navigate('/a', function() {
    this.navigate('/b', function() {
      this.navigate('/a', function() {
        this.navigate('/b', function() {
          deepEqual(shared.fired, 6);
          this.finish();
        });
      });
    });
  });
});

createTest('overlapping routes with tokens.', {
  '/a/:b/c' : function() {
    shared.fired.push(location.hash);
  },
  '/a/:b/c/:d' : function() {
    shared.fired.push(location.hash);
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/b/c', function() {

    this.navigate('/a/b/c/d', function() {
      deepEqual(shared.fired, ['#/a/b/c', '#/a/b/c/d']);
      this.finish();
    });
  });
});

// // //
// // // Recursion features
// // // ----------------------------------------------------------

createTest('Nested routes with no recursion', {
  '/a': {
    '/b': {
      '/c': {
        on: function c() {
          shared.fired.push('c');
        }
      },
      on: function b() {
        shared.fired.push('b');
      }
    },
    on: function a() {
      shared.fired.push('a');
    }
  }
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['c']);
    this.finish();
  });
});

createTest('Nested routes with backward recursion', {
  '/a': {
    '/b': {
      '/c': {
        on: function c() {
          shared.fired.push('c');
        }
      },
      on: function b() {
        shared.fired.push('b');
      }
    },
    on: function a() {
      shared.fired.push('a');
    }
  }
}, {
  recurse: 'backward'
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['c', 'b', 'a']);
    this.finish();
  });
});

createTest('Breaking out of nested routes with backward recursion', {
  '/a': {
    '/:b': {
      '/c': {
        on: function c() {
          shared.fired.push('c');
        }
      },
      on: function b() {
        shared.fired.push('b');
        return false;
      }
    },
    on: function a() {
      shared.fired.push('a');
    }
  }
}, {
  recurse: 'backward'
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['c', 'b']);
    this.finish();
  });
});

createTest('Nested routes with forward recursion', {
  '/a': {
    '/b': {
      '/c': {
        on: function c() {
          shared.fired.push('c');
        }
      },
      on: function b() {
        shared.fired.push('b');
      }
    },
    on: function a() {
      shared.fired.push('a');
    }
  }
}, {
  recurse: 'forward'
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['a', 'b', 'c']);
    this.finish();
  });
});

createTest('Nested routes with forward recursion, single route with an after event.', {
  '/a': {
    '/b': {
      '/c': {
        on: function c() {
          shared.fired.push('c');
        },
        after: function() {
          shared.fired.push('c-after');
        }
      },
      on: function b() {
        shared.fired.push('b');
      }
    },
    on: function a() {
      shared.fired.push('a');
    }
  }
}, {
  recurse: 'forward'
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    this.navigate('/a/b', function() {
      deepEqual(shared.fired, ['a', 'b', 'c', 'c-after', 'a', 'b']);
      this.finish();
    });
  });
});

createTest('Breaking out of nested routes with forward recursion', {
  '/a': {
    '/b': {
      '/c': {
        on: function c() {
          shared.fired.push('c');
        }
      },
      on: function b() {
        shared.fired.push('b');
        return false;
      }
    },
    on: function a() {
      shared.fired.push('a');
    }
  }
}, {
  recurse: 'forward'
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    deepEqual(shared.fired, ['a', 'b']);
    this.finish();
  });
});

//
// ABOVE IS WORKING
//

// //
// // Special Events
// // ----------------------------------------------------------

createTest('All global event should fire after every route', {
  '/a': {
    on: function a() {
      shared.fired.push('a');
    }
  },
  '/b': {
    '/c': {
      on: function a() {
        shared.fired.push('a');
      }
    }
  },
  '/d': {
    '/:e': {
      on: function a() {
        shared.fired.push('a');
      }
    }
  }
}, {
  after: function() {
    shared.fired.push('b');
  }
}, function() {
  shared.fired = [];

  this.navigate('/a', function() {
    this.navigate('/b/c', function() {
      this.navigate('/d/e', function() {
        deepEqual(shared.fired, ['a', 'b', 'a', 'b', 'a']);
        this.finish();
      });
    });
  });

});

createTest('Not found.', {
  '/a': {
    on: function a() {
      shared.fired.push('a');
    }
  },
  '/b': {
    on: function a() {
      shared.fired.push('b');
    }
  }
}, {
  notfound: function() {
    shared.fired.push('notfound');
  }
}, function() {
  shared.fired = [];

  this.navigate('/c', function() {
    this.navigate('/d', function() {
      deepEqual(shared.fired, ['notfound', 'notfound']);
      this.finish();
    });
  });
});

createTest('On all.', {
  '/a': {
    on: function a() {
      shared.fired.push('a');
    }
  },
  '/b': {
    on: function a() {
      shared.fired.push('b');
    }
  }
}, {
  on: function() {
    shared.fired.push('c');
  }
}, function() {
  shared.fired = [];

  this.navigate('/a', function() {
    this.navigate('/b', function() {
      deepEqual(shared.fired, ['a', 'c', 'b', 'c']);
      this.finish();
    });
  });
});


createTest('After all.', {
  '/a': {
    on: function a() {
      shared.fired.push('a');
    }
  },
  '/b': {
    on: function a() {
      shared.fired.push('b');
    }
  }
}, {
  after: function() {
    shared.fired.push('c');
  }
}, function() {
  shared.fired = [];

  this.navigate('/a', function() {
    this.navigate('/b', function() {
      deepEqual(shared.fired, ['a', 'c', 'b']);
      this.finish();
    });
  });
});

createTest('resource object.', {
  '/a': {
    '/b/:c': {
      on: 'f1'
    },
    on: 'f2'
  },
  '/d': {
    on: ['f1', 'f2']
  }
},
{
  resource: {
    f1: function (name){
        shared.fired.push("f1-" + name);
    },
    f2: function (name){
        shared.fired.push("f2");
    }
  }
}, function() {
  shared.fired = [];

  this.navigate('/a/b/c', function() {
    this.navigate('/d', function() {
      deepEqual(shared.fired, ['f1-c', 'f1-undefined', 'f2']);
      this.finish();
    });
  });
});

createTest('argument matching should be case agnostic', {
  '/fooBar/:name': {
      on: function(name) {
        shared.fired.push("fooBar-" + name);
      }
    }
}, function() {
  shared.fired = [];
  this.navigate('/fooBar/tesTing', function() {
    deepEqual(shared.fired, ['fooBar-tesTing']);
    this.finish();
  });
});

createTest('sanity test', {
  '/is/:this/:sane': {
    on: function(a, b) {
      shared.fired.push('yes ' + a + ' is ' + b);
    }
  },
  '/': {
    on: function() {
      shared.fired.push('is there sanity?');
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/is/there/sanity', function() {
    deepEqual(shared.fired, ['yes there is sanity']);
    this.finish();
  });
});

createTest('`/` route should be navigable from the routing table', {
  '/': {
    on: function root() {
      shared.fired.push('/');
    }
  },
  '/:username': {
    on: function afunc(username) {
      shared.fired.push('/' + username);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/', function root() {
    deepEqual(shared.fired, ['/']);
    this.finish();
  });
});

createTest('`/` route should not override a `/:token` route', {
  '/': {
    on: function root() {
      shared.fired.push('/');
    }
  },
  '/:username': {
    on: function afunc(username) {
      shared.fired.push('/' + username);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a', function afunc() {
    deepEqual(shared.fired, ['/a']);
    this.finish();
  });
});

createTest('should accept the root as a token.', {
  '/:a': {
    on: function root() {
      shared.fired.push(location.hash);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a', function root() {
    deepEqual(shared.fired, ['#/a']);
    this.finish();
  });
});

createTest('routes should allow wildcards.', {
  '/:a/b*d': {
    on: function() {
      shared.fired.push(location.hash);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a/bcd', function root() {
    deepEqual(shared.fired, ['#/a/bcd']);
    this.finish();
  });
});

createTest('functions should have |this| context of the router instance.', {
  '/': {
    on: function root() {
      shared.fired.push(!!this.routes);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/', function root() {
    deepEqual(shared.fired, [true]);
    this.finish();
  });
});

createTest('setRoute with a single parameter should change location correctly', {
  '/bonk': {
    on: function() {
      shared.fired.push(window.location.hash);
    }
  }
}, function() {
  var self = this;
  shared.fired = [];
  this.router.setRoute('/bonk');
  setTimeout(function() {
    deepEqual(shared.fired, ['#/bonk']);
    self.finish();
  }, 14)
});

createTest('route should accept _ and . within parameters', {
  '/:a': {
    on: function root() {
      shared.fired.push(location.hash);
    }
  }
}, function() {
  shared.fired = [];
  this.navigate('/a_complex_route.co.uk', function root() {
    deepEqual(shared.fired, ['#/a_complex_route.co.uk']);
    this.finish();
  });
});

createTest('initializing with a default route should only result in one route handling', {
    '/': {
      on: function root() {
        if (!shared.init){
            shared.init = 0;
        }
        shared.init++;
      }
    },
    '/test': {
      on: function root() {
        if (!shared.test){
            shared.test = 0;
        }
        shared.test++;
      }
    }
  }, function() {
    this.navigate('/test', function root() {
      equal(shared.init, 1);
      equal(shared.test, 1);
      this.finish();
    });
  },
  null,
  '/');

createTest('changing the hash twice should call each route once', {
    '/hash1': {
      on: function root() {
          shared.fired.push('hash1');
      }
    },
    '/hash2': {
        on: function root() {
            shared.fired.push('hash2');
        }
      }
  }, function() {
    shared.fired = [];
    this.navigate('/hash1', function(){});
    this.navigate('/hash2', function(){
      deepEqual(shared.fired, ['hash1', 'hash2']);
      this.finish();
    });
  }
);

// This test doesn't use the createTest since createTest runs init on the router before
// running the test, which is what we want to test.
test('fire the correct route when initializing the router', function(){
  window.location.hash = 'initial';
  var fired = [];
  var router = new (window.Router || window.RouterAlias)({
    '/initial': function(){
      fired.push('/initial');
    },
    'initial': function(){
      fired.push('initial');
    }
  });
  router.init();
  deepEqual(fired, ['/initial', 'initial']);
  router.destroy();
});

test('do not combine hash if convert_hash_in_init is false', function(){
  window.location.hash = 'initial';
  var fired = [];
  var initialPath = window.location.pathname;
  var routes = {
    '/initial': function(){
      fired.push('/initial');
    },
    'initial': function(){
      fired.push('initial');
    }
  };
  routes[initialPath] = function(){
    fired.push('*');
  };

  var router = new (window.Router || window.RouterAlias)(routes);

  router.configure({
    html5history: true,
    convert_hash_in_init: false
  });
  router.init();
  deepEqual(fired, ['*']);
  router.destroy();
});
