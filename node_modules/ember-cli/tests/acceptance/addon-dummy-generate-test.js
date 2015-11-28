/*jshint quotmark: false*/

'use strict';

var Promise              = require('../../lib/ext/promise');
var assertFile           = require('../helpers/assert-file');
var assertFileEquals     = require('../helpers/assert-file-equals');
var assertFileToNotExist = require('../helpers/assert-file-to-not-exist');
var conf                 = require('../helpers/conf');
var ember                = require('../helpers/ember');
var fs                   = require('fs-extra');
var path                 = require('path');
var remove               = Promise.denodeify(fs.remove);
var root                 = process.cwd();
var tmp                  = require('tmp-sync');
var tmproot              = path.join(root, 'tmp');
var EOL                  = require('os').EOL;
var BlueprintNpmTask     = require('../helpers/disable-npm-on-blueprint');
var expect               = require('chai').expect;

describe('Acceptance: ember generate in-addon-dummy', function() {
  this.timeout(20000);
  var tmpdir;

  before(function() {
    BlueprintNpmTask.disableNPM();
    conf.setup();
  });

  after(function() {
    BlueprintNpmTask.restoreNPM();
    conf.restore();
  });

  beforeEach(function() {
    tmpdir = tmp.in(tmproot);
    process.chdir(tmpdir);
  });

  afterEach(function() {
    process.chdir(root);
    return remove(tmproot);
  });

  function initAddon() {
    return ember([
      'addon',
      'my-addon',
      '--skip-npm',
      '--skip-bower'
    ]);
  }

  function generateInAddon(args) {
    var generateArgs = ['generate'].concat(args);

    return initAddon().then(function() {
      return ember(generateArgs);
    });
  }

  it('dummy controller foo', function() {
    return generateInAddon(['controller', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/controllers/foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Controller.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/controllers/foo-test.js');
      assertFileToNotExist('tests/unit/controllers/foo-test.js');
    });
  });

  it('dummy controller foo/bar', function() {
    return generateInAddon(['controller', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/controllers/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Controller.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/controllers/foo/bar.js');
      assertFileToNotExist('tests/unit/controllers/foo/bar-test.js');
    });
  });

  it('dummy component x-foo', function() {
    return generateInAddon(['component', 'x-foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/components/x-foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Component.extend({",
          "});"
        ]
      });
      assertFile('tests/dummy/app/templates/components/x-foo.hbs', {
        contains: "{{yield}}"
      });
      assertFileToNotExist('app/components/x-foo.js');
      assertFileToNotExist('tests/unit/components/x-foo-test.js');
    });
  });

  it('dummy component-test x-foo', function() {
    return generateInAddon(['component-test', 'x-foo', '--dummy']).then(function() {
      assertFile('tests/integration/components/x-foo-test.js', {
        contains: [
          "import { moduleForComponent, test } from 'ember-qunit';",
          "import hbs from 'htmlbars-inline-precompile';",
          "moduleForComponent('x-foo'"
        ]
      });
      assertFileToNotExist('app/component-test/x-foo.js');
    });
  });

  it('dummy component nested/x-foo', function() {
    return generateInAddon(['component', 'nested/x-foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/components/nested/x-foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Component.extend({",
          "});"
        ]
      });
      assertFile('tests/dummy/app/templates/components/nested/x-foo.hbs', {
        contains: "{{yield}}"
      });
      assertFileToNotExist('app/components/nested/x-foo.js');
      assertFileToNotExist('tests/unit/components/nested/x-foo-test.js');
    });
  });

  it('dummy helper foo-bar', function() {
    return generateInAddon(['helper', 'foo-bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/helpers/foo-bar.js', {
        contains: "import Ember from 'ember';" + EOL + EOL +
                  "export function fooBar(params/*, hash*/) {" + EOL +
                  "  return params;" + EOL +
                  "}" +  EOL + EOL +
                  "export default Ember.Helper.helper(fooBar);"
      });
      assertFileToNotExist('app/helpers/foo-bar.js');
      assertFileToNotExist('tests/unit/helpers/foo-bar-test.js');
    });
  });

  it('dummy helper foo/bar-baz', function() {
    return generateInAddon(['helper', 'foo/bar-baz', '--dummy']).then(function() {
      assertFile('tests/dummy/app/helpers/foo/bar-baz.js', {
        contains: "import Ember from 'ember';" + EOL + EOL +
                  "export function fooBarBaz(params/*, hash*/) {" + EOL +
                  "  return params;" + EOL +
                  "}" + EOL + EOL +
                  "export default Ember.Helper.helper(fooBarBaz);"
      });
      assertFileToNotExist('app/helpers/foo/bar-baz.js');
      assertFileToNotExist('tests/unit/helpers/foo/bar-baz-test.js');
    });
  });

  it('dummy model foo', function() {
    return generateInAddon(['model', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/models/foo.js', {
        contains: [
          "import DS from 'ember-data';",
          "export default DS.Model.extend"
        ]
      });
      assertFileToNotExist('app/models/foo.js');
      assertFileToNotExist('tests/unit/models/foo-test.js');
    });
  });

  it('dummy model foo with attributes', function() {
    return generateInAddon([
      'model',
      'foo',
      'noType',
      'firstName:string',
      'created_at:date',
      'is-published:boolean',
      'rating:number',
      'bars:has-many',
      'baz:belongs-to',
      'echo:hasMany',
      'bravo:belongs_to',
      'foo-names:has-many',
      'barName:has-many',
      'bazName:belongs-to',
      'test-name:belongs-to',
      'echoName:hasMany',
      'bravoName:belongs_to',
      '--dummy'
    ]).then(function() {
      assertFile('tests/dummy/app/models/foo.js', {
        contains: [
          "noType: DS.attr()",
          "firstName: DS.attr('string')",
          "createdAt: DS.attr('date')",
          "isPublished: DS.attr('boolean')",
          "rating: DS.attr('number')",
          "bars: DS.hasMany('bar')",
          "baz: DS.belongsTo('baz')",
          "echos: DS.hasMany('echo')",
          "bravo: DS.belongsTo('bravo')",
          "fooNames: DS.hasMany('foo-name')",
          "barNames: DS.hasMany('bar-name')",
          "bazName: DS.belongsTo('baz-name')",
          "testName: DS.belongsTo('test-name')",
          "echoNames: DS.hasMany('echo-name')",
          "bravoName: DS.belongsTo('bravo-name')"
        ]
      });
      assertFileToNotExist('app/models/foo.js');
      assertFileToNotExist('tests/unit/models/foo-test.js');
    });
  });

  it('dummy model foo/bar', function() {
    return generateInAddon(['model', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/models/foo/bar.js', {
        contains: [
          "import DS from 'ember-data';",
          "export default DS.Model.extend"
        ]
      });
      assertFileToNotExist('app/models/foo/bar.js');
      assertFileToNotExist('tests/unit/models/foo/bar-test.js');
    });
  });

  it('dummy model-test foo', function() {
    return generateInAddon(['model-test', 'foo', '--dummy']).then(function() {
      assertFile('tests/unit/models/foo-test.js', {
        contains: [
          "import { moduleForModel, test } from 'ember-qunit';",
          "moduleForModel('foo'"
        ]
      });
      assertFileToNotExist('app/model-test/foo.js');
    });
  });

  it('dummy route foo', function() {
    return generateInAddon(['route', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/routes/foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Route.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/routes/foo.js');
      assertFile('tests/dummy/app/templates/foo.hbs', {
        contains: '{{outlet}}'
      });
      assertFile('tests/dummy/app/router.js', {
        contains: "this.route('foo');"
      });
      assertFileToNotExist('app/templates/foo.js');
      assertFileToNotExist('tests/unit/routes/foo-test.js');
    });
  });

  it('dummy route foo/bar', function() {
    return generateInAddon(['route', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/routes/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Route.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/routes/foo/bar.js');
      assertFile('tests/dummy/app/templates/foo/bar.hbs', {
        contains: '{{outlet}}'
      });
      assertFile('tests/dummy/app/router.js', {
        contains: [
          "this.route('foo', function() {",
          "this.route('bar');",
        ]
      });
      assertFileToNotExist('tests/unit/routes/foo/bar-test.js');
    });
  });

  it('dummy route-test foo', function() {
    return generateInAddon(['route-test', 'foo']).then(function() {
      assertFile('tests/unit/routes/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('route:foo'"
        ]
      });
      assertFileToNotExist('app/route-test/foo.js');
    });
  });

  it('dummy template foo', function() {
    return generateInAddon(['template', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/templates/foo.hbs');
    });
  });

  it('dummy template foo/bar', function() {
    return generateInAddon(['template', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/templates/foo/bar.hbs');
    });
  });

  it('dummy view foo', function() {
    return generateInAddon(['view', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/views/foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.View.extend({" + EOL + "})"
        ]
      });
      assertFileToNotExist('app/views/foo.js');
      assertFileToNotExist('tests/unit/views/foo-test.js');
    });
  });

  it('dummy view foo/bar', function() {
    return generateInAddon(['view', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/views/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.View.extend({" + EOL + "})"
        ]
      });
      assertFileToNotExist('app/views/foo/bar.js');
      assertFileToNotExist('tests/unit/views/foo/bar-test.js');
    });
  });

  it('dummy resource foos', function() {
    return generateInAddon(['resource', 'foos', '--dummy']).catch(function(error) {
      expect(error.message).to.include('blueprint does not support ' +
        'generating inside addons.');
    });
  });

  it('dummy initializer foo', function() {
    return generateInAddon(['initializer', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/initializers/foo.js', {
        contains: "export function initialize(/* application */) {" + EOL +
                  "  // application.inject('route', 'foo', 'service:foo');" + EOL +
                  "}" + EOL +
                  "" + EOL+
                  "export default {" + EOL +
                  "  name: 'foo'," + EOL +
                  "  initialize" + EOL +
                  "};"
      });
      assertFileToNotExist('app/initializers/foo.js');
      assertFileToNotExist('tests/unit/initializers/foo-test.js');
    });
  });

  it('dummy initializer foo/bar', function() {
    return generateInAddon(['initializer', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/initializers/foo/bar.js', {
        contains: "export function initialize(/* application */) {" + EOL +
                  "  // application.inject('route', 'foo', 'service:foo');" + EOL +
                  "}" + EOL +
                  "" + EOL+
                  "export default {" + EOL +
                  "  name: 'foo/bar'," + EOL +
                  "  initialize" + EOL +
                  "};"
      });
      assertFileToNotExist('app/initializers/foo/bar.js');
      assertFileToNotExist('tests/unit/initializers/foo/bar-test.js');
    });
  });

  it('dummy mixin foo', function() {
    return generateInAddon(['mixin', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/mixins/foo.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Mixin.create({' + EOL + '});'
        ]
      });
      assertFileToNotExist('tests/unit/mixins/foo-test.js');
      assertFileToNotExist('app/mixins/foo.js');
    });
  });

  it('dummy mixin foo/bar', function() {
    return generateInAddon(['mixin', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/mixins/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Mixin.create({' + EOL + '});'
        ]
      });
      assertFileToNotExist('tests/unit/mixins/foo/bar-test.js');
      assertFileToNotExist('app/mixins/foo/bar.js');
    });
  });

  it('dummy mixin foo/bar/baz', function() {
    return generateInAddon(['mixin', 'foo/bar/baz', '--dummy']).then(function() {
      assertFile('tests/dummy/app/mixins/foo/bar/baz.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Mixin.create({' + EOL + '});'
        ]
      });
      assertFileToNotExist('tests/unit/mixins/foo/bar/baz-test.js');
      assertFileToNotExist('app/mixins/foo/bar/baz.js');
    });
  });

  it('dummy adapter application', function() {
    return generateInAddon(['adapter', 'application', '--dummy']).then(function() {
      assertFile('tests/dummy/app/adapters/application.js', {
        contains: [
          "import DS from \'ember-data\';",
          "export default DS.RESTAdapter.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/adapters/application.js');
      assertFileToNotExist('tests/unit/adapters/application-test.js');
    });
  });

  it('dummy adapter foo', function() {
    return generateInAddon(['adapter', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/adapters/foo.js', {
        contains: [
          "import DS from \'ember-data\';",
          "export default DS.RESTAdapter.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/adapters/foo.js');
      assertFileToNotExist('tests/unit/adapters/foo-test.js');
    });
  });

  it('dummy adapter foo/bar (with base class foo)', function() {
    return generateInAddon(['adapter', 'foo/bar', '--base-class=foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/adapters/foo/bar.js', {
        contains: [
          "import FooAdapter from \'../foo\';",
          "export default FooAdapter.extend({" + EOL + "});"
        ]
      });
      assertFileToNotExist('app/adapters/foo/bar.js');
      assertFileToNotExist('tests/unit/adapters/foo/bar-test.js');
    });
  });

  it('dummy adapter-test foo', function() {
    return generateInAddon(['adapter-test', 'foo', '--dummy']).then(function() {
      assertFile('tests/unit/adapters/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('adapter:foo'"
        ]
      });
      assertFileToNotExist('app/adapter-test/foo.js');
    });
  });

  it('dummy serializer foo', function() {
    return generateInAddon(['serializer', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/serializers/foo.js', {
        contains: [
          "import DS from 'ember-data';",
          'export default DS.RESTSerializer.extend({' + EOL + '});'
        ]
      });
      assertFileToNotExist('app/serializers/foo.js');
      assertFileToNotExist('tests/unit/serializers/foo-test.js');
    });
  });

  it('dummy serializer foo/bar', function() {
    return generateInAddon(['serializer', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/serializers/foo/bar.js', {
        contains: [
          "import DS from 'ember-data';",
          'export default DS.RESTSerializer.extend({' + EOL + '});'
        ]
      });
      assertFileToNotExist('app/serializers/foo/bar.js');
      assertFileToNotExist('tests/unit/serializers/foo/bar-test.js');
    });
  });

  it('dummy serializer-test foo', function() {
    return generateInAddon(['serializer-test', 'foo', '--dummy']).then(function() {
      assertFile('tests/unit/serializers/foo-test.js', {
        contains: [
          "import { moduleForModel, test } from 'ember-qunit';",
          "moduleForModel('foo'"
        ]
      });
      assertFileToNotExist('app/serializer-test/foo.js');
    });
  });

  it('dummy transform foo', function() {
    return generateInAddon(['transform', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/transforms/foo.js', {
        contains: [
          "import DS from 'ember-data';",
          'export default DS.Transform.extend({' + EOL +
          '  deserialize(serialized) {' + EOL +
          '    return serialized;' + EOL +
          '  },' + EOL +
          EOL +
          '  serialize(deserialized) {' + EOL +
          '    return deserialized;' + EOL +
          '  }' + EOL +
          '});'
        ]
      });
      assertFileToNotExist('app/transforms/foo.js');
      assertFileToNotExist('tests/unit/transforms/foo-test.js');
    });
  });

  it('dummy transform foo/bar', function() {
    return generateInAddon(['transform', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/transforms/foo/bar.js', {
        contains: [
          "import DS from 'ember-data';",
          'export default DS.Transform.extend({' + EOL +
          '  deserialize(serialized) {' + EOL +
          '    return serialized;' + EOL +
          '  },' + EOL +
          '' + EOL +
          '  serialize(deserialized) {' + EOL +
          '    return deserialized;' + EOL +
          '  }' + EOL +
          '});'
        ]
      });
      assertFileToNotExist('app/transforms/foo/bar.js');
      assertFileToNotExist('tests/unit/transforms/foo/bar-test.js');
    });
  });

  it('dummy util foo-bar', function() {
    return generateInAddon(['util', 'foo-bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/utils/foo-bar.js', {
        contains: 'export default function fooBar() {' + EOL +
                  '  return true;' + EOL +
                  '}'
      });
      assertFileToNotExist('app/utils/foo-bar.js');
      assertFileToNotExist('tests/unit/utils/foo-bar-test.js');
    });
  });

  it('dummy util foo-bar/baz', function() {
    return generateInAddon(['util', 'foo/bar-baz', '--dummy']).then(function() {
      assertFile('tests/dummy/app/utils/foo/bar-baz.js', {
        contains: 'export default function fooBarBaz() {' + EOL +
                  '  return true;' + EOL +
                  '}'
      });
      assertFileToNotExist('app/utils/foo/bar-baz.js');
      assertFileToNotExist('tests/unit/utils/foo/bar-baz-test.js');
    });
  });

  it('dummy service foo', function() {
    return generateInAddon(['service', 'foo', '--dummy']).then(function() {
      assertFile('tests/dummy/app/services/foo.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Service.extend({' + EOL + '});'
        ]
      });
      assertFileToNotExist('app/services/foo.js');
      assertFileToNotExist('tests/unit/services/foo-test.js');
    });
  });

  it('dummy service foo/bar', function() {
    return generateInAddon(['service', 'foo/bar', '--dummy']).then(function() {
      assertFile('tests/dummy/app/services/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Service.extend({' + EOL + '});'
        ]
      });
      assertFileToNotExist('app/services/foo/bar.js');
      assertFileToNotExist('tests/unit/services/foo/bar-test.js');
    });
  });


  it('dummy service-test foo', function() {
    return generateInAddon(['service-test', 'foo', '--dummy']).then(function() {
      assertFile('tests/unit/services/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('service:foo'"
        ]
      });
      assertFileToNotExist('app/service-test/foo.js');
    });
  });

  it('dummy blueprint foo', function() {
    return generateInAddon(['blueprint', 'foo', '--dummy']).then(function() {
      assertFile('blueprints/foo/index.js', {
        contains: "module.exports = {" + EOL +
                  "  description: ''"+ EOL +
                  EOL +
                  "  // locals: function(options) {" + EOL +
                  "  //   // Return custom template variables here." + EOL +
                  "  //   return {" + EOL +
                  "  //     foo: options.entity.options.foo" + EOL +
                  "  //   };" + EOL +
                  "  // }" + EOL +
                  EOL +
                  "  // afterInstall: function(options) {" + EOL +
                  "  //   // Perform extra work here." + EOL +
                  "  // }" + EOL +
                  "};"
        });
      });
    });

    it('dummy blueprint foo/bar', function() {
      return generateInAddon(['blueprint', 'foo/bar', '--dummy']).then(function() {
        assertFile('blueprints/foo/bar/index.js', {
          contains: "module.exports = {" + EOL +
                    "  description: ''"+ EOL +
                    EOL +
                    "  // locals: function(options) {" + EOL +
                    "  //   // Return custom template variables here." + EOL +
                    "  //   return {" + EOL +
                    "  //     foo: options.entity.options.foo" + EOL +
                    "  //   };" + EOL +
                    "  // }" + EOL +
                    EOL +
                    "  // afterInstall: function(options) {" + EOL +
                    "  //   // Perform extra work here." + EOL +
                    "  // }" + EOL +
                    "};"
        });
      });
    });

    it('dummy http-mock foo', function() {
      return generateInAddon(['http-mock', 'foo', '--dummy']).then(function() {
        assertFile('server/index.js', {
          contains:"mocks.forEach(function(route) { route(app); });"
        });
        assertFile('server/mocks/foo.js', {
          contains: "module.exports = function(app) {" + EOL +
                    "  var express = require('express');" + EOL +
                    "  var fooRouter = express.Router();" + EOL +
                    EOL +
                    "  fooRouter.get('/', function(req, res) {" + EOL +
                    "    res.send({" + EOL +
                    "      'foo': []" + EOL +
                    "    });" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooRouter.post('/', function(req, res) {" + EOL +
                    "    res.status(201).end();" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooRouter.get('/:id', function(req, res) {" + EOL +
                    "    res.send({" + EOL +
                    "      'foo': {" + EOL +
                    "        id: req.params.id" + EOL +
                    "      }" + EOL +
                    "    });" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooRouter.put('/:id', function(req, res) {" + EOL +
                    "    res.send({" + EOL +
                    "      'foo': {" + EOL +
                    "        id: req.params.id" + EOL +
                    "      }" + EOL +
                    "    });" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooRouter.delete('/:id', function(req, res) {" + EOL +
                    "    res.status(204).end();" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  // The POST and PUT call will not contain a request body" + EOL +
                    "  // because the body-parser is not included by default." + EOL +
                    "  // To use req.body, run:" + EOL +
                    EOL +
                    "  //    npm install --save-dev body-parser" + EOL +
                    EOL +
                    "  // After installing, you need to `use` the body-parser for" + EOL +
                    "  // this mock uncommenting the following line:" + EOL +
                    "  //" + EOL +
                    "  //app.use('/api/foo', require('body-parser'));" + EOL +
                    "  app.use('/api/foo', fooRouter);" + EOL +
                    "};" + EOL
        });
        assertFile('server/.jshintrc', {
          contains: '{' + EOL + '  "node": true' + EOL + '}'
        });
      });
    });

    it('dummy http-mock foo-bar', function() {
      return generateInAddon(['http-mock', 'foo-bar', '--dummy']).then(function() {
        assertFile('server/index.js', {
          contains: "mocks.forEach(function(route) { route(app); });"
        });
        assertFile('server/mocks/foo-bar.js', {
          contains: "module.exports = function(app) {" + EOL +
                    "  var express = require('express');" + EOL +
                    "  var fooBarRouter = express.Router();" + EOL +
                    EOL +
                    "  fooBarRouter.get('/', function(req, res) {" + EOL +
                    "    res.send({" + EOL +
                    "      'foo-bar': []" + EOL +
                    "    });" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooBarRouter.post('/', function(req, res) {" + EOL +
                    "    res.status(201).end();" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooBarRouter.get('/:id', function(req, res) {" + EOL +
                    "    res.send({" + EOL +
                    "      'foo-bar': {" + EOL +
                    "        id: req.params.id" + EOL +
                    "      }" + EOL +
                    "    });" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooBarRouter.put('/:id', function(req, res) {" + EOL +
                    "    res.send({" + EOL +
                    "      'foo-bar': {" + EOL +
                    "        id: req.params.id" + EOL +
                    "      }" + EOL +
                    "    });" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  fooBarRouter.delete('/:id', function(req, res) {" + EOL +
                    "    res.status(204).end();" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  // The POST and PUT call will not contain a request body" + EOL +
                    "  // because the body-parser is not included by default." + EOL +
                    "  // To use req.body, run:" + EOL +
                    EOL +
                    "  //    npm install --save-dev body-parser" + EOL +
                    EOL +
                    "  // After installing, you need to `use` the body-parser for" + EOL +
                    "  // this mock uncommenting the following line:" + EOL +
                    "  //" + EOL +
                    "  //app.use('/api/foo-bar', require('body-parser'));" + EOL +
                    "  app.use('/api/foo-bar', fooBarRouter);" + EOL +
                    "};" + EOL
        });
        assertFile('server/.jshintrc', {
          contains: '{' + EOL + '  "node": true' + EOL + '}'
        });
      });
    });

    it('dummy http-proxy foo', function() {
      return generateInAddon(['http-proxy', 'foo', 'http://localhost:5000', '--dummy']).then(function() {
        assertFile('server/index.js', {
          contains: "proxies.forEach(function(route) { route(app); });"
        });
        assertFile('server/proxies/foo.js', {
          contains: "var proxyPath = '/foo';" + EOL +
                    EOL +
                    "module.exports = function(app) {" + EOL +
                    "  // For options, see:" + EOL +
                    "  // https://github.com/nodejitsu/node-http-proxy" + EOL +
                    "  var proxy = require('http-proxy').createProxyServer({});" + EOL +
                    EOL +
                    "  proxy.on('error', function(err, req) {" + EOL +
                    "    console.error(err, req.url);" + EOL +
                    "  });" + EOL +
                    EOL +
                    "  app.use(proxyPath, function(req, res, next){" + EOL +
                    "    // include root path in proxied request" + EOL +
                    "    req.url = proxyPath + '/' + req.url;" + EOL +
                    "    proxy.web(req, res, { target: 'http://localhost:5000' });" + EOL +
                    "  });" + EOL +
                    "};"
        });
        assertFile('server/.jshintrc', {
          contains: '{' + EOL + '  "node": true' + EOL + '}'
        });
      });
    });

    it('dummy server', function() {
      return generateInAddon(['server', '--dummy']).then(function() {
        assertFile('server/index.js');
        assertFile('server/.jshintrc');
      });
    });

    it('dummy acceptance-test foo', function() {
      return generateInAddon(['acceptance-test', 'foo', '--dummy']).then(function() {
        var expected = path.join(__dirname, '../fixtures/generate/addon-acceptance-test-expected.js');

        assertFileEquals('tests/acceptance/foo-test.js', expected);
        assertFileToNotExist('app/acceptance-tests/foo.js');
      });
    });

    it('dummy acceptance-test foo/bar', function() {
      return generateInAddon(['acceptance-test', 'foo/bar', '--dummy']).then(function() {
        var expected = path.join(__dirname, '../fixtures/generate/addon-acceptance-test-nested-expected.js');

        assertFileEquals('tests/acceptance/foo/bar-test.js', expected);
        assertFileToNotExist('app/acceptance-tests/foo/bar.js');
      });
    });

});
