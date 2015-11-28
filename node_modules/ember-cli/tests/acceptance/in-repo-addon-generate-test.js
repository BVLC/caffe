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

describe('Acceptance: ember generate in-repo-addon', function() {
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

  function initApp() {
    return ember([
      'init',
      '--name=my-app',
      '--skip-npm',
      '--skip-bower'
    ]);
  }

  function initInRepoAddon() {
    return initApp().then(function() {
      return ember([
        'generate',
        'in-repo-addon',
        'my-addon'
      ]);
    });
  }

  function generateInRepoAddon(args) {
    var generateArgs = ['generate'].concat(args);

    return initInRepoAddon().then(function() {
      return ember(generateArgs);
    });
  }

  it('in-repo-addon controller foo', function() {
    return generateInRepoAddon(['controller', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/controllers/foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Controller.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/controllers/foo.js', {
        contains: [
          "export { default } from 'my-addon/controllers/foo';"
        ]
      });
      assertFile('tests/unit/controllers/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('controller:foo'"
        ]
      });
    });
  });

  it('in-repo-addon controller foo/bar', function() {
    return generateInRepoAddon(['controller', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/controllers/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Controller.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/controllers/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/controllers/foo/bar';"
        ]
      });
      assertFile('tests/unit/controllers/foo/bar-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('controller:foo/bar'"
        ]
      });
    });
  });

  it('in-repo-addon component x-foo', function() {
    return generateInRepoAddon(['component', 'x-foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/components/x-foo.js', {
        contains: [
          "import Ember from 'ember';",
          "import layout from '../templates/components/x-foo';",
          "export default Ember.Component.extend({",
          "layout: layout",
          "});"
        ]
      });
      assertFile('lib/my-addon/addon/templates/components/x-foo.hbs', {
        contains: "{{yield}}"
      });
      assertFile('lib/my-addon/app/components/x-foo.js', {
        contains: [
          "export { default } from 'my-addon/components/x-foo';"
        ]
      });
      assertFile('tests/integration/components/x-foo-test.js', {
        contains: [
          "import { moduleForComponent, test } from 'ember-qunit';",
          "import hbs from 'htmlbars-inline-precompile';",
          "moduleForComponent('x-foo'",
          "integration: true",
          "{{x-foo}}",
          "{{#x-foo}}"
        ]
      });
    });
  });

  it('in-repo-addon component-test x-foo', function() {
    return generateInRepoAddon(['component-test', 'x-foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('tests/integration/components/x-foo-test.js', {
        contains: [
          "import { moduleForComponent, test } from 'ember-qunit';",
          "import hbs from 'htmlbars-inline-precompile';",
          "moduleForComponent('x-foo'",
          "integration: true",
          "{{x-foo}}",
          "{{#x-foo}}"
        ]
      });
    });
  });

  it('in-repo-addon component-test x-foo --unit', function() {
    return generateInRepoAddon(['component-test', 'x-foo', '--in-repo-addon=my-addon', '--unit']).then(function() {
      assertFile('tests/unit/components/x-foo-test.js', {
        contains: [
          "import { moduleForComponent, test } from 'ember-qunit';",
          "moduleForComponent('x-foo'",
          "unit: true"
        ]
      });
    });
  });

  it('in-repo-addon component nested/x-foo', function() {
    return generateInRepoAddon(['component', 'nested/x-foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/components/nested/x-foo.js', {
        contains: [
          "import Ember from 'ember';",
          "import layout from '../../templates/components/nested/x-foo';",
          "export default Ember.Component.extend({",
          "layout: layout",
          "});"
        ]
      });
      assertFile('lib/my-addon/addon/templates/components/nested/x-foo.hbs', {
        contains: "{{yield}}"
      });
      assertFile('lib/my-addon/app/components/nested/x-foo.js', {
        contains: [
          "export { default } from 'my-addon/components/nested/x-foo';"
        ]
      });
      assertFile('tests/integration/components/nested/x-foo-test.js', {
        contains: [
          "import { moduleForComponent, test } from 'ember-qunit';",
          "import hbs from 'htmlbars-inline-precompile';",
          "moduleForComponent('nested/x-foo'",
          "integration: true"
        ]
      });
    });
  });

  it('in-repo-addon helper foo-bar', function() {
    return generateInRepoAddon(['helper', 'foo-bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/helpers/foo-bar.js', {
        contains: "import Ember from 'ember';" + EOL + EOL +
                  "export function fooBar(params/*, hash*/) {" + EOL +
                  "  return params;" + EOL +
                  "}" +  EOL + EOL +
                  "export default Ember.Helper.helper(fooBar);"
      });
      assertFile('lib/my-addon/app/helpers/foo-bar.js', {
        contains: [
          "export { default, fooBar } from 'my-addon/helpers/foo-bar';"
        ]
      });
      assertFile('tests/unit/helpers/foo-bar-test.js', {
        contains: "import { fooBar } from '../../../helpers/foo-bar';"
      });
    });
  });

  it('in-repo-addon helper foo/bar-baz', function() {
    return generateInRepoAddon(['helper', 'foo/bar-baz', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/helpers/foo/bar-baz.js', {
        contains: "import Ember from 'ember';" + EOL + EOL +
                  "export function fooBarBaz(params/*, hash*/) {" + EOL +
                  "  return params;" + EOL +
                  "}" + EOL + EOL +
                  "export default Ember.Helper.helper(fooBarBaz);"
      });
      assertFile('lib/my-addon/app/helpers/foo/bar-baz.js', {
        contains: [
          "export { default, fooBarBaz } from 'my-addon/helpers/foo/bar-baz';"
        ]
      });
      assertFile('tests/unit/helpers/foo/bar-baz-test.js', {
        contains: "import { fooBarBaz } from '../../../../helpers/foo/bar-baz';"
      });
    });
  });

  it('in-repo-addon model foo', function() {
    return generateInRepoAddon(['model', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/models/foo.js', {
        contains: [
          "import DS from 'ember-data';",
          "export default DS.Model.extend"
        ]
      });
      assertFile('lib/my-addon/app/models/foo.js', {
        contains: [
          "export { default } from 'my-addon/models/foo';"
        ]
      });
      assertFile('tests/unit/models/foo-test.js', {
        contains: [
          "import { moduleForModel, test } from 'ember-qunit';",
          "moduleForModel('foo'"
        ]
      });
    });
  });

  it('in-repo-addon model foo with attributes', function() {
    return generateInRepoAddon([
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
      '--in-repo-addon=my-addon'
    ]).then(function() {
      assertFile('lib/my-addon/addon/models/foo.js', {
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
      assertFile('lib/my-addon/app/models/foo.js', {
        contains: [
          "export { default } from 'my-addon/models/foo';"
        ]
      });
      assertFile('tests/unit/models/foo-test.js', {
        contains: [
          "needs: [",
          "'model:bar',",
          "'model:baz',",
          "'model:echo',",
          "'model:bravo',",
          "'model:foo-name',",
          "'model:bar-name',",
          "'model:baz-name',",
          "'model:echo-name',",
          "'model:test-name',",
          "'model:bravo-name'",
          "]"
        ]
      });
    });
  });

  it('in-repo-addon model foo/bar', function() {
    return generateInRepoAddon(['model', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/models/foo/bar.js', {
        contains: [
          "import DS from 'ember-data';",
          "export default DS.Model.extend"
        ]
      });
      assertFile('lib/my-addon/app/models/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/models/foo/bar';"
        ]
      });
      assertFile('tests/unit/models/foo/bar-test.js', {
        contains: [
          "import { moduleForModel, test } from 'ember-qunit';",
          "moduleForModel('foo/bar'"
        ]
      });
    });
  });

  it('in-repo-addon route foo', function() {
    return generateInRepoAddon(['route', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/routes/foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Route.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/routes/foo.js', {
        contains: "export { default } from 'my-addon/routes/foo';"
      });
      assertFile('lib/my-addon/addon/templates/foo.hbs', {
        contains: '{{outlet}}'
      });
      assertFile('lib/my-addon/app/templates/foo.js', {
        contains: "export { default } from 'my-addon/templates/foo';"
      });
      assertFile('tests/unit/routes/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('route:foo'"
        ]
      });
    });
  });

  it('in-repo-addon route foo/bar', function() {
    return generateInRepoAddon(['route', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/routes/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.Route.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/routes/foo/bar.js', {
        contains: "export { default } from 'my-addon/routes/foo/bar';"
      });
      assertFile('lib/my-addon/addon/templates/foo/bar.hbs', {
        contains: '{{outlet}}'
      });
      assertFile('lib/my-addon/app/templates/foo/bar.js', {
        contains: "export { default } from 'my-addon/templates/foo/bar';"
      });
      assertFile('tests/unit/routes/foo/bar-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('route:foo/bar'"
        ]
      });
    });
  });

  it('in-repo-addon template foo', function() {
    return generateInRepoAddon(['template', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/templates/foo.hbs');
    });
  });

  it('in-repo-addon template foo/bar', function() {
    return generateInRepoAddon(['template', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/templates/foo/bar.hbs');
    });
  });

  it('in-repo-addon view foo', function() {
    return generateInRepoAddon(['view', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/views/foo.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.View.extend({" + EOL + "})"
        ]
      });
      assertFile('lib/my-addon/app/views/foo.js', {
        contains: [
          "export { default } from 'my-addon/views/foo';"
        ]
      });
      assertFile('tests/unit/views/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('view:foo'"
        ]
      });
    });
  });

  it('in-repo-addon view foo/bar', function() {
    return generateInRepoAddon(['view', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/views/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          "export default Ember.View.extend({" + EOL + "})"
        ]
      });
      assertFile('lib/my-addon/app/views/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/views/foo/bar';"
        ]
      });
      assertFile('tests/unit/views/foo/bar-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('view:foo/bar'"
        ]
      });
    });
  });

  it('in-repo-addon resource foos', function() {
    return generateInRepoAddon(['resource', 'foos', '--in-repo-addon=my-addon']).catch(function(error) {
      expect(error.message).to.include('blueprint does not support ' +
        'generating inside addons.');
    });
  });

  it('in-repo-addon initializer foo', function() {
    return generateInRepoAddon(['initializer', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/initializers/foo.js', {
        contains: "export function initialize(/* application */) {" + EOL +
                  "  // application.inject('route', 'foo', 'service:foo');" + EOL +
                  "}" + EOL +
                  "" + EOL+
                  "export default {" + EOL +
                  "  name: 'foo'," + EOL +
                  "  initialize" + EOL +
                  "};"
      });
      assertFile('lib/my-addon/app/initializers/foo.js', {
        contains: [
          "export { default, initialize } from 'my-addon/initializers/foo';"
        ]
      });
      assertFile('tests/unit/initializers/foo-test.js');
    });
  });

  it('in-repo-addon initializer foo/bar', function() {
    return generateInRepoAddon(['initializer', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/initializers/foo/bar.js', {
        contains: "export function initialize(/* application */) {" + EOL +
                  "  // application.inject('route', 'foo', 'service:foo');" + EOL +
                  "}" + EOL +
                  "" + EOL+
                  "export default {" + EOL +
                  "  name: 'foo/bar'," + EOL +
                  "  initialize" + EOL +
                  "};"
      });
      assertFile('lib/my-addon/app/initializers/foo/bar.js', {
        contains: [
          "export { default, initialize } from 'my-addon/initializers/foo/bar';"
        ]
      });
      assertFile('tests/unit/initializers/foo/bar-test.js');
    });
  });

  it('in-repo-addon mixin foo', function() {
    return generateInRepoAddon(['mixin', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/mixins/foo.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Mixin.create({' + EOL + '});'
        ]
      });
      assertFile('tests/unit/mixins/foo-test.js', {
        contains: [
          "import FooMixin from '../../../mixins/foo';"
        ]
      });
    });
  });

  it('in-repo-addon mixin foo/bar', function() {
    return generateInRepoAddon(['mixin', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/mixins/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Mixin.create({' + EOL + '});'
        ]
      });
      assertFile('tests/unit/mixins/foo/bar-test.js', {
        contains: [
          "import FooBarMixin from '../../../mixins/foo/bar';"
        ]
      });
    });
  });

  it('in-repo-addon mixin foo/bar/baz', function() {
    return generateInRepoAddon(['mixin', 'foo/bar/baz', '--in-repo-addon=my-addon']).then(function() {
      assertFile('tests/unit/mixins/foo/bar/baz-test.js', {
        contains: [
          "import FooBarBazMixin from '../../../mixins/foo/bar/baz';"
        ]
      });
    });
  });

  it('in-repo-addon adapter application', function() {
    return generateInRepoAddon(['adapter', 'application', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/adapters/application.js', {
        contains: [
          "import DS from \'ember-data\';",
          "export default DS.RESTAdapter.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/adapters/application.js', {
        contains: [
          "export { default } from 'my-addon/adapters/application';"
        ]
      });
      assertFile('tests/unit/adapters/application-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('adapter:application'"
        ]
      });
    });
  });

  it('in-repo-addon adapter foo', function() {
    return generateInRepoAddon(['adapter', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/adapters/foo.js', {
        contains: [
          "import DS from \'ember-data\';",
          "export default DS.RESTAdapter.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/adapters/foo.js', {
        contains: [
          "export { default } from 'my-addon/adapters/foo';"
        ]
      });
      assertFile('tests/unit/adapters/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('adapter:foo'"
        ]
      });
    });
  });

  it('in-repo-addon adapter foo/bar (with base class foo)', function() {
    return generateInRepoAddon(['adapter', 'foo/bar', '--in-repo-addon=my-addon', '--base-class=foo']).then(function() {
      assertFile('lib/my-addon/addon/adapters/foo/bar.js', {
        contains: [
          "import FooAdapter from \'../foo\';",
          "export default FooAdapter.extend({" + EOL + "});"
        ]
      });
      assertFile('lib/my-addon/app/adapters/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/adapters/foo/bar';"
        ]
      });
      assertFile('tests/unit/adapters/foo/bar-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('adapter:foo/bar'"
        ]
      });
    });
  });


  it('in-repo-addon serializer foo', function() {
    return generateInRepoAddon(['serializer', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/serializers/foo.js', {
        contains: [
          "import DS from 'ember-data';",
          'export default DS.RESTSerializer.extend({' + EOL + '});'
        ]
      });
      assertFile('lib/my-addon/app/serializers/foo.js', {
        contains: [
          "export { default } from 'my-addon/serializers/foo';"
        ]
      });
      assertFile('tests/unit/serializers/foo-test.js', {
        contains: [
          "import { moduleForModel, test } from 'ember-qunit';",
        ]
      });
    });
  });

  it('in-repo-addon serializer foo/bar', function() {
    return generateInRepoAddon(['serializer', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/serializers/foo/bar.js', {
        contains: [
          "import DS from 'ember-data';",
          'export default DS.RESTSerializer.extend({' + EOL + '});'
        ]
      });
      assertFile('lib/my-addon/app/serializers/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/serializers/foo/bar';"
        ]
      });
      assertFile('tests/unit/serializers/foo/bar-test.js', {
        contains: [
          "import { moduleForModel, test } from 'ember-qunit';",
          "moduleForModel('foo/bar'"
        ]
      });
    });
  });

  it('in-repo-addon transform foo', function() {
    return generateInRepoAddon(['transform', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/transforms/foo.js', {
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
      assertFile('lib/my-addon/app/transforms/foo.js', {
        contains: [
          "export { default } from 'my-addon/transforms/foo';"
        ]
      });
      assertFile('tests/unit/transforms/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('transform:foo'"
        ]
      });
    });
  });

  it('in-repo-addon transform foo/bar', function() {
    return generateInRepoAddon(['transform', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/transforms/foo/bar.js', {
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
      assertFile('lib/my-addon/app/transforms/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/transforms/foo/bar';"
        ]
      });
      assertFile('tests/unit/transforms/foo/bar-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('transform:foo/bar'"
        ]
      });
    });
  });

  it('in-repo-addon util foo-bar', function() {
    return generateInRepoAddon(['util', 'foo-bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/utils/foo-bar.js', {
        contains: 'export default function fooBar() {' + EOL +
                  '  return true;' + EOL +
                  '}'
      });
      assertFile('lib/my-addon/app/utils/foo-bar.js', {
        contains: [
          "export { default } from 'my-addon/utils/foo-bar';"
        ]
      });
      assertFile('tests/unit/utils/foo-bar-test.js', {
        contains: [
          "import fooBar from '../../../utils/foo-bar';"
        ]
      });
    });
  });

  it('in-repo-addon util foo-bar/baz', function() {
    return generateInRepoAddon(['util', 'foo/bar-baz', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/utils/foo/bar-baz.js', {
        contains: 'export default function fooBarBaz() {' + EOL +
                  '  return true;' + EOL +
                  '}'
      });
      assertFile('lib/my-addon/app/utils/foo/bar-baz.js', {
        contains: [
          "export { default } from 'my-addon/utils/foo/bar-baz';"
        ]
      });
      assertFile('tests/unit/utils/foo/bar-baz-test.js', {
        contains: [
          "import fooBarBaz from '../../../utils/foo/bar-baz';"
        ]
      });
    });
  });

  it('in-repo-addon service foo', function() {
    return generateInRepoAddon(['service', 'foo', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/services/foo.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Service.extend({' + EOL + '});'
        ]
      });
      assertFile('lib/my-addon/app/services/foo.js', {
        contains: [
          "export { default } from 'my-addon/services/foo';"
        ]
      });
      assertFile('tests/unit/services/foo-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('service:foo'"
        ]
      });
    });
  });

  it('in-repo-addon service foo/bar', function() {
    return generateInRepoAddon(['service', 'foo/bar', '--in-repo-addon=my-addon']).then(function() {
      assertFile('lib/my-addon/addon/services/foo/bar.js', {
        contains: [
          "import Ember from 'ember';",
          'export default Ember.Service.extend({' + EOL + '});'
        ]
      });
      assertFile('lib/my-addon/app/services/foo/bar.js', {
        contains: [
          "export { default } from 'my-addon/services/foo/bar';"
        ]
      });
      assertFile('tests/unit/services/foo/bar-test.js', {
        contains: [
          "import { moduleFor, test } from 'ember-qunit';",
          "moduleFor('service:foo/bar'"
        ]
      });
    });
  });
  it('in-repo-addon acceptance-test foo', function() {
    return generateInRepoAddon(['acceptance-test', 'foo', '--in-repo-addon=my-addon']).then(function() {
      var expected = path.join(__dirname, '../fixtures/generate/acceptance-test-expected.js');

      assertFileEquals('tests/acceptance/foo-test.js', expected);
      assertFileToNotExist('app/acceptance-tests/foo.js');
    });
  });

  it('in-repo-addon adds path to lib', function() {
    return initInRepoAddon().then(function() {
      assertFile('package.json', {
        contains: [
          'lib/my-addon'
        ]
      });
    });
  });

});
