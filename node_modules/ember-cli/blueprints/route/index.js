/*jshint node:true*/

var fs          = require('fs-extra');
var path        = require('path');
var chalk       = require('chalk');
var EmberRouterGenerator = require('ember-router-generator');

module.exports = {
  description: 'Generates a route and a template, and registers the route with the router.',

  availableOptions: [
    {
      name: 'path',
      type: String,
      default: ''
    },
    {
      name: 'skip-router',
      type: Boolean,
      default: false
    },
    {
      name: 'reset-namespace',
      type: Boolean
    }
  ],

  fileMapTokens: function() {
    return {
      __templatepath__: function(options) {
        if (options.pod) {
          return path.join(options.podPath, options.dasherizedModuleName);
        }
        return 'templates';
      },
      __templatename__: function(options) {
        if (options.pod) {
          return 'template';
        }
        return options.dasherizedModuleName;
      },
      __root__: function(options) {
        if (options.inRepoAddon) {
          return path.join('lib', options.inRepoAddon, 'addon');
        }

        if (options.inDummy) {
          return path.join('tests','dummy','app');
        }

        if (options.inAddon) {
          return 'addon';
        }

        return 'app';
      }
    };
  },

  shouldEntityTouchRouter: function(name) {
    var isIndex = name === 'index';
    var isBasic = name === 'basic';
    var isApplication = name === 'application';

    return !isBasic && !isIndex && !isApplication;
  },

  shouldTouchRouter: function(name, options) {
    var entityTouchesRouter = this.shouldEntityTouchRouter(name);
    var isDummy = !!options.dummy;
    var isAddon = !!options.project.isEmberCLIAddon();
    var isAddonDummyOrApp = (isDummy === isAddon);

    return (entityTouchesRouter && isAddonDummyOrApp && !options.dryRun && !options.inRepoAddon && !options.skipRouter);
  },

  afterInstall: function(options) {
    updateRouter.call(this, 'add', options);
  },

  afterUninstall: function(options) {
    updateRouter.call(this, 'remove', options);
  }
};

function updateRouter(action, options) {
  var entity = options.entity;
  var actionColorMap = {
    add: 'green',
    remove: 'red'
  };
  var color = actionColorMap[action] || 'gray';

  if (this.shouldTouchRouter(entity.name, options)) {
    writeRoute(action, entity.name, options);

    this.ui.writeLine('updating router');
    this._writeStatusToUI(chalk[color], action + ' route', entity.name);
  }
}

function findRouter(options) {
  var routerPathParts = [options.project.root];

  if (options.dummy && options.project.isEmberCLIAddon()) {
    routerPathParts = routerPathParts.concat(['tests', 'dummy', 'app', 'router.js']);
  } else {
    routerPathParts = routerPathParts.concat(['app', 'router.js']);
  }

  return routerPathParts;
}

function writeRoute(action, name, options) {
  var routerPath = path.join.apply(null, findRouter(options));
  var source = fs.readFileSync(routerPath, 'utf-8');

  var routes = new EmberRouterGenerator(source);
  var newRoutes = routes[action](name, options);

  fs.writeFileSync(routerPath, newRoutes.code());
}
