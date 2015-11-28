/*jshint node:true*/

var stringUtil         = require('ember-cli-string-utils');
var pathUtil           = require('ember-cli-path-utils');
var validComponentName = require('../../lib/utilities/valid-component-name');
var getPathOption      = require('../../lib/utilities/get-component-path-option');
var path               = require('path');

var normalizeEntityName = require('ember-cli-normalize-entity-name');

module.exports = {
  description: 'Generates a component. Name must contain a hyphen.',

  availableOptions: [
    {
      name: 'path',
      type: String,
      default: 'components',
      aliases:[
        {'no-path': ''}
      ]
    }
  ],

  fileMapTokens: function() {
    return {
      __path__: function(options) {
        if (options.pod) {
          return path.join(options.podPath, options.locals.path, options.dasherizedModuleName);
        }
        return 'components';
      },
      __templatepath__: function(options) {
        if (options.pod) {
          return path.join(options.podPath, options.locals.path, options.dasherizedModuleName);
        }
        return 'templates/components';
      },
      __templatename__: function(options) {
        if (options.pod) {
          return 'template';
        }
        return options.dasherizedModuleName;
      }
    };
  },

  normalizeEntityName: function(entityName) {
    entityName = normalizeEntityName(entityName);

    return validComponentName(entityName);
  },

  locals: function(options) {
    var templatePath   = '';
    var importTemplate = '';
    var contents       = '';
    // if we're in an addon, build import statement
    if (options.project.isEmberCLIAddon() || options.inRepoAddon && !options.inDummy) {
      if(options.pod) {
        templatePath   = './template';
      } else {
        templatePath   = pathUtil.getRelativeParentPath(options.entity.name) +
          'templates/components/' + stringUtil.dasherize(options.entity.name);
      }
      importTemplate   = 'import layout from \'' + templatePath + '\';\n';
      contents         = '\n  layout: layout';
    }

    return {
      importTemplate: importTemplate,
      contents: contents,
      path: getPathOption(options)
    };
  }
};
