/*jshint node:true*/

var inflection  = require('inflection');
var stringUtils = require('ember-cli-string-utils');
var EOL         = require('os').EOL;

module.exports = {
  description: 'Generates an ember-data model.',

  anonymousOptions: [
    'name',
    'attr:type'
  ],

  locals: function(options) {
    var attrs = [];
    var needs = [];
    var entityOptions = options.entity.options;

    for (var name in entityOptions) {
      var type = entityOptions[name] || '';
      var foreignModel = name;
      if (type.indexOf(':') > -1) {
        foreignModel = type.split(':')[1];
        type = type.split(':')[0];
      }
      var dasherizedName = stringUtils.dasherize(name);
      var camelizedName = stringUtils.camelize(name);
      var dasherizedType = stringUtils.dasherize(type);
      var dasherizedForeignModel = stringUtils.dasherize(foreignModel);
      var dasherizedForeignModelSingular = inflection.singularize(dasherizedForeignModel);

      var attr;
      if (/has-many/.test(dasherizedType)) {
        var camelizedNamePlural = inflection.pluralize(camelizedName);
        attr = dsAttr(dasherizedForeignModelSingular, dasherizedType);
        attrs.push(camelizedNamePlural + ': ' + attr);
      } else if (/belongs-to/.test(dasherizedType)) {
        attr = dsAttr(dasherizedForeignModel, dasherizedType);
        attrs.push(camelizedName + ': ' + attr);
      } else {
        attr = dsAttr(dasherizedName, dasherizedType);
        attrs.push(camelizedName + ': ' + attr);
      }

      if (/has-many|belongs-to/.test(dasherizedType)) {
        needs.push("'model:" + dasherizedForeignModelSingular + "'");
      }
    }
    var needsDeduplicated = needs.filter(function(need, i) {
      return needs.indexOf(need) === i;
    });

    attrs = attrs.join(',' + EOL + '  ');
    needs = '  needs: [' + needsDeduplicated.join(', ') + ']';

    return {
      attrs: attrs,
      needs: needs
    };
  }
};

function dsAttr(name, type) {
  switch (type) {
  case 'belongs-to':
    return 'DS.belongsTo(\'' + name + '\')';
  case 'has-many':
    return 'DS.hasMany(\'' + name + '\')';
  case '':
    //"If you don't specify the type of the attribute, it will be whatever was provided by the server"
    //http://emberjs.com/guides/models/defining-models/
    return 'DS.attr()';
  default:
    return 'DS.attr(\'' + type + '\')';
  }
}
