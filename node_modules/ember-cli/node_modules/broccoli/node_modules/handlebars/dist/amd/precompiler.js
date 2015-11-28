define(['exports', 'fs', './handlebars', 'path', 'source-map', 'uglify-js'], function (exports, _fs, _handlebars, _path, _sourceMap, _uglifyJs) {
  'use strict';

  var _interopRequire = function (obj) { return obj && obj.__esModule ? obj['default'] : obj; };

  /*eslint-disable no-console */

  var _fs2 = _interopRequire(_fs);

  var _uglify = _interopRequire(_uglifyJs);

  module.exports.cli = function (opts) {
    if (opts.version) {
      console.log(_handlebars.VERSION);
      return;
    }

    if (!opts.templates.length) {
      throw new _handlebars.Exception('Must define at least one template or directory.');
    }

    opts.templates.forEach(function (template) {
      try {
        _fs2.statSync(template);
      } catch (err) {
        throw new _handlebars.Exception('Unable to open template file "' + template + '"');
      }
    });

    if (opts.simple && opts.min) {
      throw new _handlebars.Exception('Unable to minimize simple output');
    }
    if (opts.simple && (opts.templates.length !== 1 || _fs2.statSync(opts.templates[0]).isDirectory())) {
      throw new _handlebars.Exception('Unable to output multiple templates in simple mode');
    }

    // Convert the known list into a hash
    var known = {};
    if (opts.known && !Array.isArray(opts.known)) {
      opts.known = [opts.known];
    }
    if (opts.known) {
      for (var i = 0, len = opts.known.length; i < len; i++) {
        known[opts.known[i]] = true;
      }
    }

    // Build file extension pattern
    var extension = opts.extension.replace(/[\\^$*+?.():=!|{}\-\[\]]/g, function (arg) {
      return '\\' + arg;
    });
    extension = new RegExp('\\.' + extension + '$');

    var output = new _sourceMap.SourceNode();
    if (!opts.simple) {
      if (opts.amd) {
        output.add('define([\'' + opts.handlebarPath + 'handlebars.runtime\'], function(Handlebars) {\n  Handlebars = Handlebars["default"];');
      } else if (opts.commonjs) {
        output.add('var Handlebars = require("' + opts.commonjs + '");');
      } else {
        output.add('(function() {\n');
      }
      output.add('  var template = Handlebars.template, templates = ');
      if (opts.namespace) {
        output.add(opts.namespace);
        output.add(' = ');
        output.add(opts.namespace);
        output.add(' || ');
      }
      output.add('{};\n');
    }
    function processTemplate(template, root) {
      var path = template,
          stat = _fs2.statSync(path);
      if (stat.isDirectory()) {
        _fs2.readdirSync(template).map(function (file) {
          var childPath = template + '/' + file;

          if (extension.test(childPath) || _fs2.statSync(childPath).isDirectory()) {
            processTemplate(childPath, root || template);
          }
        });
      } else {
        var data = _fs2.readFileSync(path, 'utf8');

        if (opts.bom && data.indexOf('﻿') === 0) {
          data = data.substring(1);
        }

        var options = {
          knownHelpers: known,
          knownHelpersOnly: opts.o
        };

        if (opts.map) {
          options.srcName = path;
        }
        if (opts.data) {
          options.data = true;
        }

        // Clean the template name
        if (!root) {
          template = _path.basename(template);
        } else if (template.indexOf(root) === 0) {
          template = template.substring(root.length + 1);
        }
        template = template.replace(extension, '');

        var precompiled = _handlebars.precompile(data, options);

        // If we are generating a source map, we have to reconstruct the SourceNode object
        if (opts.map) {
          var consumer = new _sourceMap.SourceMapConsumer(precompiled.map);
          precompiled = _sourceMap.SourceNode.fromStringWithSourceMap(precompiled.code, consumer);
        }

        if (opts.simple) {
          output.add([precompiled, '\n']);
        } else if (opts.partial) {
          if (opts.amd && (opts.templates.length == 1 && !_fs2.statSync(opts.templates[0]).isDirectory())) {
            output.add('return ');
          }
          output.add(['Handlebars.partials[\'', template, '\'] = template(', precompiled, ');\n']);
        } else {
          if (opts.amd && (opts.templates.length == 1 && !_fs2.statSync(opts.templates[0]).isDirectory())) {
            output.add('return ');
          }
          output.add(['templates[\'', template, '\'] = template(', precompiled, ');\n']);
        }
      }
    }

    opts.templates.forEach(function (template) {
      processTemplate(template, opts.root);
    });

    // Output the content
    if (!opts.simple) {
      if (opts.amd) {
        if (opts.templates.length > 1 || opts.templates.length == 1 && _fs2.statSync(opts.templates[0]).isDirectory()) {
          if (opts.partial) {
            output.add('return Handlebars.partials;\n');
          } else {
            output.add('return templates;\n');
          }
        }
        output.add('});');
      } else if (!opts.commonjs) {
        output.add('})();');
      }
    }

    if (opts.map) {
      output.add('\n//# sourceMappingURL=' + opts.map + '\n');
    }

    output = output.toStringWithSourceMap();
    output.map = output.map + '';

    if (opts.min) {
      output = _uglify.minify(output.code, {
        fromString: true,

        outSourceMap: opts.map,
        inSourceMap: JSON.parse(output.map)
      });
      if (opts.map) {
        output.code += '\n//# sourceMappingURL=' + opts.map + '\n';
      }
    }

    if (opts.map) {
      _fs2.writeFileSync(opts.map, output.map, 'utf8');
    }
    output = output.code;

    if (opts.output) {
      _fs2.writeFileSync(opts.output, output, 'utf8');
    } else {
      console.log(output);
    }
  };
});