'use strict';

/**
@module ember-cli
*/
var FileInfo            = require('./file-info');
var Promise             = require('../ext/promise');
var chalk               = require('chalk');
var MarkdownColor       = require('../utilities/markdown-color');
var printableProperties = require('../utilities/printable-properties').blueprint;
var sequence            = require('../utilities/sequence');
var deprecateUI         = require('../utilities/deprecate').deprecateUI;
var fs                  = require('fs-extra');
var existsSync          = require('exists-sync');
var inflector           = require('inflection');
var minimatch           = require('minimatch');
var path                = require('path');
var stat                = Promise.denodeify(fs.stat);
var stringUtils         = require('ember-cli-string-utils');
var compact             = require('lodash/array/compact');
var intersect           = require('lodash/array/intersection');
var uniq                = require('lodash/array/uniq');
var zipObject           = require('lodash/array/zipObject');
var contains            = require('lodash/collection/contains');
var any                 = require('lodash/collection/some');
var cloneDeep           = require('lodash/lang/cloneDeep');
var keys                = require('lodash/object/keys');
var merge               = require('lodash/object/merge');
var values              = require('lodash/object/values');
var walkSync            = require('walk-sync');
var writeFile           = Promise.denodeify(fs.outputFile);
var removeFile          = Promise.denodeify(fs.remove);
var SilentError         = require('silent-error');
var CoreObject          = require('core-object');
var EOL                 = require('os').EOL;
var bowEpParser         = require('bower-endpoint-parser');
var debug               = require('debug')('ember-cli:blueprint');
var normalizeEntityName = require('ember-cli-normalize-entity-name');

module.exports = Blueprint;

/**
  A blueprint is a bundle of template files with optional install
  logic.

  Blueprints follow a simple structure. Let's take the built-in
  `controller` blueprint as an example:

  ```
  blueprints/controller
  ├── files
  │   ├── app
  │   │   └── __path__
  │   │       └── __name__.js
  └── index.js

  blueprints/controller-test
  ├── files
  │   └── tests
  │       └── unit
  │           └── controllers
  │               └── __test__.js
  └── index.js
  ```

  ## Files

  `files` contains templates for the all the files to be
  installed into the target directory.

  The `__name__` token is subtituted with the dasherized
  entity name at install time. For example, when the user
  invokes `ember generate controller foo` then `__name__` becomes
  `foo`. When the `--pod` flag is used, for example `ember
  generate controller foo --pod` then `__name__` becomes
  `controller`.

  The `__path__` token is substituted with the blueprint
  name at install time. For example, when the user invokes
  `ember generate controller foo` then `__path__` becomes
  `controller`. When the `--pod` flag is used, for example
  `ember generate controller foo --pod` then `__path__`
  becomes `foo` (or `<podModulePrefix>/foo` if the
  podModulePrefix is defined). This token is primarily for
  pod support, and is only necessary if the blueprint can be
  used in pod structure. If the blueprint does not require pod
  support, simply use the blueprint name instead of the
  `__path__` token.

  The `__test__` token is substituted with the dasherized
  entity name and appended with `-test` at install time.
  This token is primarily for pod support and only necessary
  if the blueprint requires support for a pod structure. If
  the blueprint does not require pod support, simply use the
  `__name__` token instead.

  ## Template Variables (AKA Locals)

  Variables can be inserted into templates with
  `<%= someVariableName %>`.

  For example, the built-in `util` blueprint
  `files/app/utils/__name__.js` looks like this:

  ```js
  export default function <%= camelizedModuleName %>() {
    return true;
  }
  ```

  `<%= camelizedModuleName %>` is replaced with the real
  value at install time.

  The following template variables are provided by default:

  - `dasherizedPackageName`
  - `classifiedPackageName`
  - `dasherizedModuleName`
  - `classifiedModuleName`
  - `camelizedModuleName`

  `packageName` is the project name as found in the project's
  `package.json`.

  `moduleName` is the name of the entity being generated.

  The mechanism for providing custom template variables is
  described below.

  ## Index.js

  Custom installation and uninstallation behaviour can be added
  by overriding the hooks documented below. `index.js` should
  export a plain object, which will extend the prototype of the
  `Blueprint` class. If needed, the original `Blueprint` prototype
  can be accessed through the `_super` property.

  ```js
  module.exports = {
    locals: function(options) {
      // Return custom template variables here.
      return {};
    },

    normalizeEntityName: function(entityName) {
      // Normalize and validate entity name here.
      return entityName;
    },

    fileMapTokens: function(options) (
      // Return custom tokens to be replaced in your files
      return {
        __token__: function(options){
          // logic to determine value goes here
          return 'value';
        }
      }
    },

    beforeInstall: function(options) {},
    afterInstall: function(options) {},
    beforeUninstall: function(options) {},
    afterUninstall: function(options) {}

  };
  ```

  ## Blueprint Hooks

  As shown above, the following hooks are available to
  blueprint authors:

  - `locals`
  - `normalizeEntityName`
  - `fileMapTokens`
  - `beforeInstall`
  - `afterInstall`
  - `beforeUninstall`
  - `afterUninstall`

  ### locals

  Use `locals` to add custom tempate variables. The method
  receives one argument: `options`. Options is an object
  containing general and entity-specific options.

  When the following is called on the command line:

  ```sh
  ember generate controller foo --type=array --dry-run
  ```

  The object passed to `locals` looks like this:

  ```js
  {
    entity: {
      name: 'foo',
      options: {
        type: 'array'
      }
    },
    dryRun: true
  }
  ```

  This hook must return an object. It will be merged with the
  aforementioned default locals.

  ### normalizeEntityName

  Use the `normalizeEntityName` hook to add custom normalization and
  validation of the provided entity name. The default hook does not
  make any changes to the entity name, but makes sure an entity name
  is present and that it doesn't have a trailing slash.

  This hook receives the entity name as its first argument. The string
  returned by this hook will be used as the new entity name.

  ### fileMapTokens

  Use `fileMapTokens` to add custom fileMap tokens for use
  in the `mapFile` method. The hook must return an object in the
  following pattern:

  ```js
  {
    __token__: function(options){
      // logic to determine value goes here
      return 'value';
    }
  }
  ```

  It will be merged with the default `fileMapTokens`, and can be used
  to override any of the default tokens.

  Tokens are used in the files folder (see `files`), and get replaced with
  values when the `mapFile` method is called.

  ### beforeInstall & beforeUninstall

  Called before any of the template files are processed and receives
  the the `options` and `locals` hashes as parameters. Typically used for
  validating any additional command line options or for any asynchronous
  setup that is needed.   As an example, the `controller` blueprint validates
  its `--type` option in this hook.  If you need to run any asynchronous code,
  wrap it in a promise and return that promise from these hooks.  This will
  ensure that your code is executed correctly.

  ### afterInstall & afterUninstall

  The `afterInstall` and `afterUninstall` hooks receives the same
  arguments as `locals`. Use it to perform any custom work after the
  files are processed. For example, the built-in `route` blueprint
  uses these hooks to add and remove relevant route declarations in
  `app/router.js`.

  ### Overriding Install

  If you don't want your blueprint to install the contents of
  `files` you can override the `install` method. It receives the
  same `options` object described above and must return a promise.
  See the built-in `resource` blueprint for an example of this.

  @class Blueprint
  @constructor
  @extends CoreObject
  @param {String} [blueprintPath]
*/
function Blueprint(blueprintPath) {
  this.path = blueprintPath;
  this.name = path.basename(blueprintPath);
}

Blueprint.__proto__ = CoreObject;
Blueprint.prototype.constructor = Blueprint;

Blueprint.prototype.availableOptions = [];
Blueprint.prototype.anonymousOptions = ['name'];

/**
  Used to determine the path to where files will be stored. By default
  this is `path.join(this.path, 'files)`.

  @method filesPath
  @return {String} Path to the blueprints files directory.
*/

Blueprint.prototype.filesPath = function() {
  return path.join(this.path, 'files');
};

/**
  Used to retrieve files for blueprint. The `file` param is an
  optional string that is turned into a glob.

  @method files
  @return {Array} Contents of the blueprint's files directory
*/
Blueprint.prototype.files = function() {
  if (this._files) { return this._files; }

  var filesPath = this.filesPath();
  if (existsSync(filesPath)) {
    this._files = walkSync(filesPath);
  } else {
    this._files = [];
  }

  return this._files;
};

/**
  @method srcPath
  @param {String} file
  @return {String} Resolved path to the file
*/
Blueprint.prototype.srcPath = function(file) {
  return path.resolve(this.filesPath(), file);
};

/**
  Hook for normalizing entity name
  @method normalizeEntityName
  @param {String} entityName
  @return {null}
*/
Blueprint.prototype.normalizeEntityName = function(entityName) {
  return normalizeEntityName(entityName);
};

/**
  Write a status and message to the UI
  @private
  @method _writeStatusToUI
  @param {Function} chalkColor
  @param {String} keyword
  @param {String} message
*/
Blueprint.prototype._writeStatusToUI = function(chalkColor, keyword, message) {
  if (this.ui) {
    this.ui.writeLine('  ' + chalkColor(keyword) + ' ' + message);
  }
};

/**
  @private
  @method _writeFile
  @param {Object} info
  @return {Promise}
*/
Blueprint.prototype._writeFile = function(info) {
  if (!this.dryRun) {
    return writeFile(info.outputPath, info.render());
  }
};

/**
  Actions lookup
  @private
*/

Blueprint.prototype._actions = {
  write: function(info) {
    this._writeStatusToUI(chalk.green, 'create', info.displayPath);
    return this._writeFile(info);
  },
  skip: function(info) {
    var label = 'skip';

    if (info.resolution === 'identical') {
      label = 'identical';
    }

    this._writeStatusToUI(chalk.yellow, label, info.displayPath);
  },

  overwrite: function(info) {
    this._writeStatusToUI(chalk.yellow, 'overwrite', info.displayPath);
    return this._writeFile(info);
  },

  edit: function(info) {
    this._writeStatusToUI(chalk.green, 'edited', info.displayPath);
  },

  remove: function(info) {
    this._writeStatusToUI(chalk.red, 'remove', info.displayPath);
    if (!this.dryRun) {
      return removeFile(info.outputPath);
    }
  }
};

/**
  Calls an action.
  @private
  @method _commit
  @param {Object} result
  @return {Promise}
  @throws {Error} Action doesn't exist.
*/
Blueprint.prototype._commit = function(result) {
  var action = this._actions[result.action];

  if (action) {
    return action.call(this, result);
  } else {
    throw new Error('Tried to call action \"' + result.action + '\" but it does not exist');
  }
};

/**
  Prints warning for pod unsupported.
  @private
  @method _checkForPod
*/
Blueprint.prototype._checkForPod = function(verbose) {
  if (!this.hasPathToken && this.pod && verbose) {
    this.ui.writeLine(chalk.yellow('You specified the pod flag, but this' +
      ' blueprint does not support pod structure. It will be generated with' +
      ' the default structure.'));
  }
};

/**
  @private
  @method _normalizeEntityName
  @param {Object} entity
*/
Blueprint.prototype._normalizeEntityName = function(entity) {
  if (entity) {
    entity.name = this.normalizeEntityName(entity.name);
  }
};

/**
  @private
  @method _checkInRepoAddonExists
  @param {String} inRepoAddon
*/
Blueprint.prototype._checkInRepoAddonExists = function(inRepoAddon) {
  if (inRepoAddon) {
    if (!inRepoAddonExists(inRepoAddon, this.project.root)) {
      throw new SilentError('You specified the in-repo-addon flag, but the' +
        ' in-repo-addon \'' + inRepoAddon + '\' does not exist. Please' +
        ' check the name and try again.');
    }
  }
};

/**
  @private
  @method _process
  @param {Object} options
  @param {Function} beforeHook
  @param {Function} process
  @param {Function} afterHook
*/
Blueprint.prototype._process = function(options, beforeHook, process, afterHook) {
  var intoDir = options.target;
  var locals  = this._locals(options);

  return Promise.resolve()
    .then(beforeHook.bind(this, options, locals))
    .then(process.bind(this, intoDir, locals)).map(this._commit.bind(this))
    .then(afterHook.bind(this, options));
};

/**
  @method install
  @param {Object} options
  @return {Promise}
*/
Blueprint.prototype.install = function(options) {
  var ui       = this.ui     = options.ui;
  var dryRun   = this.dryRun = options.dryRun;
  this.project = options.project;
  this.pod     = options.pod;
  this.hasPathToken = hasPathToken(this.files());

  podDeprecations(this.project.config(), ui);

  ui.writeLine('installing ' + this.name);

  if (dryRun) {
    ui.writeLine(chalk.yellow('You specified the dry-run flag, so no' +
      ' changes will be written.'));
  }

  this._normalizeEntityName(options.entity);
  this._checkForPod(options.verbose);
  this._checkInRepoAddonExists(options.inRepoAddon);

  debug('START: processing blueprint: `%s`', this.name);
  var start = new Date();
  return this._process(
    options,
    this.beforeInstall,
    this.processFiles,
    this.afterInstall).finally(function() {
      debug('END: processing blueprint: `%s` in (%dms)', this.name, new Date() - start);
    }.bind(this));
};

/**
  @method uninstall
  @param {Object} options
  @return {Promise}
*/
Blueprint.prototype.uninstall = function(options) {
  var ui       = this.ui     = options.ui;
  var dryRun   = this.dryRun = options.dryRun;
  this.project = options.project;
  this.pod     = options.pod;
  this.hasPathToken = hasPathToken(this.files());

  podDeprecations(this.project.config(), ui);

  ui.writeLine('uninstalling ' + this.name);

  if (dryRun) {
    ui.writeLine(chalk.yellow('You specified the dry-run flag, so no' +
      ' files will be deleted.'));
  }

  this._normalizeEntityName(options.entity);
  this._checkForPod(options.verbose);

  return this._process(
    options,
    this.beforeUninstall,
    this.processFilesForUninstall,
    this.afterUninstall);
};

/**
  Hook for running operations before install.
  @method beforeInstall
  @return {Promise|null}
*/
Blueprint.prototype.beforeInstall = function() {};

/**
  Hook for running operations after install.
  @method afterInstall
  @return {Promise|null}
*/
Blueprint.prototype.afterInstall = function() {};

/**
  Hook for running operations before uninstall.
  @method beforeUninstall
  @return {Promise|null}
*/
Blueprint.prototype.beforeUninstall = function() {};

/**
  Hook for running operations after uninstall.
  @method afterUninstall
  @return {Promise|null}
*/
Blueprint.prototype.afterUninstall = function() {};

/**
  Hook for adding additional locals
  @method locals
  @return {Object|null}
*/
Blueprint.prototype.locals = function() {};

/**
  Hook to add additional or override existing fileMapTokens.
  @method fileMapTokens
  @return {Object|null}
*/
Blueprint.prototype.fileMapTokens = function() {
};

/**
  @private
  @method _fileMapTokens
  @param {Object} options
  @return {Object}
*/
Blueprint.prototype._fileMapTokens = function(options) {
  var standardTokens = {
    __name__: function(options) {
      if (options.pod && options.hasPathToken) {
        return options.blueprintName;
      }
      return options.dasherizedModuleName;
    },
    __path__: function(options) {
      var blueprintName = options.blueprintName;

      if(blueprintName.match(/-test/)) {
        blueprintName = options.blueprintName.slice(0, options.blueprintName.indexOf('-test'));
      }
      if (options.pod && options.hasPathToken) {
        return path.join(options.podPath, options.dasherizedModuleName);
      }
      return inflector.pluralize(blueprintName);
    },
    __root__: function(options) {
      if (options.inRepoAddon) {
        return path.join('lib',options.inRepoAddon, 'addon');
      }
      if (options.inDummy) {
        return path.join('tests','dummy','app');
      }
      if (options.inAddon) {
        return 'addon';
      }
      return 'app';
    },
    __test__: function(options) {
      if (options.pod && options.hasPathToken) {
        return options.blueprintName;
      }
      return options.dasherizedModuleName + '-test';
    }
  };

  var customTokens = this.fileMapTokens(options) || options.fileMapTokens || {};
  return merge(standardTokens, customTokens);
};

/**
  Used to generate fileMap tokens for mapFile.

  @method generateFileMap
  @param {Object} fileMapVariables
  @return {Object}
*/
Blueprint.prototype.generateFileMap = function(fileMapVariables){
  var tokens        = this._fileMapTokens(fileMapVariables);
  var fileMapValues = values(tokens);
  var tokenValues   = fileMapValues.map(function(token) { return token(fileMapVariables); });
  var tokenKeys     = keys(tokens);
  return zipObject(tokenKeys,tokenValues);
};

/**
  @method buildFileInfo
  @param {Function} destPath
  @param {Object} templateVariables
  @param {String} file
  @return {FileInfo}
*/
Blueprint.prototype.buildFileInfo = function(destPath, templateVariables, file) {
  var mappedPath = this.mapFile(file, templateVariables);

  return new FileInfo({
    action: 'write',
    outputPath: destPath(mappedPath),
    displayPath: path.normalize(mappedPath),
    inputPath: this.srcPath(file),
    templateVariables: templateVariables,
    ui: this.ui
  });
};

/**
  @method isUpdate
  @return {Boolean}
*/
Blueprint.prototype.isUpdate = function() {
  if (this.project && this.project.isEmberCLIProject) {
    return this.project.isEmberCLIProject();
  }
};

/**
  @private
  @method _getFileInfos
  @param {Array} files
  @param {String} intoDir
  @param {Object} templateVariables
  @return {Array} file infos
*/
Blueprint.prototype._getFileInfos = function(files, intoDir, templateVariables) {
  return files.map(this.buildFileInfo.bind(this, destPath.bind(null, intoDir), templateVariables));
};

/**
  Add update files to ignored files
  @private
  @method _ignoreUpdateFiles
*/
Blueprint.prototype._ignoreUpdateFiles = function() {
  if (this.isUpdate()) {
    Blueprint.ignoredFiles = Blueprint.ignoredFiles.concat(Blueprint.ignoredUpdateFiles);
  }
};

/**
  @private
  @method _getFilesForInstall
  @param {Array} targetFiles
  @return {Array} files
*/
Blueprint.prototype._getFilesForInstall = function(targetFiles) {
  var files = this.files();

  // if we've defined targetFiles, get file info on ones that match
  return targetFiles && targetFiles.length > 0 && intersect(files, targetFiles) || files;
};

/**
  @private
  @method _checkForNoMatch
  @param {Array} fileInfos
  @param {String} rawArgs
*/
Blueprint.prototype._checkForNoMatch = function(fileInfos, rawArgs) {
  if (fileInfos.filter(isFilePath).length < 1 && rawArgs) {
    this.ui.writeLine(chalk.yellow('The globPattern \"' + rawArgs +
      '\" did not match any files, so no file updates will be made.'));
  }
};

function finishProcessingForInstall(infos) {
  infos.forEach(markIdenticalToBeSkipped);

  var infosNeedingConfirmation = infos.reduce(gatherConfirmationMessages, []);

  return sequence(infosNeedingConfirmation).returns(infos);
}

function finishProcessingForUninstall(infos) {
  infos.forEach(markToBeRemoved);
  return infos;
}

/**
  @method processFiles
  @param {String} intoDir
  @param {Object} templateVariables
*/
Blueprint.prototype.processFiles = function(intoDir, templateVariables) {
  var files = this._getFilesForInstall(templateVariables.targetFiles);
  var fileInfos = this._getFileInfos(files, intoDir, templateVariables);

  this._checkForNoMatch(fileInfos, templateVariables.rawArgs);

  this._ignoreUpdateFiles();

  return Promise.filter(fileInfos, isValidFile).
    map(prepareConfirm).
    then(finishProcessingForInstall);
};

/**
  @method processFilesForUninstall
  @param {String} intoDir
  @param {Object} templateVariables
*/
Blueprint.prototype.processFilesForUninstall = function(intoDir, templateVariables) {
  var fileInfos = this._getFileInfos(this.files(), intoDir, templateVariables);

  this._ignoreUpdateFiles();

  return Promise.filter(fileInfos, isValidFile).
    then(finishProcessingForUninstall);
};


/**
  @method mapFile
  @param {String} file
  @return {String}
*/
Blueprint.prototype.mapFile = function(file, locals) {
  var pattern, i;
  var fileMap = locals.fileMap || { __name__: locals.dasherizedModuleName };
  file = Blueprint.renamedFiles[file] || file;
  for (i in fileMap) {
    pattern = new RegExp(i, 'g');
    file = file.replace(pattern, fileMap[i]);
  }
  return file;
};

/**
  Looks for a __root__ token in the files folder. Must be present for
  the blueprint to support addon tokens. The `server`, `blueprints`, and `test`

  @private
  @method supportsAddon
  @return {Boolean}
*/
Blueprint.prototype.supportsAddon = function() {
  return this.files().join().match(/__root__/);
};

/**
  @private
  @method _generateFileMapVariables
  @param {Object} options
  @return {Object}
*/
Blueprint.prototype._generateFileMapVariables = function(moduleName, locals, options) {
  var originBlueprintName = options.originBlueprintName || this.name;
  var podModulePrefix = this.project.config().podModulePrefix || '';
  var podPath = podModulePrefix.substr(podModulePrefix.lastIndexOf('/') + 1);
  var inAddon = this.project.isEmberCLIAddon() || !!options.inRepoAddon;
  var inDummy = this.project.isEmberCLIAddon() ? options.dummy : false;

  return {
    pod: this.pod,
    podPath: podPath,
    hasPathToken: this.hasPathToken,
    inAddon: inAddon,
    inRepoAddon: options.inRepoAddon,
    inDummy: inDummy,
    blueprintName: this.name,
    originBlueprintName: originBlueprintName,
    dasherizedModuleName: stringUtils.dasherize(moduleName),
    locals: locals
  };
};

/**
  @private
  @method _locals
  @param {Object} options
  @return {Object}
*/
Blueprint.prototype._locals = function(options) {
  var packageName = options.project.name();
  var moduleName = options.entity && options.entity.name || packageName;
  var sanitizedModuleName = moduleName.replace(/\//g, '-');
  var customLocals = this.locals(options);
  var fileMapVariables = this._generateFileMapVariables(moduleName, customLocals, options);
  var fileMap = this.generateFileMap(fileMapVariables);

  var standardLocals = {
    dasherizedPackageName: stringUtils.dasherize(packageName),
    classifiedPackageName: stringUtils.classify(packageName),
    dasherizedModuleName: stringUtils.dasherize(moduleName),
    classifiedModuleName: stringUtils.classify(sanitizedModuleName),
    camelizedModuleName: stringUtils.camelize(sanitizedModuleName),
    decamelizedModuleName: stringUtils.decamelize(sanitizedModuleName),
    fileMap: fileMap,
    hasPathToken: this.hasPathToken,
    targetFiles: options.targetFiles,
    rawArgs: options.rawArgs
  };

  return merge({}, standardLocals, customLocals);
};

/**
  Used to add a package to the project's `package.json`.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that a package that is required by a given blueprint is
  available.

  @method addPackageToProject
  @param {String} packageName
  @param {String} target
  @return {Promise}
*/
Blueprint.prototype.addPackageToProject = function(packageName, target) {
  var packageObject = {name: packageName};

  if (target) {
    packageObject.target = target;
  }

  return this.addPackagesToProject([packageObject]);
};

/**
  Used to add multiple packages to the project's `package.json`.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that a package that is required by a given blueprint is
  available.

  Expects each array item to be an object with a `name`.  Each object
  may optionally have a `target` to specify a specific version.

  @method addPackagesToProject
  @param {Array} packages
  @return {Promise}
*/
Blueprint.prototype.addPackagesToProject = function(packages) {
  var task = this.taskFor('npm-install');
  var installText = (packages.length > 1) ? 'install packages' : 'install package';
  var packageNames = [];
  var packageArray = [];

  for (var i = 0; i < packages.length; i++) {
    packageNames.push(packages[i].name);

    var packageNameAndVersion = packages[i].name;

    if (packages[i].target) {
      packageNameAndVersion += '@' + packages[i].target;
    }

    packageArray.push(packageNameAndVersion);
  }

  this._writeStatusToUI(chalk.green, installText, packageNames.join(', '));

  return task.run({
    'save-dev': true,
    verbose: false,
    packages: packageArray
  });
};

/**
  Used to remove a package from the project's `package.json`.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that any package conflicts can be resolved before the
  addon is used.

  @method removePackageFromProject
  @param {String} packageName
  @return {Promise}
*/
Blueprint.prototype.removePackageFromProject = function(packageName) {
  var packageObject = {name: packageName};

  return this.removePackagesFromProject([packageObject]);
};

/**
  Used to remove multiple packages from the project's `package.json`.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that any package conflicts can be resolved before the
  addon is used.

  Expects each array item to be an object with a `name` property.

  @method removePackagesFromProject
  @param {Array} packages
  @return {Promise}
*/
Blueprint.prototype.removePackagesFromProject = function(packages) {
  var task = this.taskFor('npm-uninstall');
  var installText = (packages.length > 1) ? 'uninstall packages' : 'uninstall package';
  var packageNames = [];
  var packageArray = [];

  for (var i = 0; i < packages.length; i++) {
    packageNames.push(packages[i].name);

    var packageNameAndVersion = packages[i].name;

    packageArray.push(packageNameAndVersion);
  }

  this._writeStatusToUI(chalk.green, installText, packageNames.join(', '));

  return task.run({
    'save-dev': true,
    verbose: false,
    packages: packageArray
  });
};

/**
  Used to add a package to the projects `bower.json`.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that a package that is required by a given blueprint is
  available.

  `localPackageName` and `target` may be thought of as equivalent
  to the key-value pairs in the `dependency` or `devDepencency`
  objects contained within a bower.json file.

  Examples:

  addBowerPackageToProject('jquery', '~1.11.1');
  addBowerPackageToProject('old_jquery', 'jquery#~1.9.1');
  addBowerPackageToProject('bootstrap-3', 'http://twitter.github.io/bootstrap/assets/bootstrap');

  @method addBowerPackageToProject
  @param {String} localPackageName
  @param {String} target
  @param {Object} installOptions
  @return {Promise}
*/
Blueprint.prototype.addBowerPackageToProject = function(localPackageName, target, installOptions) {
  var lpn = localPackageName;
  var tar = target;
  if (localPackageName.indexOf('#') >= 0) {
    if (arguments.length === 1) {
      var parts = localPackageName.split('#');
      lpn = parts[0];
      tar = parts[1];
      deprecateUI(this.ui)('passing ' + localPackageName +
        ' directly to `addBowerPackageToProject` will soon be unsupported. \n' +
        'You may want to replace this with ' +
        '`addBowerPackageToProject(\'' + lpn +'\', \'' + tar + '\')`', true);
    } else {
      deprecateUI(this.ui)('passing ' + localPackageName +
        ' directly to `addBowerPackageToProject` will soon be unsupported', true);
    }
  }
  var packageObject = bowEpParser.json2decomposed(lpn, tar);
  return this.addBowerPackagesToProject([packageObject], installOptions);
};

/**
  Used to add an array of packages to the projects `bower.json`.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that a package that is required by a given blueprint is
  available.

  Expects each array item to be an object with a `name`.  Each object
  may optionally have a `target` to specify a specific version.

  @method addBowerPackagesToProject
  @param {Array} packages
  @param {Object} installOptions
  @return {Promise}
*/
Blueprint.prototype.addBowerPackagesToProject = function(packages, installOptions) {
  var task = this.taskFor('bower-install');
  var installText = (packages.length > 1) ? 'install bower packages' : 'install bower package';
  var packageNames = [];
  var packageNamesAndVersions = packages.map(function (pkg) {
    pkg.source = pkg.source || pkg.name;
    packageNames.push(pkg.name);
    return pkg;
  }).map(bowEpParser.compose);

  this._writeStatusToUI(chalk.green, installText, packageNames.join(', '));

  return task.run({
    verbose: true,
    packages: packageNamesAndVersions,
    installOptions: installOptions
  });
};

/**
  Used to add an addon to the project's `package.json` and run it's
  `defaultBlueprint` if it provides one.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that a package that is required by a given blueprint is
  available.

  @method addAddonToProject
  @param {Object} options
  @return {Promise}
*/
Blueprint.prototype.addAddonToProject = function(options) {
  return this.addAddonsToProject({
    packages: [options],
    extraArgs: options.extraArgs || {},
    blueprintOptions: options.blueprintOptions || {}
  });
};

/**
  Used to add multiple addons to the project's `package.json` and run their
  `defaultBlueprint` if they provide one.

  Generally, this would be done from the `afterInstall` hook, to
  ensure that a package that is required by a given blueprint is
  available.

  @method addAddonsToProject
  @param {Object} options
  @return {Promise}
*/
Blueprint.prototype.addAddonsToProject = function (options) {
  var taskOptions = {
    packages: [],
    extraArgs: options.extraArgs || [],
    blueprintOptions: options.blueprintOptions || {}
  };

  var packages = options.packages;
  if (packages && packages.length) {
    taskOptions.packages = packages.map(function (pkg) {
      if (typeof pkg === 'string') {
        return pkg;
      }

      if (!pkg.name) {
        throw new SilentError('You must provide a package `name` to addAddonsToProject');
      }

      if (pkg.target) {
        pkg.name += '@' + pkg.target;
      }

      return pkg.name;
    });
  } else {
    throw new SilentError('You must provide package to addAddonsToProject');
  }

  var installText = (packages.length > 1) ? 'install addons' : 'install addon';
  this._writeStatusToUI(chalk.green, installText, taskOptions['packages'].join(', '));

  return this.taskFor('addon-install').run(taskOptions);
};

/**
  Used to retrieve a task with the given name. Passes the new task
  the standard information available (like `ui`, `analytics`, `project`, etc).

  @method taskFor
  @param dasherizedName
  @public
*/
Blueprint.prototype.taskFor = function(dasherizedName) {
  var Task = require('../tasks/' + dasherizedName);

  return new Task({
    ui: this.ui,
    project: this.project,
    analytics: this.analytics
  });
};

/*

  Inserts the given content into a file. If the `contentsToInsert` string is already
  present in the current contents, the file will not be changed unless `force` option
  is passed.

  If `options.before` is specified, `contentsToInsert` will be inserted before
  the first instance of that string.  If `options.after` is specified, the
  contents will be inserted after the first instance of that string.
  If the string specified by options.before or options.after is not in the file,
  no change will be made.

  If neither `options.before` nor `options.after` are present, `contentsToInsert`
  will be inserted at the end of the file.

  Example:
  ```
  // app/router.js
  Router.map(function(){
  });

  insertIntoFile('app/router.js',
                 '  this.route("admin");',
                 {after:'Router.map(function() {'+EOL});

  // new app/router.js
  Router.map(function(){
    this.route("admin");
  });
  ```

  @method insertIntoFile
  @param {String} pathRelativeToProjectRoot
  @param {String} contentsToInsert
  @param {Object} options
  @return {Promise}
*/
Blueprint.prototype.insertIntoFile = function(pathRelativeToProjectRoot, contentsToInsert, providedOptions) {
  var fullPath          = path.join(this.project.root, pathRelativeToProjectRoot);
  var originalContents  = '';

  if (existsSync(fullPath)) {
    originalContents = fs.readFileSync(fullPath, { encoding: 'utf8' });
  }

  var contentsToWrite   = originalContents;

  var options           = providedOptions || {};
  var alreadyPresent    = originalContents.indexOf(contentsToInsert) > -1;
  var insert            = !alreadyPresent;
  var insertBehavior    = 'end';

  if (options.before) { insertBehavior = 'before'; }
  if (options.after)  { insertBehavior = 'after'; }

  if (options.force) { insert = true; }

  if (insert) {
    if (insertBehavior === 'end') {
      contentsToWrite += contentsToInsert;
    } else {
      var contentMarker      = options[insertBehavior];
      var contentMarkerIndex = contentsToWrite.indexOf(contentMarker);

      if (contentMarkerIndex !== -1) {
        var insertIndex = contentMarkerIndex;
        if (insertBehavior === 'after') { insertIndex += contentMarker.length; }

        contentsToWrite = contentsToWrite.slice(0, insertIndex) +
                          contentsToInsert + EOL +
                          contentsToWrite.slice(insertIndex);
      }
    }
  }

  var returnValue = {
    path: fullPath,
    originalContents: originalContents,
    contents: contentsToWrite,
    inserted: false
  };

  if (contentsToWrite !== originalContents) {
    returnValue.inserted = true;

    return writeFile(fullPath, contentsToWrite)
      .then(function() {
        return returnValue;
      });
  } else {
    return Promise.resolve(returnValue);
  }
};

Blueprint.prototype.printBasicHelp = function(verbose) {
  var initialMargin = '      ';
  var output = initialMargin;
  if (this.overridden) {
    output += chalk.grey('(overridden) ' + this.name);
  } else {
    output += this.name;

    var options = this.anonymousOptions;

    if (options.length > 0) {
      output += ' ' + chalk.yellow(options.map(function(opt) {
        return '<' + opt + '>';
      }).join(' '));
    }

    options = this.availableOptions;

    if (options.length > 0) {
      output += ' ' + chalk.cyan('<options...>');
    }

    if (this.description) {
      output += EOL + initialMargin + '  ' + chalk.grey(this.description);
    }

    options.forEach(function(opt) {
      output += EOL + initialMargin + '  ' + chalk.cyan('--' + opt.name);

      if (opt.values) {
        output += chalk.cyan('=' + opt.values.join('|'));
      }

      if (opt.default !== undefined) {
        output += ' ' + chalk.cyan('(Default: ' + opt.default + ')');
      }

      if (opt.required) {
        output += ' ' + chalk.cyan('(Required)');
      }

      if (opt.aliases && opt.aliases.length) {
        output += EOL + initialMargin + '    ' + chalk.grey('aliases: ' + opt.aliases.map(function(a) {
          if (typeof a === 'string') {
            return '-' + a + (opt.type === Boolean ? '' : ' <value>');
          } else {
            var key = Object.keys(a)[0];
            return '-' + key + ' (--' + opt.name + '=' + a[key] + ')';
          }
        }).join(', '));
      }

      if (opt.description) {
        output += ' ' + opt.description;
      }
    });

    // I don't think we should support nulling out printDetailedHelp
    if (verbose && this.printDetailedHelp) {
      output += EOL + this.printDetailedHelp(options);
    }
  }

  return output;
};

Blueprint.prototype._getDetailedHelpPath = function() {
  if (!this.path) {
    // I don't think it is possible to reach this line.
    return null;
  }

  return path.join(this.path, './HELP.md');
};

Blueprint.prototype.printDetailedHelp = function() {
  var markdownColor = new MarkdownColor();
  var filePath = this._getDetailedHelpPath();

  if (existsSync(filePath)) {
    return markdownColor.renderFile(filePath, { indent: '        ' });
  }
  return '';
};

Blueprint.prototype.getJson = function(verbose) {
  var json = {};

  printableProperties.forEachWithProperty(function(key) {
    var value = this[key];
    if (key === 'availableOptions') {
      value = cloneDeep(value);
      value.forEach(function(option) {
        if (typeof option.type === 'function') {
          option.type = option.type.name;
        }
      });
    }
    json[key] = value;
  }, this);

  // I don't think we should support nulling out printDetailedHelp
  if (verbose && this.printDetailedHelp) {
    var detailedHelp = this.printDetailedHelp(this.availableOptions);
    if (detailedHelp) {
      json.detailedHelp = detailedHelp;
    }
  }

  return json;
};

/**
  Used to retrieve a blueprint with the given name.

  @method lookupBlueprint
  @param dasherizedName
  @public
*/
Blueprint.prototype.lookupBlueprint = function(dasherizedName) {
  var projectPaths = this.project ? this.project.blueprintLookupPaths() : [];

  return Blueprint.lookup(dasherizedName, {
    paths: projectPaths
  });
};

/**
  @static
  @method lookup
  @namespace Blueprint
  @param {String} [name]
  @param {Object} [options]
  @param {Array} [options.paths] Extra paths to search for blueprints
  @param {Object} [options.properties] Properties
  @return {Blueprint}
*/
Blueprint.lookup = function(name, options) {
  options = options || {};

  var lookupPaths = generateLookupPaths(options.paths);

  var lookupPath;
  var blueprintPath;

  for (var i = 0; lookupPath = lookupPaths[i]; i++) {
    blueprintPath = path.resolve(lookupPath, name);

    if (existsSync(blueprintPath)) {
      return Blueprint.load(blueprintPath);
    }
  }

  if (!options.ignoreMissing) {
    throw new SilentError('Unknown blueprint: ' + name);
  }
};

/**
  Loads a blueprint from given path.
  @static
  @method load
  @namespace Blueprint
  @param {String} blueprintPath
  @return {Blueprint} blueprint instance
*/
Blueprint.load = function(blueprintPath) {
  var constructorPath = path.resolve(blueprintPath, 'index.js');
  var blueprintModule;
  var Constructor = Blueprint;

  if (fs.lstatSync(blueprintPath).isDirectory()) {

    if (existsSync(constructorPath)) {
      blueprintModule = require(constructorPath);

      if (typeof blueprintModule === 'function') {
        Constructor = blueprintModule;
      } else {
        Constructor = Blueprint.extend(blueprintModule);
      }
    }

    return new Constructor(blueprintPath);
  }

  return;
};

/**
  @static
  @method list
  @namespace Blueprint
  @param {Object} [options]
  @param {Array} [options.paths] Extra paths to search for blueprints
  @return {Blueprint}
*/
Blueprint.list = function(options) {
  options = options || {};

  var lookupPaths = generateLookupPaths(options.paths);
  var seen = [];

  return lookupPaths.map(function(lookupPath) {
    var blueprints = dir(lookupPath);
    var packagePath = path.join(lookupPath, '../package.json');
    var source;

    if (existsSync(packagePath)) {
      source = require(packagePath).name;
    } else {
      source = path.basename(path.join(lookupPath, '..'));
    }

    blueprints = blueprints.map(function(blueprintPath) {
      var blueprint = Blueprint.load(blueprintPath);
      var name;

      if (blueprint) {
        name = blueprint.name;
        blueprint.overridden = contains(seen, name);
        seen.push(name);

        return blueprint;
      }

      return;
    });

    return {
      source: source,
      blueprints: compact(blueprints)
    };
  });
};

/**
  @static
  @property renameFiles
*/
Blueprint.renamedFiles = {
  'gitignore': '.gitignore'
};

/**
  @static
  @property ignoredFiles
*/
Blueprint.ignoredFiles = [
  '.DS_Store'
];

/**
  @static
  @property ignoredUpdateFiles
*/
Blueprint.ignoredUpdateFiles = [
  '.gitkeep',
  'app.css'
];

/**
  @static
  @property defaultLookupPaths
*/
Blueprint.defaultLookupPaths = function() {
  return [
    path.resolve(__dirname, '..', '..', 'blueprints')
  ];
};

/**
  @private
  @method prepareConfirm
  @param {FileInfo} info
  @return {Promise}
*/
function prepareConfirm(info) {
  return info.checkForConflict().then(function(resolution) {
    info.resolution = resolution;
    return info;
  });
}

/**
  @private
  @method markIdenticalToBeSkipped
  @param {FileInfo} info
*/
function markIdenticalToBeSkipped(info) {
  if (info.resolution === 'identical') {
    info.action = 'skip';
  }
}

/**
  @private
  @method markToBeRemoved
  @param {FileInfo} info
*/
function markToBeRemoved(info) {
  info.action = 'remove';
}

/**
  @private
  @method gatherConfirmationMessages
  @param {Array} collection
  @param {FileInfo} info
  @return {Array}
*/
function gatherConfirmationMessages(collection, info) {
  if (info.resolution === 'confirm') {
    collection.push(info.confirmOverwriteTask());
  }
  return collection;
}

/**
  @private
  @method isFile
  @param {FileInfo} info
  @return {Boolean}
*/
function isFile(info) {
  return stat(info.inputPath).invoke('isFile');
}

/**
  @private
  @method isIgnored
  @param {FileInfo} info
  @return {Boolean}
*/
function isIgnored(info) {
  var fn = info.inputPath;

  return any(Blueprint.ignoredFiles, function(ignoredFile) {
    return minimatch(fn, ignoredFile, { matchBase: true });
  });
}

/**
  Combines provided lookup paths with defaults and removes
  duplicates.

  @private
  @method generateLookupPaths
  @param {Array} lookupPaths
  @return {Array}
*/
function generateLookupPaths(lookupPaths) {
  lookupPaths = lookupPaths || [];
  lookupPaths = lookupPaths.concat(Blueprint.defaultLookupPaths());
  return uniq(lookupPaths);
}

/**
  Looks for a __path__ token in the files folder. Must be present for
  the blueprint to support pod tokens.

  @private
  @method hasPathToken
  @param {files} files
  @return {Boolean}
*/
function hasPathToken(files) {
  return files.join().match(/__path__/);
}

function inRepoAddonExists(name, root) {
  var addonPath = path.join(root, 'lib', name);
  return existsSync(addonPath);
}

function podDeprecations(config, ui){
  /*
  var podModulePrefix = config.podModulePrefix || '';
  var podPath = podModulePrefix.substr(podModulePrefix.lastIndexOf('/') + 1);
  // Disabled until we are ready to deprecate podModulePrefix
  deprecateUI(ui)('`podModulePrefix` is deprecated and will be removed from future versions of ember-cli.'+
    ' Please move existing pods from \'app/' + podPath + '/\' to \'app/\'.', config.podModulePrefix);
  */
  deprecateUI(ui)('`usePodsByDefault` is no longer supported in \'config/environment.js\','+
    ' use `usePods` in \'.ember-cli\' instead.', config.usePodsByDefault);
}

/**
  @private
  @method destPath
  @param {String} intoDir
  @param {String} file
  @return {String} new path
*/
function destPath(intoDir, file) {
  return path.join(intoDir, file);
}

/**
  @private
  @method isValidFile
  @param {Object} fileInfo
  @return {Promise}
*/
function isValidFile(fileInfo) {
  if (isIgnored(fileInfo)) {
    return Promise.resolve(false);
  } else {
    return isFile(fileInfo);
  }
}

/**
  @private
  @method isFilePath
  @param {Object} fileInfo
  @return {Promise}
*/
function isFilePath(fileInfo) {
  return fs.statSync(fileInfo.inputPath).isFile();
}

/**
 @private

 @method dir
 @returns {Array} list of files in the given directory or and empty array if no directory exists
*/
function dir(fullPath) {
  if (existsSync(fullPath)) {
    return fs.readdirSync(fullPath).map(function(fileName) {
      return path.join(fullPath, fileName);
    });
  } else {
    return [];
  }
}
