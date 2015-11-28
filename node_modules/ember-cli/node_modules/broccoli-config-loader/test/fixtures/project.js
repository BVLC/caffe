var join = require('path').join,
    tmpRoot = join(process.cwd(), 'tmp'),
    tmp = require('tmp-sync'),
    fs = require('fs');

var Project = function(config) {
  this.root = tmp.in(tmpRoot);
  var configDir = join(this.root, 'config');
  fs.mkdirSync(configDir);
  this.writeConfig(config);
};

Project.prototype.writeConfig = function(config) {
  var configPath = this.absoluteConfigPath(),
      contents = 'module.exports = function() { return ' + JSON.stringify(config) + '; };';

  fs.writeFileSync(configPath + '.js', contents, { encoding: 'utf8' });
};

Project.prototype.configPath = function() {
  return join('config', 'environment');
};

Project.prototype.absoluteConfigPath = function() {
  return join(this.root, this.configPath());
};

Project.prototype.config = function(env) {
  return require(this.absoluteConfigPath())(env);
};

module.exports = Project;
