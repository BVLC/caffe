# configstore [![Build Status](https://secure.travis-ci.org/yeoman/configstore.svg?branch=master)](http://travis-ci.org/yeoman/configstore)

> Easily load and persist config without having to think about where and how

Config is stored in a JSON file located in `$XDG_CONFIG_HOME` or `~/.config`.  
Example: `~/.config/configstore/some-id.json`


## Usage

```js
const Configstore = require('configstore');
const pkg = require('./package.json');

// Init a Configstore instance with an unique ID e.g.
// package name and optionally some default values
const conf = new Configstore(pkg.name, {foo: 'bar'});

conf.set('awesome', true);

console.log(conf.get('awesome'));
//=> true

console.log(conf.get('foo'));
//=> bar

conf.del('awesome');

console.log(conf.get('awesome'));
//=> undefined
```


## API

### Configstore(packageName, [defaults], [options])

Create a new Configstore instance `config`.

#### packageName

Type: `string`

Name of your package.

#### defaults

Type: `object`

Default content to init the config store with.

#### options

Type: `object`

##### globalConfigPath

Type: `boolean`  
Default: `false`

Store the config at `$CONFIG/package-name/config.json` instead of the default `$CONFIG/configstore/package-name.json`. This is not recommended as you might end up conflicting with other tools, rendering the "without having to think" idea moot.

### config.set(key, value)

Set an item.

### config.get(key)

Get an item.

### config.del(key)

Delete an item.

### config.clear()

Delete all items.

### config.all

Get all items as an object or replace the current config with an object:

```js
conf.all = {
	hello: 'world'
};
```

### config.size

Get the item count.

### config.path

Get the path to the config file. Can be used to show the user where the config file is located or even better open it for them.


## License

[BSD license](http://opensource.org/licenses/bsd-license.php)  
Copyright Google
