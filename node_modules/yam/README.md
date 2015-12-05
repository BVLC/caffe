yam
====

[![Build Status](https://travis-ci.org/twokul/yam.svg)](https://travis-ci.org/twokul/yam)

Dead simple lazy storage interface, useful to store cli or project settings. The file being parsed is expected in valid JSON format that can include comments.

#### Yam Constructor
```javascript
var Yam = require('yam');

//looks for a .test file in the current directory as well as your home directory
var yam = new Yam('test');

//customize where the file is located
var yam = new Yam('test', {
  primary: 'path/to/primary/location',
  secondary: 'path/to/secondary/location'
});
```

`.test` file example:
```javascript
{
  //comments are valid
  'foo': 'bar'
}
```

#### Get

```javascript
yam.get('foo'); // => 'bar'
```
#### GetAll

```javascript
yam.getAll();
```
