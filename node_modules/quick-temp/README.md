# node-quick-temp

Create and remove temporary directories. Useful for build tools, like Broccoli
plugins. Smart about naming, and placing them in `./tmp` if possible, so you
don't have to worry about this.

## Installation

```bash
npm install --save quick-temp
```

## Usage

```js
var quickTemp = require('quick-temp');
```

### Creating a temporary directory

To make a temporary and assign its path to `this.tmpDestDir`, call either one
of these:

```js
quickTemp.makeOrRemake(this, 'tmpDestDir');
// or
quickTemp.makeOrReuse(this, 'tmpDestDir');
```

If `this.tmpDestDir` already contains a path, `makeOrRemake` will remove it
first and then create a new directory, whereas `makeOrReuse` will be a no-op.

Both functions also return the path of the temporary directory.

An optional third argument lets you override the class-name component of the
temporary directory name:

```js
quickTemp.makeOrRemake(this, 'tmpDestDir', 'TreeMerger');
quickTemp.makeOrRemake(this, 'tmpDestDir', this.constructor.name); // default
```

### Removing a temporary directory

To remove a previously-created temporary directory and all its contents, call

```js
quickTemp.remove(this, 'tmpDestDir');
```

This will also assign `this.tmpDestDir = null`. If `this.tmpDestDir` is
already null or undefined, it will be a no-op.
