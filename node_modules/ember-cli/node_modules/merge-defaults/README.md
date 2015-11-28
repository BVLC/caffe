# merge-defaults

Implements a deep version of `_.defaults`.

> **Important!**
>
> This module DOES NOT merge arrays or dates.


## Installation

```sh
$ npm install merge-defaults
```

## Usage

```javascript

var _ = require('lodash');

// Override basic `_.defaults`
_.defaults = require('merge-defaults');

// Or you can add it as a new method
_.mergeDefaults = require('merge-defaults');

```

## Why?

This module is a temporary solution, until lodash has something
similar in core that can be called as a single method.
In the mean time, this is a hack to make our code more readable.
i.e. I know what `_.defaults` means intuitively, but I have to look
up `_.partialRight` every time.

To get the latest status, see the [original issue in the lodash repo](https://github.com/lodash/lodash/issues/154#issuecomment-32140379).

I'll update this repo with install/version info if something comparable is
added to lodash core at some point.



## License

MIT &copy; Mike McNeil 2014
