# ember-router-generator Changelog

# 1.1.1

Fix bug so route options object is not generated if there is no valid
option.


# 1.1.0

Add support for `resetNamespace` when generating routes. For more info
see [#12](https://github.com/ember-cli/ember-router-generator/pull/12).

## 1.0.0

Removes support for `resource` routes. Now only `route` is supported.

See [#11](https://github.com/ember-cli/ember-router-generator/pull/11)
for more info.

### 0.4.0

Adds special handling for index routes. When running `ember g route
foo/index` the generated route will be `this.route('foo', function()
{})`.

If there is a route like `this.route('foo', function() {})` running
`ember d route foo/index` will modify it to `this.route('foo');` since we
are just removing the index.

For more info see [#10](https://github.com/ember-cli/ember-router-generator/pull/10).
