'use strict';
var stringUtils = require('ember-cli-string-utils');
/**
  Utility to provide more friendly test names and descriptions
  allows for descriptive prefixes such as
  - Unit | Component | x-foo
  - Unit | Route | foo
  - Unit | Controller | foo
  - Acceptance | my cool feature
  etc...
*/

var SEPARATOR = ' | ';

module.exports = {
  /**
    Converts a string into friendly human form.

    ```javascript
    humanize("SomeCoolString") // 'some cool string'
    ```

    @method humanize
    @param {String} str The string to humanize.
    @return {String} the humanize string.
  */
  humanize: function(str) {
    var ret;
    ret = stringUtils.dasherize(str).replace(/[-]/g,' ');
    return ret;
  },

  /**
  @private
  @method _friendlyText
  @param {String} text Text that will be humanized.
  @param {String} testType The type of test (Unit, Acceptance, etc).
  @param {String} blueprintType The type of bluprint (Component, Mixin, etc).
  @return {String} A normalized text with type and blueprint prefix.
  */
  _friendlyText: function(text, testType, blueprintType) {
    var ret = testType + SEPARATOR;
    if (blueprintType) {
      ret += blueprintType + SEPARATOR;
    }
    ret += this.humanize(text);
    return ret;
  },

  /**
    Return a friendly test name with type prefix
    Unit | Components | x-foo

    ```javascript
    name("x-foo", "Unit", "Component") // Unit | Component | x foo
    ```

    @method name
    @param {String} name The name of the generated item.
    @param {String} testType The type of test (Unit, Acceptance, etc).
    @param {String} blueprintType The type of bluprint (Component, Mixin, etc).
    @return {String} A normalized name with type and blueprint prefix.
  */

  name: function(name, testType, blueprintType) {
    return this._friendlyText(name, testType, blueprintType);
  },

  /**
    Return a friendly test description

    ```javascript
    description("x-foo", "Unit", "Component") // Unit | Component | x foo
    ```

    @method description
    @param {String} description The description of the generated item.
    @param {String} testType The type of test (Unit, Acceptance, etc).
    @param {String} blueprintType The type of bluprint (Component, Mixin, etc).
    @return {String} A normalized description with type and blueprint prefix.
  */

  description: function(description, testType, blueprintType) {
    return this._friendlyText(description, testType, blueprintType);
  }

};
