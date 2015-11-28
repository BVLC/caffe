module.exports = Plugin
function Plugin(inputNodes, options) {
  if (!(this instanceof Plugin)) throw new Error('Missing `new` operator')
  if (!Array.isArray(inputNodes)) throw new Error('Expected an array of input nodes (input trees), got ' + inputNodes)

  this._baseConstructorCalled = true
  this._inputNodes = inputNodes

  // Remember current call stack (minus "Error" line)
  this._instantiationStack = (new Error).stack.replace(/[^\n]*\n/, '')

  options = options || {}
  if (options.name != null) {
    this._name = options.name
  } else if (this.constructor && this.constructor.name != null) {
    this._name = this.constructor.name
  } else {
    this._name = 'Plugin'
  }
  this._annotation = options.annotation
  this._persistentOutput = !!options.persistentOutput

  this._checkOverrides()
}

Plugin.prototype._checkOverrides = function() {
  if (typeof this.rebuild === 'function') {
    throw new Error('For compatibility, plugins must not define a plugin.rebuild() function')
  }
  if (this.read !== Plugin.prototype.read) {
    throw new Error('For compatibility, plugins must not define a plugin.read() function')
  }
  if (this.cleanup !== Plugin.prototype.cleanup) {
    throw new Error('For compatibility, plugins must not define a plugin.cleanup() function')
  }
}

// For future extensibility, we version the API using feature flags
Plugin.prototype.__broccoliFeatures__ = Object.freeze({
  persistentOutputFlag: true
})

// The Broccoli builder calls plugin.__broccoliGetInfo__
Plugin.prototype.__broccoliGetInfo__ = function(builderFeatures) {
  this._checkBuilderFeatures(builderFeatures)
  if (!this._baseConstructorCalled) throw new Error('Plugin subclasses must call the superclass constructor: Plugin.call(this, inputNodes)')

  return {
    inputNodes: this._inputNodes,
    setup: this._setup.bind(this),
    getCallbackObject: this.getCallbackObject.bind(this), // .build, indirectly
    instantiationStack: this._instantiationStack,
    name: this._name,
    annotation: this._annotation,
    persistentOutput: this._persistentOutput
  }
}

Plugin.prototype._checkBuilderFeatures = function(builderFeatures) {
  if (!builderFeatures || !builderFeatures.persistentOutputFlag) {
    // No builder in the wild implements less than these.
    throw new Error('Minimum builderFeatures required: { persistentOutputFlag: true }')
  }
}

Plugin.prototype._setup = function(builderFeatures, options) {
  this._checkBuilderFeatures(builderFeatures)
  this._builderFeatures = builderFeatures
  this.inputPaths = options.inputPaths
  this.outputPath = options.outputPath
  this.cachePath = options.cachePath
}

Plugin.prototype.toString = function() {
  return '[' + this._name
    + (this._annotation != null ? ': ' + this._annotation : '')
    + ']'
}

// Return obj on which the builder will call obj.build() repeatedly
//
// This indirection allows subclasses like broccoli-caching-writer to hook
// into calls from the builder, by returning { build: someFunction }
Plugin.prototype.getCallbackObject = function() {
  return this
}

Plugin.prototype.build = function() {
  throw new Error('Plugin subclasses must implement a .build() function')
}


// Compatibility code so plugins can run on old, .read-based Broccoli:

Plugin.prototype.read = function(readTree) {
  if (this._readCompat == null) {
    try {
      this._initializeReadCompat() // call this.__broccoliGetInfo__()
    } catch (err) {
      // Prevent trying to initialize again on next .read
      this._readCompat = false
      // Remember error so we can throw it on all subsequent .read calls
      this._readCompatError = err
    }
  }

  if (this._readCompatError != null) throw this._readCompatError

  return this._readCompat.read(readTree)
}

Plugin.prototype.cleanup = function() {
  if (this._readCompat) return this._readCompat.cleanup()
}

Plugin.prototype._initializeReadCompat = function() {
  var ReadCompat = require('./read_compat')
  this._readCompat = new ReadCompat(this)
}
