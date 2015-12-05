var fs = require('fs')
var path = require('path')
var RSVP = require('rsvp')
var fixturify = require('fixturify')
var fixtureNode = require('broccoli-fixturify')
var Plugin = require('../index')
var chai = require('chai'), expect = chai.expect
var chaiAsPromised = require('chai-as-promised')
chai.use(chaiAsPromised)

var Builder_0_16_3 = require('./dependencies/broccoli-0.16.3').Builder


AnnotatingPlugin.prototype = Object.create(Plugin.prototype)
AnnotatingPlugin.prototype.constructor = AnnotatingPlugin
function AnnotatingPlugin() {
  Plugin.apply(this, arguments)
}
AnnotatingPlugin.prototype.build = function() {
  for (var i = 0; i < this.inputPaths.length; i++) {
    var files = fs.readdirSync(this.inputPaths[i])
    for (var j = 0; j < files.length; j++) {
      var content = fs.readFileSync(path.join(this.inputPaths[i], files[j]))
      content += ' - from input node #' + i
      fs.writeFileSync(path.join(this.outputPath, files[j]), content)
    }
  }
}

NoopPlugin.prototype = Object.create(Plugin.prototype)
NoopPlugin.prototype.constructor = NoopPlugin
function NoopPlugin() {
  Plugin.apply(this, arguments)
}
NoopPlugin.prototype.build = function() {}

var build = function(builder) {
  return RSVP.Promise.resolve()
    .then(function() {
      return builder.build()
    })
    .then(function(hash) {
      return fixturify.readSync(hash.directory)
    })
    .finally(function() {
      return builder.cleanup()
    })
}


describe('integration test', function(){
  var node1, node2

  beforeEach(function() {
    node1 = fixtureNode({ 'foo.txt': 'foo contents' })
    node2 = fixtureNode({ 'bar.txt': 'bar contents' })
  })

  describe('Broccoli with .read API', function(){
    it('works without errors', function(){
      return expect(build(new Builder_0_16_3(new AnnotatingPlugin([node1, node2]))))
        .to.eventually.deep.equal({
          'foo.txt': 'foo contents - from input node #0',
          'bar.txt': 'bar contents - from input node #1'
        })
    })

    it('sets description', function() {
      var node = new NoopPlugin([], {
        name: 'SomePlugin',
        annotation: 'some annotation'
      })
      var builder = new Builder_0_16_3(node)
      return builder.build()
        .then(function(hash) {
          return expect(hash.graph.toJSON().description).to.equal('SomePlugin: some annotation')
        })
        .finally(function() { builder.cleanup() })
    })

    it('handles readCompat initialization errors', function() {
      var node = new AnnotatingPlugin([])
      var initializeReadCompatCalls = 0
      node._initializeReadCompat = function() { // stub
        throw new Error('someError ' + (++initializeReadCompatCalls))
      }
      var builder = new Builder_0_16_3(node)
      return RSVP.Promise.resolve()
        .then(function() {
          return expect(builder.build()).to.be.rejectedWith(Error, 'someError 1')
        })
        .then(function() {
          return expect(builder.build()).to.be.rejectedWith(Error, 'someError 1')
        })
    })

    describe('persistent output', function() {
      BuildOnce.prototype = Object.create(Plugin.prototype)
      BuildOnce.prototype.constructor = BuildOnce
      function BuildOnce(options) {
        Plugin.call(this, [], options)
      }

      BuildOnce.prototype.build = function() {
        if (!(this.builtOnce)) {
          this.builtOnce = true
          fs.writeFileSync(path.join(this.outputPath, 'foo.txt'), 'test')
        }
      }

      function isPersistent(options) {
        var builder = new Builder_0_16_3(new BuildOnce(options))
        function buildAndCheckExistence() {
          return builder.build()
            .then(function(hash) {
              return fs.existsSync(path.join(hash.directory, 'foo.txt'))
            })
        }
        return expect(buildAndCheckExistence()).to.eventually.equal(true)
          .then(buildAndCheckExistence)
          .finally(function() { builder.cleanup() })
      }

      it('is not persistent by default', function() {
        return expect(isPersistent({})).to.eventually.equal(false)
      })

      it('is not persistent when persistentOutput is false', function() {
        return expect(isPersistent({ persistentOutput: false })).to.eventually.equal(false)
      })

      it('is persistent when persistentOutput is true', function() {
        return expect(isPersistent({ persistentOutput: true })).to.eventually.equal(true)
      })
    })
  })
})


describe('unit tests', function() {
  it('toString', function() {
    expect(new NoopPlugin([]) + '').to.equal('[NoopPlugin]')
    expect(new NoopPlugin([], { name: 'FooPlugin' }) + '').to.equal('[FooPlugin]')
    expect(new NoopPlugin([], { annotation: 'some note' }) + '').to.equal('[NoopPlugin: some note]')
  })

  describe('usage errors', function() {
    it('requires the base constructor to be called (super)', function() {
      TestPlugin.prototype = Object.create(Plugin.prototype)
      TestPlugin.prototype.constructor = TestPlugin
      function TestPlugin() { /* no Plugin.apply(this, arguments) here */ }
      TestPlugin.prototype.build = function() {}

      return expect(build(new Builder_0_16_3(new TestPlugin)))
        .to.be.rejectedWith(Error, /must call the superclass constructor/)
    })

    it('disallows overriding read, cleanup, and rebuild', function() {
      var prohibitedNames = ['read', 'rebuild', 'cleanup']
      for (var i = 0; i < prohibitedNames.length; i++) {
        var BadPlugin = function BadPlugin() {
          Plugin.apply(this, arguments)
        }
        BadPlugin.prototype = Object.create(Plugin.prototype)
        BadPlugin.prototype.constructor = BadPlugin
        BadPlugin.prototype.build = function() {}
        BadPlugin.prototype[prohibitedNames[i]] = function() {}

        expect(function() { new BadPlugin([]) })
          .to.throw(/For compatibility, plugins must not define/)
      }
    })

    it('checks that the inputNodes argument is an array', function() {
      expect(function() { new AnnotatingPlugin('notAnArray') })
        .to.throw(/Expected an array/)
    })

    it('provides a helpful error message on missing `new`', function() {
      expect(function() { AnnotatingPlugin([]) })
        .to.throw(/Missing `new`/)
    })
  })
})
