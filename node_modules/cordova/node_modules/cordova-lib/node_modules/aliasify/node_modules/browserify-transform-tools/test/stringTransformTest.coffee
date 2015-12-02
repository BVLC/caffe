transformTools = require '../src/transformTools'
path = require 'path'
assert = require 'assert'

dummyJsFile = path.resolve __dirname, "../testFixtures/testWithConfig/dummy.js"
testDir = path.resolve __dirname, "../testFixtures/testWithConfig"

describe "transformTools string transforms", ->
    cwd = process.cwd()

    beforeEach ->
        process.chdir testDir

    after ->
        process.chdir cwd

    it "should transform generate a transform that operates on a string", (done) ->
        transform = transformTools.makeStringTransform "unblueify", (content, opts, cb) ->
            cb null, content.replace(/blue/g, opts.config.color);

        content = "this is a blue test"
        expectedContent = "this is a red test"

        transformTools.runTransform transform, dummyJsFile, {content}, (err, result) ->
            return done err if err
            assert.equal result, expectedContent
            done()

    it "should read content correctly", (done) ->
        transform = transformTools.makeStringTransform "xify", (content, opts, cb) ->
            cb null, content
        return transformTools.runTransform transform, dummyJsFile, {content:"lala"}, (err, result) ->
                assert.equal result, "lala"
                done()

    it "should return an error when string transform returns an error", (done) ->
        transform = transformTools.makeStringTransform "unblueify", (content, opts, cb) ->
            cb new Error("foo")

        transformTools.runTransform transform, dummyJsFile, {content:"lala"}, (err, result) ->
            assert.equal err?.message, "foo (while unblueify was processing /Users/jwalton/benbria/browserify-transform-tools/testFixtures/testWithConfig/dummy.js)"
            done()

    it "should return an error when string transform throws an error", (done) ->
        transform = transformTools.makeStringTransform "unblueify", (content, opts, cb) ->
            throw new Error("foo")

        transformTools.runTransform transform, dummyJsFile, {content:"lala"}, (err, result) ->
            assert.equal err?.message, "foo (while unblueify was processing /Users/jwalton/benbria/browserify-transform-tools/testFixtures/testWithConfig/dummy.js)"
            done()


    it "should allow manual configuration to override existing configuration", (done) ->
        transform = transformTools.makeStringTransform "xify", (content, opts, cb) ->
            if opts.config
                cb null, "x"
            else
                cb null, content

        configuredTransform = transform.configure {foo: "x"}

        transformTools.runTransform transform, dummyJsFile, {content:"lala"}, (err, result) ->
            assert.equal result, "lala"

            transformTools.runTransform configuredTransform, dummyJsFile, {content:"lala"}, (err, result) ->
                assert.equal result, "x"
                done()

    it "should allow configuration passed on construction", (done) ->
        transform = transformTools.makeStringTransform "xify", (content, opts, cb) ->
            if opts.config
                cb null, "x"
            else
                cb null, content

        return transformTools.runTransform transform, dummyJsFile, {content:"lala", config: {foo: "x"}},
            (err, result) ->
                assert.equal result, "x"
                done()

