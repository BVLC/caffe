transformTools = require '../src/transformTools'
path = require 'path'
assert = require 'assert'

dummyJsFile = path.resolve __dirname, "../testFixtures/testWithConfig/dummy.js"
testDir = path.resolve __dirname, "../testFixtures/testWithConfig"

describe "transformTools falafel transforms", ->
    cwd = process.cwd()

    beforeEach ->
        process.chdir testDir

    after ->
        process.chdir cwd

    it "should generate a transform that uses falafel", (done) ->
        transform = transformTools.makeFalafelTransform "unyellowify", (node, opts, cb) ->
            if node.type is "ArrayExpression"
                node.update "#{opts.config.color}(#{node.source()})"
            cb()

        content = "var x = [1,2,3];"
        expectedContent = "var x = green([1,2,3]);"
        transformTools.runTransform transform, dummyJsFile, {content}, (err, result) ->
            return done err if err
            assert.equal result, expectedContent
            done()

    it "should return an error when falafel transform returns an error", (done) ->
        transform = transformTools.makeFalafelTransform "unyellowify", (node, opts, cb) ->
            cb new Error("foo")

        transformTools.runTransform transform, dummyJsFile, {content:"lala"}, (err, result) ->
            assert.equal err?.message, "foo (while unyellowify was processing /Users/jwalton/benbria/browserify-transform-tools/testFixtures/testWithConfig/dummy.js)"
            done()

    it "should return an error when falafel transform throws an error", (done) ->
        transform = transformTools.makeFalafelTransform "unyellowify", (node, opts, cb) ->
            throw new Error("foo")

        transformTools.runTransform transform, dummyJsFile, {content:"lala"}, (err, result) ->
            assert.equal err?.message, "foo (while unyellowify was processing /Users/jwalton/benbria/browserify-transform-tools/testFixtures/testWithConfig/dummy.js)"
            done()

    it "should allow manual configuration to override existing configuration", (done) ->
        transform = transformTools.makeFalafelTransform "unyellowify", (node, opts, cb) ->
            if node.type is "ArrayExpression"
                node.update "#{opts.config.color}(#{node.source()})"
            cb()

        configuredTransform = transform.configure {color: "blue"}

        content = "var x = [1,2,3];"
        expectedContent = "var x = green([1,2,3]);"
        transformTools.runTransform transform, dummyJsFile, {content}, (err, result) ->
            return done err if err
            assert.equal result, expectedContent

            expectedContent = "var x = blue([1,2,3]);"
            transformTools.runTransform configuredTransform, dummyJsFile, {content}, (err, result) ->
                return done err if err
                assert.equal result, expectedContent
                done()

    it "should gracefully handle a syntax error", (done) ->
        transform = transformTools.makeFalafelTransform "identityify", (node, opts, cb) ->
            cb()

        content = """
            require('foo');
            require({;
            """
        transformTools.runTransform transform, dummyJsFile, {content}, (err, result) ->
            assert err != null, "Expected an error from runTransform"
            done()

    it "should not try to parse a json file if jsFilesOnly is true.", (done) ->
        dummyJsonFile = path.resolve __dirname, "../testFixtures/testWithConfig/dummy.json"

        transform = transformTools.makeFalafelTransform "identityify", {jsFilesOnly: true}, (args, opts, cb) ->
            cb()

        content = '{"foo": "bar"}'
        transformTools.runTransform transform, dummyJsonFile, {content}, (err, result) ->
            return done err if err
            assert.equal result, content
            done()



