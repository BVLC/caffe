# Framework for building Falafel based transforms for Browserify.

path    = require 'path'
fs      = require 'fs'

through   = require 'through'
falafel   = require 'falafel'

loadConfig = require './loadConfig'
skipFile = require './skipFile'

exports.loadTransformConfig = loadConfig.loadTransformConfig
exports.loadTransformConfigSync = loadConfig.loadTransformConfigSync
exports.skipFile = skipFile

# TODO: Does this work on Windows?
isRootDir = (filename) -> filename == path.resolve(filename, '/')

merge = (a={}, b={}) ->
    answer = {}
    answer[key] = a[key] for key of a
    answer[key] = b[key] for key of b
    return answer


# Create a new Browserify transform which reads and returns a string.
#
# Browserify transforms work on streams.  This is all well and good, until you want to call
# a library like "falafel" which doesn't work with streams.
#
# Suppose you are writing a transform called "redify" which replaces all occurances of "blue"
# with "red":
#
#     options = {}
#     module.exports = makeStringTransform "redify", options, (contents, transformOptions, done) ->
#         done null, contents.replace(/blue/g, "red")
#
# Parameters:
# * `transformFn(contents, transformOptions, done)` - Function which is called to
#   do the transform.  `contents` are the contents of the file.  `transformOptions.file` is the
#   name of the file (as would be passed to a normal browserify transform.)
#   `transformOptions.configData` is the configuration data for the transform (see
#   `loadTransformConfig` below for details.)  `transformOptions.config` is a copy of
#   `transformOptions.configData.config` for convenience.  `done(err, transformed)` is a callback
#   which must be called, passing the a string with the transformed contents of the file.
# * `options.excludeExtensions` - A list of extensions which will not be processed.  e.g.
#   "['.coffee', '.jade']"
# * `options.includeExtensions` - A list of extensions to process.  If this options is not
#   specified, then all extensions will be processed.  If this option is specified, then
#   any file with an extension not in this list will skipped.
# * `options.jsFilesOnly` - If true (and if includeExtensions is not set) then this transform
#   will only operate on .js files, and on files which are commonly compiled to javascript
#   (.coffee, .litcoffee, .coffee.md, .jade, etc...)
#
exports.makeStringTransform = (transformName, options={}, transformFn) ->
    if !transformFn?
        transformFn = options
        options = {}

    transform = (file, config) ->
        configData = if transform.configData?
            transform.configData
        else
            loadConfig.loadTransformConfigSync transformName, file, options

        if config?
            configData ?= {config:{}}
            configData.config = merge configData.config, config

        if skipFile file, configData, options then return through()

        # Read the file contents into `content`
        content = ''
        write = (buf) -> content += buf

        # Called when we're done reading file contents
        end = ->
            handleError = (error) =>
                suffix = " (while #{transformName} was processing #{file})"
                if error instanceof Error and error.message
                    error.message += suffix
                else
                    error = new Error("#{error}#{suffix}")
                @emit 'error', error

            try
                transformOptions = {
                    file: file,
                    configData: configData,
                    config: configData?.config,
                    opts: configData?.config
                }
                transformFn content, transformOptions, (err, transformed) =>
                    return handleError err if err
                    @queue String(transformed)
                    @queue null
            catch err
                handleError err

        return through write, end

    # Called to manually pass configuration data to the transform.  Configuration passed in this
    # way will override configuration loaded from package.json.
    #
    # * `config` is the configuration data.
    # * `configOptions.configFile` is the file that configuration data was loaded from.  If this
    #   is specified and `configOptions.configDir` is not specified, then `configOptions.configDir`
    #   will be inferred from the configFile's path.
    # * `configOptions.configDir` is the directory the configuration was loaded from.  This is used
    #   by some transforms to resolve relative paths.
    #
    # Returns a new transform that uses the configuration:
    #
    #     myTransform = require('myTransform').configure(...)
    #
    transform.configure = (config, configOptions = {}) ->
        answer = exports.makeStringTransform transformName, options, transformFn
        answer.setConfig config, configOptions
        return answer

    # Similar to `configure()`, but modifies the transform instance it is called on.  This can
    # be used to set the default configuration for the transform.
    transform.setConfig = (config, configOptions = {}) ->
        configFile = configOptions.configFile or null
        configDir = configOptions.configDir or if configFile then path.dirname configFile else null

        if !config
            @configData = null
        else
            @configData = {
                config: config,
                configFile: configFile,
                configDir: configDir,
                cached: false
            }

            if config.appliesTo
                @configData.appliesTo = config.appliesTo
                delete config.appliesTo


        return this


    return transform


# Create a new Browserify transform based on [falafel](https://github.com/substack/node-falafel).
#
# Parameters:
# * `transformFn(node, transformOptions, done)` is called once for each falafel node.  transformFn
#   is free to update the falafel node directly; any value returned via `done(err)` is ignored.
# * `options.falafelOptions` are options to pass directly to Falafel.
# * `transformName`, `options.excludeExtensions`, `options.includeExtensions`, `options.jsFilesOnly`,
#   and `tranformOptions` are the same as for `makeStringTransform()`.
#
exports.makeFalafelTransform = (transformName, options={}, transformFn) ->
    if !transformFn?
        transformFn = options
        options = {}

    falafelOptions = options.falafelOptions ? {}

    transform = exports.makeStringTransform transformName, options, (content, transformOptions, done) ->
        transformErr = null
        pending = 1 # We'll decrement this to zero at the end to prevent premature call of `done`.
        transformed = null

        transformCb = (err) ->
            if err and !transformErr
                transformErr = err
                done err

            # Stop further processing if an error has occurred
            return if transformErr

            pending--
            if pending is 0
                done null, transformed

        transformed = falafel content, falafelOptions, (node) ->
            pending++
            try
                transformFn node, transformOptions, transformCb
            catch err
                transformCb err

        # call transformCb one more time to decrement pending to 0.
        transformCb transformErr, transformed

    # Called to manually pass configuration data to the transform.  Configuration passed in this
    # way will override configuration loaded from package.json.
    #
    # * `config` is the configuration data.
    # * `configOptions.configFile` is the file that configuration data was loaded from.  If this
    #   is specified and `configOptions.configDir` is not specified, then `configOptions.configDir`
    #   will be inferred from the configFile's path.
    # * `configOptions.configDir` is the directory the configuration was loaded from.  This is used
    #   by some transforms to resolve relative paths.
    #
    # Returns a new transform that uses the configuration:
    #
    #     myTransform = require('myTransform').configure(...)
    #
    transform.configure = (config, configOptions = {}) ->
        answer = exports.makeFalafelTransform transformName, options, transformFn
        answer.setConfig config, configOptions
        return answer

    return transform

# Create a new Browserify transform that modifies requires() calls.
#
# The resulting transform will call `transformFn(requireArgs, tranformOptions, cb)` for every
# requires in a file.  transformFn should call `cb(null, str)` with a string which will replace the
# entire `require` call.
#
# Exmaple:
#
#     makeRequireTransform "xify", (requireArgs, cb) ->
#         cb null, "require(x" + requireArgs[0] + ")"
#
# would transform calls like `require("foo")` into `require("xfoo")`.
#
# `transformName`, `options.excludeExtensions`, `options.includeExtensions`, `options.jsFilesOnly`,
# and `tranformOptions` are the same as for `makeStringTransform()`.
#
# By default, makeRequireTransform will attempt to evaluate each "require" parameters.
# makeRequireTransform can handle variabls `__filename`, `__dirname`, `path`, and `join` (where
# `join` is treated as `path.join`) as well as any basic JS expressions.  If the argument is
# too complicated to parse, then makeRequireTransform will return the source for the argument.
# You can disable parsing by passing `options.evaluateArguments` as false.
#
exports.makeRequireTransform = (transformName, options={}, transformFn) ->
    if !transformFn?
        transformFn = options
        options = {}

    evaluateArguments = options.evaluateArguments ? true

    transform = exports.makeFalafelTransform transformName, options, (node, transformOptions, done) ->
        if (node.type is 'CallExpression' and node.callee.type is 'Identifier' and
        node.callee.name is 'require')
            # Parse arguemnts to calls to `require`.
            if evaluateArguments
                # Based on https://github.com/ForbesLindesay/rfileify.
                dirname = path.dirname(transformOptions.file)
                varNames = ['__filename', '__dirname', 'path', 'join']
                vars = [transformOptions.file, dirname, path, path.join]

                args = node.arguments.map (arg) ->
                    t = "return #{arg.source()}"
                    try
                        return Function(varNames, t).apply(null, vars)
                    catch err
                        # Can't evaluate the arguemnts.  Return the raw source.
                        return arg.source()
            else
                args = (arg.source() for arg in node.arguments)

            transformFn args, transformOptions, (err, transformed) ->
                return done err if err
                if transformed? then node.update(transformed)
                done()
        else
            done()

    # Called to manually pass configuration data to the transform.  Configuration passed in this
    # way will override configuration loaded from package.json.
    #
    # * `config` is the configuration data.
    # * `configOptions.configFile` is the file that configuration data was loaded from.  If this
    #   is specified and `configOptions.configDir` is not specified, then `configOptions.configDir`
    #   will be inferred from the configFile's path.
    # * `configOptions.configDir` is the directory the configuration was loaded from.  This is used
    #   by some transforms to resolve relative paths.
    #
    # Returns a new transform that uses the configuration:
    #
    #     myTransform = require('myTransform').configure(...)
    #
    transform.configure = (config, configOptions = {}) ->
        answer = exports.makeRequireTransform transformName, options, transformFn
        answer.setConfig config, configOptions
        return answer

    return transform

# Runs a Browserify-style transform on the given file.
#
# * `transform` is the transform to run (i.e. a `fn(file)` which returns a through stream.)
# * `file` is the name of the file to run the transform on.
# * `options.content` is the content of the file.  If this option is not provided, the content
#   will be read from disk.
# * `options.config` is configuration to pass along to the transform.
# * `done(err, result)` will be called with the transformed input.
#
exports.runTransform = (transform, file, options={}, done) ->
    if !done?
        done = options
        options = {}

    doTransform = (content) ->
        data = ""
        err = null

        throughStream = if options.config?
            transform(file, options.config)
        else
            transform(file)

        throughStream.on "data", (d) ->
            data += d
        throughStream.on "end", ->
            if !err then done null, data
        throughStream.on "error", (e) ->
            err = e
            done err

        throughStream.write content
        throughStream.end()

    if options.content
        process.nextTick -> doTransform options.content
    else
        fs.readFile file, "utf-8", (err, content) ->
            return done err if err
            doTransform content
