# Functions for loading a transform's configuration.
#
# More details are available in [the transform configuration documentation](https://github.com/benbria/browserify-transform-tools/wiki/Transform-Configuration).
#
# ## Config Object
#
# The config object returned has the following properties:
# * `configData.config` - The configuration for the transform.
# * `configData.configDir` - The directory the configuration was loaded from; the directory which
#   contains package.json if that's where the config came from, or the directory which contains
#   the file specified in package.json.  This is handy for resolving relative paths.  Note thate
#   this field may be null if the configuration is overridden via the `configure()` function.
# * `configData.configFile` - The file the configuration was loaded from.  Note thate
#   this field may be null if the configuration is overridden via the `configure()` function.
# * `configData.cached` - Since a transform is run once for each file in a project, configuration
#   data is cached using the location of the package.json file as the key.  If this value is true,
#   it means that data was loaded from the cache.
# * `configData.appliesTo` - The `appliesTo` from the configuration, if one was present.
#
# ## Common Configuration
#
# All modules that rely on browserify-transform-tools can contain a configuration item called
# 'appliesTo'.  This will configure what files the transform will be applied to.
#
# `appliesTo` should include exactly one of the following:
#
# * `appliesTo.includeExtensions` - A list of extensions to process.  If this option is not
#   specified, then all extensions will be processed.  If this option is specified, then
#   any file with an extension not in this list will skipped.
#
# * `appliesTo.excludeExtensions` - A list of extensions which will not be processed.  e.g.
#   "['.coffee', '.jade']"
#
# * `appliesTo.files` - A list of paths, relative to the configuration file, of files which
#   should be transformed.  Only these files will be transformed.  This is handy for transforms
#   like [includify](https://github.com/benbria/includify) which you typically only apply to one
#   or two files in your project; defining this will typically speed up your bundle time, since
#   you no longer are running the transform on all the files in your project.
#
# * `appliesTo.regex` - A regex or a list of regexes.  If any regex matches the full path of the
#   file, then the file will be processed, otherwise not.
#
# The `appliesTo` key will be stripped from the configuration before being passed to your transform,
# although it is available in the `configData` if you need it for some reason.  Note that
# `appliesTo` will override the `includeExtensions` and `excludeExtensions` provided to any of the
# `make*Transform()` functions.
#

path    = require 'path'
fs      = require 'fs'
parentDir = require './parentDir'

# This is a cache where keys are directory names, and values are the closest ancestor directory
# that contains a package.json
packageJsonCache = {}

findPackageJson = (dirname, done) ->
    answer = packageJsonCache[dirname]
    if answer
        process.nextTick ->
            done null, answer
    else
        parentDir.parentDir dirname, 'package.json', (err, packageDir) ->
            return done err if err
            if packageDir
                packageFile = path.join(packageDir, 'package.json')
            else
                packageFile = null
            packageJsonCache[dirname] = packageFile
            done null, packageFile

findPackageJsonSync = (dirname) ->
    answer = packageJsonCache[dirname]
    if !answer
        packageDir = parentDir.parentDirSync dirname, 'package.json'
        if packageDir
            packageFile = path.join(packageDir, 'package.json')
        else
            packageFile = null
        packageJsonCache[dirname] = packageFile
        answer = packageFile
    return answer

# Cache for transform configuration.
configCache = {}

getConfigFromCache = (transformName, packageFile) ->
    cacheKey = "#{transformName}:#{packageFile}"
    return if configCache[cacheKey]? then configCache[cacheKey] else null

storeConfigInCache = (transformName, packageFile, configData) ->
    cacheKey = "#{transformName}:#{packageFile}"

    # Copy the config data, so we can set `cached` to true without affecting the object passed in.
    cachedConfigData = {}
    for key, value of configData
        cachedConfigData[key] = value
    cachedConfigData.cached = true

    configCache[cacheKey] = cachedConfigData

loadJsonAsync = (filename, done) ->
    fs.readFile filename, "utf-8", (err, content) ->
        return done err if err
        try
            done null, JSON.parse(content)
        catch err
            done err

# Load external configuration from a js or JSON file.
# * `packageFile` is the package.json file which references the external configuration.
# * `relativeConfigFile` is a file name relative to the package file directory.
loadExternalConfig = (packageFile, relativeConfigFile) ->
    # Load from an external file
    packageDir = path.dirname packageFile
    configFile = path.resolve packageDir, relativeConfigFile
    configDir = path.dirname configFile
    config = require configFile
    return {config, configDir, configFile, packageFile, cached: false}

# Process config found in package.json.  Store the config in the cache.
processConfig = (transformName, packageFile, config) ->
    # Found some configuration
    if typeof config is "string"
        # Load from an external file
        configData = loadExternalConfig packageFile, config

    else
        configFile = packageFile
        configDir = path.dirname packageFile
        configData = {config, configDir, configFile, packageFile, cached: false}

    if configData.config.appliesTo
        configData.appliesTo = configData.config.appliesTo
        delete configData.config.appliesTo

    storeConfigInCache transformName, packageFile, configData

    return configData


# Load configuration for a transform.
#
exports.loadTransformConfig = (transformName, file, options, done) ->
    if !done?
        done = options
        options = {}

    if options.fromSourceFileDir
        dir = path.dirname file
    else
        dir = process.cwd()

    findConfig = (dirname) ->
        findPackageJson dirname, (err, packageFile) ->
            return done err if err

            if !packageFile?
                # Couldn't find configuration
                done null, null
            else
                configData = getConfigFromCache transformName, packageFile
                if configData
                    done null, configData
                else
                    loadJsonAsync packageFile, (err, pkg) ->
                        return done err if err

                        config = pkg[transformName]
                        packageDir = path.dirname packageFile

                        if !config?
                            if !options.fromSourceFileDir
                                # Stop here.
                                done null, null
                            else
                                # Didn't find the config in the package file.  Try the parent dir.
                                parent = path.resolve packageDir, ".."
                                if parent == packageDir
                                    # Hit the root - we're done
                                    done null, null
                                else
                                    findConfig parent

                        else
                            # Found some configuration
                            try
                                configData = processConfig transformName, packageFile, config
                                done null, configData
                            catch err
                                done err

    findConfig dir

# Synchronous version of `loadTransformConfig()`.  Returns `{config, configDir}`.
exports.loadTransformConfigSync = (transformName, file, options={}) ->
    configData = null

    if options.fromSourceFileDir
        dirname = path.dirname file
    else
        dirname = process.cwd()

    done = false
    while !done
        packageFile = findPackageJsonSync dirname

        if !packageFile?
            # Couldn't find configuration
            configData = null
            done = true

        else
            configData = getConfigFromCache transformName, packageFile

            if configData
                done = true
            else
                pkg = require packageFile
                config = pkg[transformName]
                packageDir = path.dirname packageFile

                if !config?
                    if !options.fromSourceFileDir
                        # Stop here
                        done = true
                    else
                        # Didn't find the config in the package file.  Try the parent dir.
                        dirname = path.resolve packageDir, ".."
                        if dirname == packageDir
                            # Hit the root - we're done
                            done = true
                else
                    # Found some configuration
                    configData = processConfig transformName, packageFile, config
                    done = true

    return configData

exports.clearConfigCache = ->
    packageJsonCache = {}
    configCache = {}

