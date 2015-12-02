path = require 'path'

# Extensions for all the various types of files that can get compiled to js.
JS_EXTENSIONS = [
    ".js",
    ".coffee", ".coffee.md", ".litcoffee", # via coffeeify
    "._js", "._coffee", # Streamline.js
    ".jsx" # React
]

isArray = (obj) -> Object.prototype.toString.call( obj ) == '[object Array]'

endsWith = (str, suffix) ->
    return str.indexOf(suffix, str.length - suffix.length) != -1

# Returns true if the given file should not be procesed, given the specified options and configData.
module.exports = (file, configData={}, options={}) ->
    file = path.resolve file
    skip = false

    appliesTo = configData.appliesTo

    # If there's no appliesTo, then use options.
    if !appliesTo? or (
        !appliesTo.includeExtensions? and
        !appliesTo.excludeExtensions? and
        !appliesTo.regex? and
        !appliesTo.files?
    )
        appliesTo = options

    includeExtensions = appliesTo?.includeExtensions
    if appliesTo?.jsFilesOnly and !includeExtensions then includeExtensions = JS_EXTENSIONS

    if appliesTo.regex?
        regexes = appliesTo.regex
        includeThisFile = false
        if !isArray(regexes) then regexes = [regexes]
        for regex in regexes
            if !regex.test then regex = new RegExp(regex)
            if regex.test file
                includeThisFile = true
                break

        if !includeThisFile then skip = true

    else if appliesTo.files?
        includeThisFile = false
        for fileToTest in appliesTo.files
            fileToTest = path.resolve configData.configDir, fileToTest
            if fileToTest == file
                includeThisFile = true
                break
        if !includeThisFile then skip = true

    else if appliesTo.excludeExtensions?
        for extension in appliesTo.excludeExtensions
            if endsWith(file, extension)
                skip = true
                break

    else if includeExtensions?
        includeThisFile = false
        for extension in includeExtensions
            if endsWith(file, extension)
                includeThisFile = true
                break
        if !includeThisFile then skip = true

    return skip

