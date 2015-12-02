path    = require 'path'
fs      = require 'fs'

# Temporarily don't use this while waiting for fix for https://github.com/thlorenz/find-parent-dir/issues/1
# exports.parentDir = require 'find-parent-dir'

# Find the first parent direcotry of 'dir' which contains a file named 'fileToFind'.
exports.parentDir = (dir, fileToFind, done) ->
    exists = fs.exists ? path.exists
    exists path.join(dir, fileToFind), (fileExists) ->
        if fileExists
            done null, dir
        else
            parent = path.resolve dir, ".."
            if parent == dir
                # Hit the root directory
                done null, null
            else
                # Recursive call to walk up the tree.
                exports.parentDir parent, fileToFind, done


# Find the first parent directory of `dir` which contains a file named `fileToFind`.
exports.parentDirSync = (dir, fileToFind) ->
    existsSync = fs.existsSync ? path.existsSync

    dirToCheck = path.resolve dir

    answer = null
    while true
        if existsSync path.join(dirToCheck, fileToFind)
            answer = dirToCheck
            break

        oldDirToCheck = dirToCheck
        dirToCheck = path.resolve dirToCheck, ".."
        if oldDirToCheck == dirToCheck
            # We've hit '/'.  We're done
            break

    return answer

