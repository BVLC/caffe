var hasKeys = require("./has-keys")

module.exports = extend

function extend(target) {
    var sources = [].slice.call(arguments, 1)

    sources.forEach(function (source) {
        if (!hasKeys(source)) {
            return
        }

        Object.keys(source).forEach(function (name) {
            target[name] = source[name]
        })
    })

    return target
}
