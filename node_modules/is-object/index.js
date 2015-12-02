module.exports = isObject

function isObject(x) {
    return typeof x === "object" && x !== null
}
