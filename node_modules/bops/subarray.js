module.exports = function(source, from, to) {
  return arguments.length === 2 ?
    source.slice(from) :
    source.slice(from, to)
}
