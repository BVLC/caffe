# [dep-graph](http://github.com/TrevorBurnham/dep-graph)

_ = require 'underscore'

class DepGraph
  constructor: ->
    # The internal representation of the dependency graph in the format
    # `id: [ids]`, indicating only *direct* dependencies.
    @map = {}

  # Add a direct dependency. Returns `false` if that dependency is a duplicate.
  add: (id, depId) ->
    @map[id] ?= []
    return false if depId in @map[id]
    @map[id].push depId
    @map[id]

  # Generate a list of all dependencies (direct and indirect) for the given id,
  # in logical order with no duplicates.
  getChain: (id) ->
    # First, get a list of all dependencies (unordered)
    deps = @descendantsOf id

    # Second, order them (using the Tarjan algorithm)
    chain = []
    visited = {}
    visit = (node) =>
      return if visited[node] or node is id
      visited[node] = true
      visit parent for parent in @parentsOf(node) when parent in deps
      chain.unshift node

    for leafNode in _.intersection(deps, @leafNodes()).reverse()
      visit leafNode

    chain

  leafNodes: ->
    allNodes = _.uniq _.flatten _.values @map
    node for node in allNodes when !@map[node]?.length

  parentsOf: (child) ->
    node for node in _.keys(@map) when child in @map[node]

  descendantsOf: (parent, descendants = [], branch = []) ->
    descendants.push parent
    branch.push parent
    for child in @map[parent] ? []
      if child in branch                # cycle
        throw new Error("Cyclic dependency from #{parent} to #{child}")
      continue if child in descendants  # duplicate
      @descendantsOf child, descendants, branch.slice(0)
    descendants[1..]

# Export the class in Node, make it global in the browser.
if module?.exports?
  module.exports = DepGraph
else
  @DepGraph = DepGraph
