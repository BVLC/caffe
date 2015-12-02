DepGraph = require('../lib/dep-graph.js')
depGraph = new DepGraph

exports['Direct dependencies are chained in original order'] = (test) ->
  depGraph.add '0', '1'
  depGraph.add '0', '2'
  depGraph.add '0', '3'
  test.deepEqual depGraph.getChain('0'), ['1', '2', '3']
  test.done()

exports['Indirect dependencies are chained before their dependents'] = (test) ->
  depGraph.add '2', 'A'
  depGraph.add '2', 'B'
  test.deepEqual depGraph.getChain('0'), ['1', 'A', 'B', '2', '3']
  test.done()

exports['getChain can safely be called for unknown resources'] = (test) ->
  test.doesNotThrow -> depGraph.getChain('Z')
  test.deepEqual depGraph.getChain('Z'), []
  test.done()

exports['Cyclic dependencies are detected'] = (test) ->
  depGraph.add 'yin', 'yang'
  depGraph.add 'yang', 'yin'
  test.throws -> depGraph.getChain 'yin'
  test.throws -> depGraph.getChain 'yang'
  test.done()

exports['Arc direction is taken into account (issue #1)'] = (test) ->
  depGraph.add 'MAIN', 'One'
  depGraph.add 'MAIN', 'Three'
  depGraph.add 'One', 'Two'
  depGraph.add 'Two', 'Three'
  test.deepEqual depGraph.getChain('MAIN'), ['Three', 'Two', 'One']
  test.done()

exports['Dependency ordering is consistent (issue #2)'] = (test) ->
  depGraph.add 'Head', 'Neck'
  depGraph.add 'Head', 'Heart'
  depGraph.add 'Heart', 'Neck'
  depGraph.add 'Neck', 'Shoulders'
  test.deepEqual depGraph.getChain('Head'), ['Shoulders', 'Neck', 'Heart']
  test.done()

exports['Nodes with same dependencies do not depend on each other (issue #6)'] = (test) ->
  depGraph.add 'Java', 'JVM'
  depGraph.add 'JRuby', 'JVM'
  test.deepEqual depGraph.getChain('Java'), ['JVM']
  test.deepEqual depGraph.getChain('JRuby'), ['JVM']
  test.done()
