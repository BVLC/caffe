# fs-monitor-stack

Example (for broccoli')

```js
this.builder = new broccoli.Builder(this.tree);

var builder = this;

this.builder.on('start', function(node) {
  builder.monitor = new FSMonitor();
});

this.builder.on('nodeStart', function(node) {
  var metric = new NodeMetric(node);
  builder.monitor.push(metric);
});

this.builder.on('nodeEnd', function(node) {
  builder.monitor.pop();
});

this.builder.on('end', function(node) {
  console.log('endBuild');
  console.log(builder.monitor.totalStats());
});
```
