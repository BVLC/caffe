# Keymetrics Driver

![Keymetrics](https://keymetrics.io/assets/images/application-demo.png)
![Build Status](https://api.travis-ci.org/keymetrics/pmx.png?branch=master)

PMX is a module that allows you to create advanced interactions with Keymetrics.

It allows you to:

- **Expose Functions** remotely triggerable from Keymetrics
- **Expose Metrics** displayed in realtime and tracked over the time
- **Report Alerts** like exceptions or critical issues
- **Report Events** to inform about anything
- **Monitor network traffic** at the application level and display used ports
- Analyze HTTP latency

# Installation

Install PMX and add it to your package.json with:

```bash
$ npm install pmx --save
```

```javascript
var pmx = require('pmx').init({
  http          : true, // HTTP routes logging (default: true)
  ignore_routes : [/socket\.io/, /notFound/], // Ignore http routes with this pattern (Default: [])
  errors        : true, // Exceptions loggin (default: true)
  custom_probes : true, // Auto expose JS Loop Latency and HTTP req/s as custom metrics
  network       : true, // Network monitoring at the application level
  ports         : true,  // Shows which ports your app is listening on (default: false)
  alert_enabled : true  // Enable alert sub field in custom metrics   (default: false)
});
```

## Expose Functions: Trigger Functions remotely

Remotely trigger functions from Keymetrics. These metrics takes place in the main Keymetrics Dashboard page under the Custom Action section.

### Simple actions

Simple action allows to trigger a function from Keymetrics. The function takes a function as a parameter (reply here) and need to be called once the job is finished.

Example:

```javascript
var pmx = require('pmx');

pmx.action('db:clean', function(reply) {
  clean.db(function() {
    /**
     * reply() must be called at the end of the action
     */
     reply({success : true});
  });
});
```

### Scoped actions

Scoped Actions are advanced remote actions that can be also triggered from Keymetrics.

Two arguments are passed to the function, data (optionnal data sent from Keymetrics) and res that allows to emit log data and to end the scoped action.

Example:

```javascript
pmx.scopedAction('long running lsof', function(data, res) {
  var child = spawn('lsof', []);

  child.stdout.on('data', function(chunk) {
    chunk.toString().split('\n').forEach(function(line) {
      res.send(line); // This send log to Keymetrics to be saved (for tracking)
    });
  });

  child.stdout.on('end', function(chunk) {
    res.end('end'); // This end the scoped action
  });

  child.on('error', function(e) {
    res.error(e);  // This report an error to Keymetrics
  });

});
```

## Expose Metrics: Measure anything

Keymetrics allows you to expose any metrics from you code to the Keymetrics Dashboard, in realtime. These metrics takes place in the main Keymetrics Dashboard page under the Custom Metrics section.

4 helpers are available:

- **Simple metrics**: Values that can be read instantly
    - eg. Monitor variable value
- **Counter**: Things that increment or decrement
    - eg. Downloads being processed, user connected
- **Meter**: Things that are measured as events / interval
    - eg. Request per minute for a http server
- **Histogram**: Keeps a resevoir of statistically relevant values biased towards the last 5 minutes to explore their distribution
    - eg. Monitor the mean of execution of a query into database

### Metric: Simple value reporting

This allow to expose values that can be read instantly.

```javascript
var probe = pmx.probe();

// Here the value function will be called each second to get the value
var metric = probe.metric({
  name    : 'Realtime user',
  value   : function() {
    return Object.keys(users).length;
  }
});

// Here we are going to call valvar.set() to set the new value
var valvar = probe.metric({
  name    : 'Realtime Value'
});

valvar.set(23);
```

### Counter: Sequential value change

Things that increment or decrement.

```javascript
var probe = pmx.probe();

// The counter will start at 0
var counter = probe.counter({
  name : 'Current req processed'
});

http.createServer(function(req, res) {
  // Increment the counter, counter will eq 1
  counter.inc();
  req.on('end', function() {
    // Decrement the counter, counter will eq 0
    counter.dec();
  });
});
```

### Meter: Average calculated values

Things that are measured as events / interval.

```javascript
var probe = pmx.probe();

var meter = probe.meter({
  name      : 'req/sec',
  samples   : 1,
  timeframe : 60
});

http.createServer(function(req, res) {
  meter.mark();
  res.end({success:true});
});
```

#### Options

**samples** option is the rate unit. Defaults to **1** sec.
**timeframe** option is the timeframe over which events will be analyzed. Defaults to **60** sec.

### Histogram

Keeps a resevoir of statistically relevant values biased towards the last 5 minutes to explore their distribution.

```javascript
var probe = pmx.probe();

var histogram = probe.histogram({
  name        : 'latency',
  measurement : 'mean'
});

var latency = 0;

setInterval(function() {
  latency = Math.round(Math.random() * 100);
  histogram.update(latency);
}, 100);
```

### Common Custom Metrics options

- `name` : The probe name as is will be displayed on the **Keymetrics** dashboard
- `agg_type` : This param is optionnal, it can be `sum`, `max`, `min`, `avg` (default) or `none`. It will impact the way the probe data are aggregated within the **Keymetrics** backend. Use `none` if this is irrelevant (eg: constant or string value).
- `alert` : For `Meter` and `Counter` probes. This param is optionnal. Creates an alert object (see below).

### Alert System for Custom Metrics

This alert system can monitor a Probe value and launch an exception when hitting a particular value.

Example for a `cpu_usage` var:
```javascript
var metric = probe.metric({
  name  : 'CPU usage',
  value : function() {
    return cpu_usage;
  },
  alert : {
    mode  : 'threshold',
    value : 95,
    msg   : 'Detected over 95% CPU usage', // optional
    func  : function() { //optional
      console.error('Detected over 95% CPU usage');
    },
    cmp   : "<" // optional
  }
});
```

####Options:

- `mode` : `threshold`, `threshold-avg`.
- `value` : Value that will be used for the exception check.
- `msg` : String used for the exception.
- `func` :  **optional**. Function declenched when exception reached.
- `cmp` : **optional**. If current Probe value is not `<`, `>`, `=` to Threshold value the exception is launched. Can also be a function used for exception check taking 2 arguments and returning a bool.
- `interval` : **optional**, `threshold-avg` mode. Sample length for monitored value (180 seconds default).
- `timeout` : **optional**, `threshold-avg` mode. Time after which mean comparison starts (30 000 milliseconds default).

## Report Alerts: Errors / Uncaught Exceptions

By default once PM2 is linked to Keymetrics, you will be alerted of any uncaught exception.
These errors are accessible in the **Issue** tab of Keymetrics.

### Custom alert notification

If you need to alert about any critical errors you can do it programmatically:

```javascript
var pmx = require('pmx');

pmx.notify({ success : false });

pmx.notify('This is an error');

pmx.notify(new Error('This is an error'));
```

### Add Verbosity to an Alert: Express Error handler

When an uncaught exception is happening you can track from which routes it has been thrown.
To do that you have to attach the middleware `pmx.expressErrorHandler` at then end of your routes mounting:

```javascript
var pmx = require('pmx');

// All my routes
app.get('/' ...);
app.post(...);
// All my routes

// Here I attach the middleware to get more verbosity on exception thrown
app.use(pmx.expressErrorHandler());
```

## Emit Events

Emit events and get historical and statistics.
This is available in the **Events** page of Keymetrics.

```javascript
var pmx = require('pmx');

pmx.emit('user:register', {
  user : 'Alex registered',
  email : 'thorustor@gmail.com'
});
```

## Application level network traffic monitoring / Display used ports

You can monitor the network usage of a specific application by adding the option `network: true` when initializing PMX. If you enable the flag `ports: true` when you init pmx it will show which ports your app is listenting on.

These metrics will be shown in the Keymetrics Dashboard in the Custom Metrics section.

Example:

```
pmx.init({
  [...]
  network : true, // Allow application level network monitoring
  ports   : true  // Display ports used by the application
});
```

## HTTP latency analysis

Monitor routes, latency and codes. REST compliant.

```javascript
pmx.http(); // You must do this BEFORE any require('http')
```
Ignore some routes by passing a list of regular expressions.
```javascript
pmx.http({
  http          : true, // (Default: true)
  ignore_routes : [/socket\.io/, /notFound/] // Ignore http routes with this pattern (Default: [])
});
```
This can also be done via pmx.init()
```javascript
pmx.init({
  http          : true, // (Default: true)
  ignore_routes : [/socket\.io/, /notFound/] // Ignore http routes with this pattern (Default: [])
});
```

## License

MIT
