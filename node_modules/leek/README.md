## Leek

[![Build Status](https://travis-ci.org/twokul/leek.svg)](https://travis-ci.org/twokul/leek)

Initial implementation was based on [Insight](https://github.com/yeoman/insight).

### Track Events:

```javascript
var Leek = require('./lib/leek'),
    leek = new Leek({
      trackingCode: 'xx-xxxxxxxx-x',
      globalName:   'my-nifty-package',
      version:      '1.0.1'
    });

leek.trackEvent({
  name:     'my-nifty-package',
  category: 'pony show',
  label:    'white',
  value:    'jumps'
});
```

### Track Errors:

```javascript
var Leek = require('./lib/leek'),
    leek = new Leek({
      trackingCode: 'xx-xxxxxxxx-x',
      name:         'my-nifty-package',
      version:      '1.0.1'
    });

try {
  throw new Error('ZOMG FIX ME')
} catch(e) {
  leek.trackError({
    description: e.message + ' ' + e.stack,
    isFatal: true
  });
}
```

### Track command hits:

```javascript
var Leek = require('./lib/leek'),
    leek = new Leek({
      trackingCode: 'xx-xxxxxxxx-x',
      name:         'my-nifty-package',
      version:      '1.0.1'
    });

leek.track({
  name:    'my-nifty-package show pony',
  message: 'pony is here'
});
```

### Disable Tracking:

#### Environment Variable

Setting the `DISABLE_LEEK` environment variable will disable tracking.

#### Alternative Tracking Endpoints

Passing in an object of adapter URLs will send the data to alternative tracking service.

```javascript
var Leek = require('./lib/leek');
var leek = new Leek({
  trackingCode: 'xx-xxxxxxxx-x',
  name:         'my-nifty-package',
  version:      '1.0.1',
  adapterUrls: {
    appview: 'http://example.com/collect-app',
    exception: 'http://example.com/collect-exception',
    timing: 'http://example.com/collect-timing',
    event: 'http://example.com/collect-event'
  }
});
```

#### Constructor Options

```javascript
var Leek = require('./lib/leek');
var leek = new Leek({
  trackingCode: 'xx-xxxxxxxx-x',
  name:         'my-nifty-package',
  version:      '1.0.1',
  silent:       true
});
```

### GA Setup
You are now able to set up alternative tracking services but to set up GA follow these steps:

* Set your new property up as a Mobile App
* Add the Tracking ID in your leek initialisation as seen above.
* To add the NodeJS as well as OS Version tracking go to the Admin -> Custom Definitions -> Custom Metrics
* Add a new Custom Dimenstion with the Name `OS` and Scope `Session`
* Add a second Custom Dimenstion with the Name `Node Version` and Scope `Session` as well
* Make sure they are both active and you have everything you need to get the tracking running properly
