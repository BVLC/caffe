
var flatiron = require('../../lib/flatiron'),
    app = module.exports = flatiron.app;

app.use(flatiron.plugins.resourceful, {
  root: __dirname, 
  engine: 'memory'
});
