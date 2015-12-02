var broadway = require('../../'),
    app = new broadway.App();

// Passes the second argument to `helloworld.attach`.
app.use(require("./plugins/helloworld"), { "delimiter": "!" } );
app.use(broadway.plugins.log, {
  logAll: true
});

app.init(function (err) {
  if (err) {
    console.log(err);
  }
});

app.hello("world");
app.emit('world:hello', { meta: 'is here' });
