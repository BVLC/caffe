
var app = new App();

app.use(HelloWorld, { "delimiter": "!" } );

app.init(function (err) {
  if (err) {
    console.log(err);
  }
});

app.hello("world");