Router.map(function() {
  this.route('foos', function() {
    this.route('bar', function() {
      this.route('baz');
    });
  });
});
