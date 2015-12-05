Router.map(function() {
  this.route('foos', function() {
    this.route('bar');
  });
});
