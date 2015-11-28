Router.map(function() {
  this.route('foos', function() {
    this.route('index', { path: 'main' }, function() {});
  });
});
