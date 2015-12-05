Router.map(function() {
  this.route('foos', function() {
    this.route('edit', { path: ':foo_id/edit' });
  });
});
