(function() {
  function vendorModule() {
    'use strict';

    return { 'default': self['<%= name %>'] };
  }

  define('<%= name %>', [], vendorModule);
})();
