var os = require('os')
  , exec = require('child_process').exec

module.exports = {
  getconf: function(keyword, options, next) {

    if(typeof options == 'function') {
      next = options
      options = { default: '' }
    }

    exec('getconf '+keyword, function(error, stdout, stderr) {
      if(error !== null) {
        console.error('Error while getting '+keyword, error)
        return next(null, options.default)
      }

      stdout = parseInt(stdout)

      if(!isNaN(stdout)) {
        return next(null, stdout)
      }

      return next(null, options.default)
    })
  },
  cpu: function(next) {
    var self = this

    self.getconf('CLK_TCK', {default: 100}, function(err, clock_tick) {
      self.getconf('PAGESIZE', {default: 4096}, function(err, pagesize) {

        next(null, {
          clock_tick: clock_tick,
          uptime: os.uptime(),
          pagesize: pagesize
        })

       })
    })
  }
}
