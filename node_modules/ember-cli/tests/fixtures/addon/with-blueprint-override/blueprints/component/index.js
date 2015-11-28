module.exports = {
  locals: function(options) {
    return this.lookupBlueprint('component').locals(options);
  }
};
