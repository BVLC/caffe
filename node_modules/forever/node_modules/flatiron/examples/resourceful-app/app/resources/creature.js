var resourceful = require('resourceful');

var Creature = module.exports = resourceful.define('creature', function () {
  this.string('diet');
  this.bool('vertebrate');
  this.array('belly');

  this.timestamps();
});

Creature.prototype.feed = function (food) {
  this.belly.push(food);
};
