var Node = require("./node");

var Anonymous = function (value, index, currentFileInfo, mapLines, rulesetLike, referenced) {
    this.value = value;
    this.index = index;
    this.mapLines = mapLines;
    this.currentFileInfo = currentFileInfo;
    this.rulesetLike = (typeof rulesetLike === 'undefined') ? false : rulesetLike;
    this.isReferenced = referenced || false;
};
Anonymous.prototype = new Node();
Anonymous.prototype.type = "Anonymous";
Anonymous.prototype.eval = function () {
    return new Anonymous(this.value, this.index, this.currentFileInfo, this.mapLines, this.rulesetLike, this.isReferenced);
};
Anonymous.prototype.compare = function (other) {
    return other.toCSS && this.toCSS() === other.toCSS() ? 0 : undefined;
};
Anonymous.prototype.isRulesetLike = function() {
    return this.rulesetLike;
};
Anonymous.prototype.genCSS = function (context, output) {
    output.add(this.value, this.currentFileInfo, this.index, this.mapLines);
};
Anonymous.prototype.markReferenced = function () {
    this.isReferenced = true;
};
Anonymous.prototype.getIsReferenced = function () {
    return !this.currentFileInfo || !this.currentFileInfo.reference || this.isReferenced;
};

module.exports = Anonymous;
