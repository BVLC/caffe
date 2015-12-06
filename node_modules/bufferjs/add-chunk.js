/*jshint strict:true node:true es5:true onevar:true laxcomma:true laxbreak:true eqeqeq:true immed:true latedef:true*/
(function () {
  "use strict";

  Buffer.prototype.__addchunk_index = 0;

  Buffer.prototype.addChunk = function (chunk) {
    var  len = Math.min(chunk.length, this.length - this.__addchunk_index);

    if (this.__addchunk_index === this.length) {
      //throw new Error("Buffer is full");
      return false;
    }

    chunk.copy(this, this.__addchunk_index, 0, len);

    this.__addchunk_index += len;

    if (len < chunk.length) {
      //remnant = new Buffer(chunk.length - len);
      //chunk.copy(remnant, 0, len, chunk.length);
      // return remnant;
      return chunk.slice(len, chunk.length);
    }

    if (this.__addchunk_index === this.length) {
      return true;
    }
  };
}());
