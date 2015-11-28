'use strict';
module.exports = function toFastProperties(obj) {
	/*jshint -W027*/
	function f() {}
	f.prototype = obj;
	new f();
	return;
	eval(obj);
};
