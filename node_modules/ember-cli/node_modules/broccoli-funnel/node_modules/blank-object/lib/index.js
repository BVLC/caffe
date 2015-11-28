export default function BlankObject() {}
BlankObject.prototype = Object.create(null, {
  constructor: { value: undefined, enumerable: false, writable: true }
});
