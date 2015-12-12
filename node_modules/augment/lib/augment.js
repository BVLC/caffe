Function.prototype.augment = function (body) {
    var base = this.prototype;
    var prototype = Object.create(base);
    body.apply(prototype, Array.from(arguments, 1).concat(base));
    if (!Object.ownPropertyOf(prototype, "constructor")) return prototype;
    var constructor = prototype.constructor;
    constructor.prototype = prototype;
    return constructor;
};

(function funct() {
    var bind = funct.bind;
    var bindable = Function.bindable = bind.bind(bind);
    var callable = Function.callable = bindable(funct.call);
    Object.ownPropertyOf = callable(funct.hasOwnProperty);
    Array.from = callable([].slice);
}());
