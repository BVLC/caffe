var name;
exports.setName = function(tyname){
	name = tyname;
};

exports.sayHello = function(){
	console.log("Hello"+name);
};