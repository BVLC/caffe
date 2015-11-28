function Hello(){
	var name;
	this.setName=function(tyname){
		name = tyname;
		
	}
	this.sayHello = function(){
		console.log("Hello"+name);
	}
}

module.exports = Hello;


