var chai           = require('chai');
var chaiAsPromised = require('chai-as-promised');

chai.use(chaiAsPromised);

module.exports = chai.assert;
