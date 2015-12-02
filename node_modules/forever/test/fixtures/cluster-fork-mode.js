var cluster = require('cluster');
console.log(cluster.isMaster ? 'master fork':'cluster fork');