function noop() {
  console.log('IGNORED!')
}
process.on('SIGTERM',noop);
process.on('SIGINT',noop);
setInterval(function (){
  console.log('heartbeat');
}, 100);