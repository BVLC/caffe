process.on('SIGTERM',function () {
  console.log('received SIGTERM');
  setTimeout(function () {
    console.log('Exiting after some time.');
    process.exit(0);
  }, 1000);
});

setInterval(function (){
  console.log('Heartbeat');
}, 100);

// run with: --killSignal
// forever --killSignal=SIGTERM -w start server.js