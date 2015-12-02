if (process.send) {


    process.on('message', function (m) {
      process.send({pong: true, message: m});
    });

}
