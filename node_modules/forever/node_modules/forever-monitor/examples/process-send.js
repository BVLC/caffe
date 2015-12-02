
setInterval(function () {
  if (process.send) {
    process.send({ from: 'child' });
  }
}, 1000)