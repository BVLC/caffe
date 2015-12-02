if (process.send) {
  process.send({from: 'child'});
  process.disconnect();
}
