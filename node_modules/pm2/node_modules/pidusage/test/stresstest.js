var pusage = require('../')

//stress test to compare with top or another tool
console.log('This is my PID: %s', process.pid)

//classic "drop somewhere"... yeah I'm a lazy guy
var formatBytes = function(bytes, precision) {
  var kilobyte = 1024
  var megabyte = kilobyte * 1024
  var gigabyte = megabyte * 1024
  var terabyte = gigabyte * 1024

  if ((bytes >= 0) && (bytes < kilobyte)) {
    return bytes + ' B   '
  } else if ((bytes >= kilobyte) && (bytes < megabyte)) {
    return (bytes / kilobyte).toFixed(precision) + ' KB  '
  } else if ((bytes >= megabyte) && (bytes < gigabyte)) {
    return (bytes / megabyte).toFixed(precision) + ' MB  '
  } else if ((bytes >= gigabyte) && (bytes < terabyte)) {
    return (bytes / gigabyte).toFixed(precision) + ' GB  '
  } else if (bytes >= terabyte) {
    return (bytes / terabyte).toFixed(precision) + ' TB  '
  } else {
    return bytes + ' B   '
  }
}

var i = 0, big_memory_leak = []

var stress = function(cb) {

  console.log('\033[2J')

  var j = 500, arr = []

  while(j--) {
    arr[j] = []

    for (var k = 0; k < 1000; k++) {
      arr[j][k] = {lorem: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum non odio venenatis, pretium ligula nec, fringilla ipsum. Sed a erat et sem blandit dignissim. Pellentesque sollicitudin felis eu mattis porta. Nullam nec nibh nisl. Phasellus convallis vulputate massa vitae fringilla. Etiam facilisis lectus in odio lacinia rutrum. Praesent facilisis vitae urna a suscipit. Aenean lacinia blandit lorem, et ullamcorper metus sagittis faucibus. Nam porta eros nisi, at adipiscing quam varius eu. Vivamus sed sem quis lorem varius posuere ut quis elit.'}
    }
  }

  big_memory_leak.push(arr)

  pusage.stat(process.pid, function(err, stat) {
    console.log('Pcpu: %s', stat.cpu)
    console.log('Mem: %s', formatBytes(stat.memory))

    //this is to compare with node-usage results, but it's broken on v11.12
    // require('usage').lookup(process.pid, {keepHistory: true}, function(err, stat) {
    // console.log('Usage Pcpu: %s', stat.cpu)
    // console.log('Usage Mem: %s', formatBytes(stat.memory))

      if(i == 100)
        return cb(true)
      else if(stat.memory > 209715200) {
        console.log("That's enough right?")
        cb(true)
      }

      i++
      return cb(false)
    // })
  })
}

var interval = function() {
  return setTimeout(function() {

    stress(function(stop) {
      if(stop)
        process.exit()
      else
        return interval()
    })

  }, 400)
}

setTimeout(function() {
  interval()
}, 2000)
