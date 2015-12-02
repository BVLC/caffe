var Log = require('./').Log;

function doThings(options) {
  console.log();
  console.log(options);
  var log = new Log(options);
  log.header("Header line.");
  log.subhead("Subhead line.");
  log.write("Testing").write(" 123...").writeln("done!");
  log.write("Verbose: ").verbose.write("YES").or.write("NO").always.write(", ");
  log.notverbose.write("NO").or.write("YES").always.writeln("!");
  log.warn("This is a warning.");
  log.write("Doing something...").warn();
  log.error("This is an error.");
  log.write("Doing something...").error();
  log.ok("This is ok.");
  log.write("Doing something...").ok();
  log.errorlns("This is a very long line in errorlns that should wrap eventually, given that it is a very long line.");
  log.oklns("This is a very long line in oklns that should wrap eventually, given that it is a very long line.");
  log.success("This is a success message.");
  log.fail("This is a fail message.");
  log.debug("This is a debug message.");
}

doThings({});
doThings({verbose: true});
