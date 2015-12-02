#!/usr/bin/env python

# ---
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ---

import os
import re
import sys
import shutil
import datetime
import subprocess

PROGRAM = sys.argv[0]

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def main():

    # all work done in directory of this program
    baseDir  = os.path.dirname(PROGRAM)
    os.chdir(baseDir)

    testsDir = os.path.abspath("../../thirdparty/commonjs-tests")
    outDir   = os.path.abspath("out")

    # validate testsDir
    if not os.path.exists(testsDir):
        error("tests dir does not exist: %s" % testsDir)
    
    if not os.path.isdir(testsDir):
        error("tests dir is not a directory: %s" % testsDir)

    # validate and reset outDir
    if os.path.exists(outDir):
        if not os.path.isdir(outDir):
            error("out dir is not a directory: %s" % outDir)

        shutil.rmtree(outDir)
        
    os.makedirs(outDir)

    tests = getTests(testsDir)
    
    # now all work done in outDir
    os.chdir("out")

    # build the individual tests
    iframes = []
    for test in tests:
        testName = test.replace('/', '-')
        htmlFile = buildTest(os.path.join(testsDir, test), testName)
        iframes.append("<iframe width='100%%' height='30' src='%s'></iframe>" % htmlFile)

    iframesLines = "\n".join(iframes)
    
    # build the browser launcher
    html = fileContents("../launcher-main.template.html")

    html = html.replace("@iframes@", iframesLines)
    html = html.replace("@date@", getDate())
    
    oFileName = "launcher-all.html"
    oFile = file(oFileName, "w")
    oFile.write(html)
    oFile.close()
    
    print
    print "Generated browser test: %s" % os.path.abspath(oFileName)
    print 
    print "You can run the test as a local file under Safari but not Chrome."
    print "To test under Chrome, access the files via http://"

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def buildTest(testDir, testName):

    log("generating tests for %s" % (testName))

    html = fileContents("../launcher-in-iframe.template.html")
    
    # get the list of modules
    modules = getModules(testDir)
    
    modulesSource = []
    
    modulesSource.append("try {")
    
    for module in modules:
        source = fileContents("%s.js" % os.path.join(testDir, module))
        
        modulesSource.append("//----------------------------------------------")
        modulesSource.append("define('%s', function(require,exports,module) {" % module)
        modulesSource.append(source.strip())
        modulesSource.append("});")
        modulesSource.append("")

    modulesSource.append("}")
    modulesSource.append("catch(e) {")
    modulesSource.append("   console.log('exception thrown loading modules: ' + e)")
    modulesSource.append("}")
        
    modulesLines = "\n".join(modulesSource)
    
    html = html.replace("@modules@", modulesLines)
    html = html.replace("@title@", testName)
    html = html.replace("@date@", getDate())

    # build HTML launcher for iframe
    oFileName = "%s.html" % testName
    oFile = file(oFileName, "w")
    oFile.write(html)
    oFile.close()
    
    return oFileName
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def getModules(testDir):
    modules = []
    for root, dirs, files in os.walk(testDir):
        for file in files:
            if not file.endswith(".js"): continue
            
            moduleSource = os.path.relpath(os.path.join(root, file), testDir)
            moduleName   = os.path.splitext(moduleSource)[0]
            modules.append(moduleName)

    return modules

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def getTests(testDir):
    tests = []
    for root, dirs, files in os.walk(testDir):
        if "program.js" in files:
            tests.append(os.path.relpath(root, testDir))
            
    return tests
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def run(cmdArgs):
    result = subprocess.Popen(cmdArgs, stdout=subprocess.PIPE).communicate()[0]
    if not re.match(r"\s*", result):
        print result

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def fileContents(iFileName):
    iFile = file(iFileName)
    contents = iFile.read()
    iFile.close()
    
    return contents

def getDate():
     return datetime.datetime.today().isoformat(" ")
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def log(message):
    print "%s: %s" % (PROGRAM, message)

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def error(message):
    log(message)
    exit(1)

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
main()