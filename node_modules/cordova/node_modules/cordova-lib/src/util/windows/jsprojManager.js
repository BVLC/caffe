/**
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
 */

/* jshint quotmark:false, unused:false */

/*
 Helper for dealing with Windows Store JS app .jsproj files
 */


var util = require('util'),
    xml_helpers = require('cordova-common').xmlHelpers,
    et = require('elementtree'),
    fs = require('fs'),
    glob = require('glob'),
    shell = require('shelljs'),
    events = require('cordova-common').events,
    path = require('path'),
    semver = require('semver');

var WinCSharpProjectTypeGUID = "{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}";  // .csproj
var WinCplusplusProjectTypeGUID = "{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}";  // .vcxproj

// Match a JavaScript Project
var JsProjectRegEx = /(Project\("\{262852C6-CD72-467D-83FE-5EEB1973A190}"\)\s*=\s*"[^"]+",\s*"[^"]+",\s*"\{[0-9a-f\-]+}"[^\r\n]*[\r\n]*)/gi;

// Chars in a string that need to be escaped when used in a RegExp
var RegExpEscRegExp = /([.?*+\^$\[\]\\(){}|\-])/g;

function jsprojManager(location) {
    this.isUniversalWindowsApp = path.extname(location).toLowerCase() === ".projitems";
    this.projects = [];
    this.master = this.isUniversalWindowsApp ? new proj(location) : new jsproj(location);
    this.projectFolder = path.dirname(location);
}

function getProjectName(pluginProjectXML, relative_path) {
    var projNameElt = pluginProjectXML.find("PropertyGroup/ProjectName");
    // Falling back on project file name in case ProjectName is missing
    return !!projNameElt ? projNameElt.text : path.basename(relative_path, path.extname(relative_path));
}

jsprojManager.prototype = {
    _projects: null,

    write: function () {
        this.master.write();
        if (this._projects) {
            var that = this;
            this._projects.forEach(function (project) {
                if (project !== that.master && project.touched) {
                    project.write();
                }
            });
        }
    },

    addSDKRef: function (incText, targetConditions) {
        events.emit('verbose', 'jsprojManager.addSDKRef(incText: ' + incText + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        var item = createItemGroupElement('ItemGroup/SDKReference', incText, targetConditions);
        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.appendToRoot(item);
        });
    },

    removeSDKRef: function (incText, targetConditions) {
        events.emit('verbose', 'jsprojManager.removeSDKRef(incText: ' + incText + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.removeItemGroupElement('ItemGroup/SDKReference', incText, targetConditions);
        });
    },

    addResourceFileToProject: function (relPath, targetConditions) {
        events.emit('verbose', 'jsprojManager.addResourceFile(relPath: ' + relPath + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        // add hint path with full path
        var link = new et.Element('Link');
        link.text = relPath;
        var children = [link];

        var copyToOutputDirectory = new et.Element('CopyToOutputDirectory');
        copyToOutputDirectory.text = 'Always';
        children.push(copyToOutputDirectory);

        var item = createItemGroupElement('ItemGroup/Content', relPath, targetConditions, children);
        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.appendToRoot(item);
        });
    },

    removeResourceFileFromProject: function (relPath, targetConditions) {
        events.emit('verbose', 'jsprojManager.removeResourceFile(relPath: ' + relPath + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.removeItemGroupElement('ItemGroup/Content', relPath, targetConditions);
        });
    },

    addReference: function (relPath, targetConditions) {
        events.emit('verbose', 'jsprojManager.addReference(incText: ' + relPath + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        // add hint path with full path
        var hint_path = new et.Element('HintPath');
        hint_path.text = relPath;
        var children = [hint_path];

        var extName = path.extname(relPath);
        if (extName === ".winmd") {
            var mdFileTag = new et.Element("IsWinMDFile");
            mdFileTag.text = "true";
            children.push(mdFileTag);
        }

        var item = createItemGroupElement('ItemGroup/Reference', path.basename(relPath, extName), targetConditions, children);
        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.appendToRoot(item);
        });

    },

    removeReference: function (relPath, targetConditions) {
        events.emit('verbose', 'jsprojManager.removeReference(incText: ' + relPath + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        var extName = path.extname(relPath);
        var includeText = path.basename(relPath, extName);

        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.removeItemGroupElement('ItemGroup/Reference', includeText, targetConditions);
        });
    },

    addSourceFile: function (relative_path) {
        events.emit('verbose', 'jsprojManager.addSourceFile(relative_path: ' + relative_path + ')');
        this.master.addSourceFile(relative_path);
    },

    removeSourceFile: function (relative_path) {
        events.emit('verbose', 'jsprojManager.removeSourceFile(incText: ' + relative_path + ')');
        this.master.removeSourceFile(relative_path);
    },

    addProjectReference: function (relative_path, targetConditions) {
        events.emit('verbose', 'jsprojManager.addProjectReference(incText: ' + relative_path + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        // relative_path is the actual path to the file in the current OS, where-as inserted_path is what we write in
        // the project file, and is always in Windows format.
        relative_path = path.normalize(relative_path);
        var inserted_path = relative_path.split('/').join('\\');

        var pluginProjectXML = xml_helpers.parseElementtreeSync(relative_path);

        // find the guid + name of the referenced project
        var projectGuid = pluginProjectXML.find("PropertyGroup/ProjectGuid").text;
        var projName = getProjectName(pluginProjectXML, relative_path);

        // get the project type
        var projectTypeGuid = getProjectTypeGuid(relative_path);
        if (!projectTypeGuid) {
            throw new Error("unrecognized project type");
        }

        var preInsertText = "\tProjectSection(ProjectDependencies) = postProject\r\n" +
            "\t\t" + projectGuid + "=" + projectGuid + "\r\n" +
            "\tEndProjectSection\r\n";
        var postInsertText = '\r\nProject("' + projectTypeGuid + '") = "' +
            projName + '", "' + inserted_path + '", ' +
            '"' + projectGuid + '"\r\nEndProject';

        var matchingProjects = this._getMatchingProjects(targetConditions);
        if (matchingProjects.length === 0) {
            // No projects meet the specified target and version criteria, so nothing to do.
            return;
        }

        // Will we be writing into the .projitems file rather than individual .jsproj files?
        var useProjItems = this.isUniversalWindowsApp && matchingProjects.length === 1 && matchingProjects[0] === this.master;

        // There may be multiple solution files (for different VS versions) - process them all
        getSolutionPaths(this.projectFolder).forEach(function (solutionPath) {
            var solText = fs.readFileSync(solutionPath, {encoding: "utf8"});

            if (useProjItems) {
                // Insert a project dependency into every jsproj in the solution.
                var jsProjectFound = false;
                solText = solText.replace(JsProjectRegEx, function (match) {
                    jsProjectFound = true;
                    return match + preInsertText;
                });

                if (!jsProjectFound) {
                    throw new Error("no jsproj found in solution");
                }
            } else {
                // Insert a project dependency only for projects that match specified target and version
                matchingProjects.forEach(function (project) {
                    solText = solText.replace(getJsProjRegExForProject(path.basename(project.location)), function (match) {
                        return match + preInsertText;
                    });
                });
            }

            // Add the project after existing projects. Note that this fairly simplistic check should be fine, since the last
            // EndProject in the file should actually be an EndProject (and not an EndProjectSection, for example).
            var pos = solText.lastIndexOf("EndProject");
            if (pos === -1) {
                throw new Error("no EndProject found in solution");
            }
            pos += 10; // Move pos to the end of EndProject text
            solText = solText.slice(0, pos) + postInsertText + solText.slice(pos);

            fs.writeFileSync(solutionPath, solText, {encoding: "utf8"});
        });

        // Add the ItemGroup/ProjectReference to each matching cordova project :
        // <ItemGroup><ProjectReference Include="blahblah.csproj"/></ItemGroup>
        var item = createItemGroupElement('ItemGroup/ProjectReference', inserted_path, targetConditions);
        matchingProjects.forEach(function (project) {
            project.appendToRoot(item);
        });
    },

    removeProjectReference: function (relative_path, targetConditions) {
        events.emit('verbose', 'jsprojManager.removeProjectReference(incText: ' + relative_path + ', targetConditions: ' + JSON.stringify(targetConditions) + ')');

        // relative_path is the actual path to the file in the current OS, where-as inserted_path is what we write in
        // the project file, and is always in Windows format.
        relative_path = path.normalize(relative_path);
        var inserted_path = relative_path.split('/').join('\\');

        // find the guid + name of the referenced project
        var pluginProjectXML = xml_helpers.parseElementtreeSync(relative_path);
        var projectGuid = pluginProjectXML.find("PropertyGroup/ProjectGuid").text;
        var projName = getProjectName(pluginProjectXML, relative_path);

        // get the project type
        var projectTypeGuid = getProjectTypeGuid(relative_path);
        if (!projectTypeGuid) {
            throw new Error("unrecognized project type");
        }

        var preInsertTextRegExp = getProjectReferencePreInsertRegExp(projectGuid);
        var postInsertTextRegExp = getProjectReferencePostInsertRegExp(projName, projectGuid, inserted_path, projectTypeGuid);

        // There may be multiple solutions (for different VS versions) - process them all
        getSolutionPaths(this.projectFolder).forEach(function (solutionPath) {
            var solText = fs.readFileSync(solutionPath, {encoding: "utf8"});

            // To be safe (to handle subtle changes in formatting, for example), use a RegExp to find and remove
            // preInsertText and postInsertText

            solText = solText.replace(preInsertTextRegExp, function () {
                return "";
            });

            solText = solText.replace(postInsertTextRegExp, function () {
                return "";
            });

            fs.writeFileSync(solutionPath, solText, {encoding: "utf8"});
        });

        this._getMatchingProjects(targetConditions).forEach(function (project) {
            project.removeItemGroupElement('ItemGroup/ProjectReference', inserted_path, targetConditions);
        });
    },

    _getMatchingProjects: function (targetConditions) {
        // If specified, target can be 'all' (default), 'phone' or 'windows'. Ultimately should probably allow a comma
        // separated list, but not needed now.
        var target = getDeviceTarget(targetConditions);
        var versions = getVersions(targetConditions);

        if (target || versions) {
            var matchingProjects = this.projects.filter(function (project) {
                return (!target || target === project.target) &&
                    (!versions || semver.satisfies(project.getSemVersion(), versions, /* loose */ true));
            });

            if (matchingProjects.length < this.projects.length) {
                return matchingProjects;
            }
        }

        // All projects match. If this is a universal project, return the projitems file. Otherwise return our single
        // project.
        return [this.master];
    },

    get projects() {
        var projects = this._projects;
        if (!projects) {
            projects = [];
            this._projects = projects;

            if (this.isUniversalWindowsApp) {
                var projectPath = this.projectFolder;
                var projectFiles = glob.sync('*.jsproj', {cwd: projectPath});
                projectFiles.forEach(function (projectFile) {
                    projects.push(new jsproj(path.join(projectPath, projectFile)));
                });
            } else {
                this.projects.push(this.master);
            }
        }

        return projects;
    }
};

function getProjectReferencePreInsertRegExp(projectGuid) {
    projectGuid = escapeRegExpString(projectGuid);
    return new RegExp("\\s*ProjectSection\\(ProjectDependencies\\)\\s*=\\s*postProject\\s*" + projectGuid + "\\s*=\\s*" + projectGuid + "\\s*EndProjectSection", "gi");
}

function getProjectReferencePostInsertRegExp(projName, projectGuid, relative_path, projectTypeGuid) {
    projName = escapeRegExpString(projName);
    projectGuid = escapeRegExpString(projectGuid);
    relative_path = escapeRegExpString(relative_path);
    projectTypeGuid = escapeRegExpString(projectTypeGuid);
    return new RegExp('\\s*Project\\("' + projectTypeGuid + '"\\)\\s*=\\s*"' + projName + '"\\s*,\\s*"' + relative_path + '"\\s*,\\s*"' + projectGuid + '"\\s*EndProject', 'gi');
}

function getSolutionPaths(projectFolder) {
    return shell.ls(path.join(projectFolder, "*.sln")); // TODO:error handling
}

function escapeRegExpString(regExpString) {
    return regExpString.replace(RegExpEscRegExp, "\\$1");
}

function getJsProjRegExForProject(projectFile) {
    projectFile = escapeRegExpString(projectFile);
    return new RegExp('(Project\\("\\{262852C6-CD72-467D-83FE-5EEB1973A190}"\\)\\s*=\\s*"[^"]+",\\s*"' + projectFile + '",\\s*"\\{[0-9a-f\\-]+}"[^\\r\\n]*[\\r\\n]*)', 'gi');
}

function getProjectTypeGuid(projectPath) {
    switch (path.extname(projectPath)) {
        case ".vcxproj":
            return WinCplusplusProjectTypeGUID;

        case ".csproj":
            return WinCSharpProjectTypeGUID;
    }
    return null;
}

function createItemGroupElement(path, incText, targetConditions, children) {
    path = path.split('/');
    path.reverse();

    var lastElement = null;
    path.forEach(function (elementName) {
        var element = new et.Element(elementName);
        if (lastElement) {
            element.append(lastElement);
        } else {
            element.attrib.Include = incText;

            var condition = createConditionAttrib(targetConditions);
            if (condition) {
                element.attrib.Condition = condition;
            }

            if (children) {
                children.forEach(function (child) {
                    element.append(child);
                });
            }
        }
        lastElement = element;
    });

    return lastElement;
}

function getDeviceTarget(targetConditions) {
    var target = targetConditions.deviceTarget;
    if (target) {
        target = target.toLowerCase().trim();
        if (target === "all") {
            target = null;
        } else if (target === "win") {
            // Allow "win" as alternative to "windows"
            target = "windows";
        } else if (target !== 'phone' && target !== 'windows') {
            throw new Error('Invalid device-target attribute (must be "all", "phone", "windows" or "win"): ' + target);
        }
    }
    return target;
}

function getVersions(targetConditions) {
    var versions = targetConditions.versions;
    if (versions && !semver.validRange(versions, /* loose */ true)) {
        throw new Error('Invalid versions attribute (must be a valid semantic version range): ' + versions);
    }
    return versions;
}


/* proj */

function proj(location) {
    // Class to handle simple project xml operations
    if (!location) {
        throw new Error('Project file location can\'t be null or empty');
    }
    this.location = location;
    this.xml = xml_helpers.parseElementtreeSync(location);
}

proj.prototype = {
    write: function () {
        fs.writeFileSync(this.location, this.xml.write({indent: 4}), 'utf-8');
    },

    appendToRoot: function (element) {
        this.touched = true;
        this.xml.getroot().append(element);
    },

    removeItemGroupElement: function (path, incText, targetConditions) {
        var xpath = path + '[@Include="' + incText + '"]';
        var condition = createConditionAttrib(targetConditions);
        if (condition) {
            xpath += '[@Condition="' + condition + '"]';
        }
        xpath += '/..';

        var itemGroup = this.xml.find(xpath);
        if (itemGroup) {
            this.touched = true;
            this.xml.getroot().remove(itemGroup);
        }
    },

    addSourceFile: function (relative_path) {
        // we allow multiple paths to be passed at once as array so that
        // we don't create separate ItemGroup for each source file, CB-6874
        if (!(relative_path instanceof Array)) {
            relative_path = [relative_path];
        }

        // make ItemGroup to hold file.
        var item = new et.Element('ItemGroup');

        relative_path.forEach(function (filePath) {
            // filePath is never used to find the actual file - it determines what we write to the project file, and so
            // should always be in Windows format.
            filePath = filePath.split('/').join('\\');

            var content = new et.Element('Content');
            content.attrib.Include = filePath;
            item.append(content);
        });

        this.appendToRoot(item);
    },

    removeSourceFile: function (relative_path) {
        var isRegexp = relative_path instanceof RegExp;
        if (!isRegexp) {
            // relative_path is never used to find the actual file - it determines what we write to the project file,
            // and so should always be in Windows format.
            relative_path = relative_path.split('/').join('\\');
        }

        var root = this.xml.getroot();
        var that = this;
        // iterate through all ItemGroup/Content elements and remove all items matched
        this.xml.findall('ItemGroup').forEach(function (group) {
            // matched files in current ItemGroup
            var filesToRemove = group.findall('Content').filter(function (item) {
                if (!item.attrib.Include) {
                    return false;
                }
                return isRegexp ? item.attrib.Include.match(relative_path) : item.attrib.Include === relative_path;
            });

            // nothing to remove, skip..
            if (filesToRemove.length < 1) {
                return;
            }

            filesToRemove.forEach(function (file) {
                // remove file reference
                group.remove(file);
            });
            // remove ItemGroup if empty
            if (group.findall('*').length < 1) {
                that.touched = true;
                root.remove(group);
            }
        });
    }
};


/* jsproj */

function jsproj(location) {
    function targetPlatformIdentifierToDevice(jsprojPlatform) {
        var index = ["Windows", "WindowsPhoneApp", "UAP"].indexOf(jsprojPlatform);
        if (index < 0) {
            throw new Error("Unknown TargetPlatformIdentifier '" + jsprojPlatform + "' in project file '" + location + "'");
        }
        return ["windows", "phone", "windows"][index];
    }

    function validateVersion(version) {
        version = version.split('.');
        while (version.length < 3) {
            version.push("0");
        }
        return version.join(".");
    }

    // Class to handle a jsproj file
    proj.call(this, location);

    var propertyGroup = this.xml.find('PropertyGroup[TargetPlatformIdentifier]');
    if (!propertyGroup) {
        throw new Error("Unable to find PropertyGroup/TargetPlatformIdentifier in project file '" + this.location + "'");
    }

    var jsprojPlatform = propertyGroup.find('TargetPlatformIdentifier').text;
    this.target = targetPlatformIdentifierToDevice(jsprojPlatform);

    var version = propertyGroup.find('TargetPlatformVersion');
    if (!version) {
        throw new Error("Unable to find PropertyGroup/TargetPlatformVersion in project file '" + this.location + "'");
    }
    this.version = validateVersion(version.text);
}

util.inherits(jsproj, proj);

jsproj.prototype.target = null;
jsproj.prototype.version = null;

// Returns valid semantic version (http://semver.org/).
jsproj.prototype.getSemVersion = function () {
    // For example, for version 10.0.10240.0 we will return 10.0.10240 (first three components)
    var semVersion = this.version;
    var splittedVersion = semVersion.split('.');
    if (splittedVersion.length > 3) {
        semVersion = splittedVersion.splice(0, 3).join('.');
    }

    return semVersion;
	// Alternative approach could be replacing last dot with plus sign to
	// be complaint w/ semver specification, for example
	// 10.0.10240.0 -> 10.0.10240+0
};

/* Common support functions */

function createConditionAttrib(targetConditions) {
    var arch = targetConditions.arch;
    if (arch) {
        if (arch === "arm") {
            // Specifcally allow "arm" as alternative to "ARM"
            arch = "ARM";
        } else if (arch !== "x86" && arch !== "x64" && arch !== "ARM") {
            throw new Error('Invalid arch attribute (must be "x86", "x64" or "ARM"): ' + arch);
        }
        return "'$(Platform)'=='" + arch + "'";
    }
    return null;
}


module.exports = jsprojManager;
