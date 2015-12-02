var util = require('util'),
    f = util.format,
    EventEmitter = require('events').EventEmitter,
    path = require('path'),
    uuid = require('node-uuid'),
    fork = require('child_process').fork,
    pbxWriter = require('./pbxWriter'),
    pbxFile = require('./pbxFile'),
    fs = require('fs'),
    parser = require('./parser/pbxproj'),
    COMMENT_KEY = /_comment$/

function pbxProject(filename) {
    if (!(this instanceof pbxProject))
        return new pbxProject(filename);

    this.filepath = path.resolve(filename)
}

util.inherits(pbxProject, EventEmitter)

pbxProject.prototype.parse = function (cb) {
    var worker = fork(__dirname + '/parseJob.js', [this.filepath])

    worker.on('message', function (msg) {
        if (msg.name == 'SyntaxError' || msg.code) {
            this.emit('error', msg);
        } else {
            this.hash = msg;
            this.emit('end', null, msg)
        }
    }.bind(this));

    if (cb) {
        this.on('error', cb);
        this.on('end', cb);
    }

    return this;
}

pbxProject.prototype.parseSync = function () {
    var file_contents = fs.readFileSync(this.filepath, 'utf-8');

    this.hash = parser.parse(file_contents);
    return this;
}

pbxProject.prototype.writeSync = function () {
    this.writer = new pbxWriter(this.hash);
    return this.writer.writeSync();
}

pbxProject.prototype.allUuids = function () {
    var sections = this.hash.project.objects,
        uuids = [],
        section;

    for (key in sections) {
        section = sections[key]
        uuids = uuids.concat(Object.keys(section))
    }

    uuids = uuids.filter(function (str) {
        return !COMMENT_KEY.test(str) && str.length == 24;
    });

    return uuids;
}

pbxProject.prototype.generateUuid = function () {
    var id = uuid.v4()
                .replace(/-/g,'')
                .substr(0,24)
                .toUpperCase()

    if (this.allUuids().indexOf(id) >= 0) {
        return this.generateUuid();
    } else {
        return id;
    }
}

pbxProject.prototype.addPluginFile = function (path, opt) {
    var file = new pbxFile(path, opt);

    file.plugin = true; // durr
    correctForPluginsPath(file, this);

    // null is better for early errors
    if (this.hasFile(file.path)) return null;

    file.fileRef = this.generateUuid();

    this.addToPbxFileReferenceSection(file);    // PBXFileReference
    this.addToPluginsPbxGroup(file);            // PBXGroup

    return file;
}

pbxProject.prototype.removePluginFile = function (path, opt) {
    var file = new pbxFile(path, opt);
    correctForPluginsPath(file, this);

    this.removeFromPbxFileReferenceSection(file);    // PBXFileReference
    this.removeFromPluginsPbxGroup(file);            // PBXGroup

    return file;
}


pbxProject.prototype.addSourceFile = function (path, opt) {
  
    var file = this.addPluginFile(path, opt);
    if (!file) return false;

    file.target = opt ? opt.target : undefined;
    file.uuid = this.generateUuid();

    this.addToPbxBuildFileSection(file);        // PBXBuildFile
    this.addToPbxSourcesBuildPhase(file);       // PBXSourcesBuildPhase

    return file;
}


pbxProject.prototype.removeSourceFile = function (path, opt) {
    var file = this.removePluginFile(path, opt)
    file.target = opt ? opt.target : undefined;
    this.removeFromPbxBuildFileSection(file);        // PBXBuildFile
    this.removeFromPbxSourcesBuildPhase(file);       // PBXSourcesBuildPhase

    return file;
}

pbxProject.prototype.addHeaderFile = function (path, opt) {
    return this.addPluginFile(path, opt)
}

pbxProject.prototype.removeHeaderFile = function (path, opt) {
    return this.removePluginFile(path, opt)
}

pbxProject.prototype.addResourceFile = function (path, opt) {
    opt = opt || {};

    var file;

    if (opt.plugin) {
        file = this.addPluginFile(path, opt);
        if (!file) return false;
    } else {
        file = new pbxFile(path, opt);
        if (this.hasFile(file.path)) return false;
    }

    file.uuid = this.generateUuid();
    file.target = opt ? opt.target : undefined;

    if (!opt.plugin) {
        correctForResourcesPath(file, this);
        file.fileRef = this.generateUuid();
    }

    this.addToPbxBuildFileSection(file);        // PBXBuildFile
    this.addToPbxResourcesBuildPhase(file);     // PBXResourcesBuildPhase

    if (!opt.plugin) {
        this.addToPbxFileReferenceSection(file);    // PBXFileReference
        this.addToResourcesPbxGroup(file);          // PBXGroup
    }

    return file;
}

pbxProject.prototype.removeResourceFile = function (path, opt) {
    var file = new pbxFile(path, opt);
    file.target = opt ? opt.target : undefined;
    
    correctForResourcesPath(file, this);

    this.removeFromPbxBuildFileSection(file);        // PBXBuildFile
    this.removeFromPbxFileReferenceSection(file);    // PBXFileReference
    this.removeFromResourcesPbxGroup(file);          // PBXGroup
    this.removeFromPbxResourcesBuildPhase(file);     // PBXResourcesBuildPhase
    
    return file;
}

pbxProject.prototype.addFramework = function (fpath, opt) {
    var file = new pbxFile(fpath, opt);
    // catch duplicates
    if (this.hasFile(file.path)) return false;

    file.uuid = this.generateUuid();
    file.fileRef = this.generateUuid();    
    file.target = opt ? opt.target : undefined;


    this.addToPbxBuildFileSection(file);        // PBXBuildFile
    this.addToPbxFileReferenceSection(file);    // PBXFileReference
    this.addToFrameworksPbxGroup(file);         // PBXGroup
    this.addToPbxFrameworksBuildPhase(file);    // PBXFrameworksBuildPhase
    
    if(opt && opt.customFramework == true) {
      this.addToFrameworkSearchPaths(file);
    }

    return file;
}

pbxProject.prototype.removeFramework = function (fpath, opt) {
    var file = new pbxFile(fpath, opt);
    file.target = opt ? opt.target : undefined;

    this.removeFromPbxBuildFileSection(file);        // PBXBuildFile
    this.removeFromPbxFileReferenceSection(file);    // PBXFileReference
    this.removeFromFrameworksPbxGroup(file);         // PBXGroup
    this.removeFromPbxFrameworksBuildPhase(file);    // PBXFrameworksBuildPhase
    
    if(opt && opt.customFramework) {
      this.removeFromFrameworkSearchPaths(path.dirname(fpath));
    }

    return file;
}

pbxProject.prototype.addStaticLibrary = function (path, opt) {
    opt = opt || {};

    var file;

    if (opt.plugin) {
        file = this.addPluginFile(path, opt);
        if (!file) return false;
    } else {
        file = new pbxFile(path, opt);
        if (this.hasFile(file.path)) return false;
    }

    file.uuid = this.generateUuid();
    file.target = opt ? opt.target : undefined;

    if (!opt.plugin) {
        file.fileRef = this.generateUuid();
        this.addToPbxFileReferenceSection(file);    // PBXFileReference
    }

    this.addToPbxBuildFileSection(file);        // PBXBuildFile
    this.addToPbxFrameworksBuildPhase(file);    // PBXFrameworksBuildPhase
    this.addToLibrarySearchPaths(file);        // make sure it gets built!

    return file;
}

// helper addition functions
pbxProject.prototype.addToPbxBuildFileSection = function (file) {
    var commentKey = f("%s_comment", file.uuid);

    this.pbxBuildFileSection()[file.uuid] = pbxBuildFileObj(file);
    this.pbxBuildFileSection()[commentKey] = pbxBuildFileComment(file);
}

pbxProject.prototype.removeFromPbxBuildFileSection = function (file) {
    var uuid;

    for(uuid in this.pbxBuildFileSection()) {
        if(this.pbxBuildFileSection()[uuid].fileRef_comment == file.basename) {
            file.uuid = uuid;
            delete this.pbxBuildFileSection()[uuid];
        }
    }
    var commentKey = f("%s_comment", file.uuid);
    delete this.pbxBuildFileSection()[commentKey];
}

pbxProject.prototype.addPbxGroup = function (filePathsArray, name, path, sourceTree) {
    var groups = this.hash.project.objects['PBXGroup'],
        pbxGroupUuid = this.generateUuid(),
        commentKey = f("%s_comment", pbxGroupUuid),
        pbxGroup = {
            isa: 'PBXGroup',
            children: [],
            name: name,
            path: path,
            sourceTree: sourceTree ? sourceTree : '"<group>"'
        },
        fileReferenceSection = this.pbxFileReferenceSection(),
        filePathToReference = {};
        
    for (var key in fileReferenceSection) {
        // only look for comments
        if (!COMMENT_KEY.test(key)) continue;
        
        var fileReferenceKey = key.split(COMMENT_KEY)[0],
            fileReference = fileReferenceSection[fileReferenceKey];
        
        filePathToReference[fileReference.path] = {fileRef: fileReferenceKey, basename: fileReferenceSection[key]};
    }

    for (var index = 0; index < filePathsArray.length; index++) {
        var filePath = filePathsArray[index],
            filePathQuoted = "\"" + filePath + "\"";
        if (filePathToReference[filePath]) {
            pbxGroup.children.push(pbxGroupChild(filePathToReference[filePath]));
            continue;
        } else if (filePathToReference[filePathQuoted]) {
            pbxGroup.children.push(pbxGroupChild(filePathToReference[filePathQuoted]));
            continue;
        }
        
        var file = new pbxFile(filePath);
        file.uuid = this.generateUuid();
        file.fileRef = this.generateUuid();
        this.addToPbxFileReferenceSection(file);    // PBXFileReference
        this.addToPbxBuildFileSection(file);        // PBXBuildFile
        pbxGroup.children.push(pbxGroupChild(file));
    }
    
    if (groups) {
        groups[pbxGroupUuid] = pbxGroup;
        groups[commentKey] = name;
    }
    
    return {uuid: pbxGroupUuid, pbxGroup: pbxGroup};
}

pbxProject.prototype.addToPbxFileReferenceSection = function (file) {
    var commentKey = f("%s_comment", file.fileRef);

    this.pbxFileReferenceSection()[file.fileRef] = pbxFileReferenceObj(file);
    this.pbxFileReferenceSection()[commentKey] = pbxFileReferenceComment(file);
}

pbxProject.prototype.removeFromPbxFileReferenceSection = function (file) {

    var i;
    var refObj = pbxFileReferenceObj(file);
    for(i in this.pbxFileReferenceSection()) {
        if(this.pbxFileReferenceSection()[i].name == refObj.name ||
           ('"' + this.pbxFileReferenceSection()[i].name + '"') == refObj.name ||
           this.pbxFileReferenceSection()[i].path == refObj.path ||
           ('"' + this.pbxFileReferenceSection()[i].path + '"') == refObj.path) {
            file.fileRef = file.uuid = i;
            delete this.pbxFileReferenceSection()[i];
            break;
        }
    }
    var commentKey = f("%s_comment", file.fileRef);
    if(this.pbxFileReferenceSection()[commentKey] != undefined) {
        delete this.pbxFileReferenceSection()[commentKey];
    }

    return file;
}

pbxProject.prototype.addToPluginsPbxGroup = function (file) {
    var pluginsGroup = this.pbxGroupByName('Plugins');
    pluginsGroup.children.push(pbxGroupChild(file));
}

pbxProject.prototype.removeFromPluginsPbxGroup = function (file) {
    var pluginsGroupChildren = this.pbxGroupByName('Plugins').children, i;
    for(i in pluginsGroupChildren) {
        if(pbxGroupChild(file).value == pluginsGroupChildren[i].value &&
           pbxGroupChild(file).comment == pluginsGroupChildren[i].comment) {
            pluginsGroupChildren.splice(i, 1);
            break;
        }
    }
}

pbxProject.prototype.addToResourcesPbxGroup = function (file) {
    var pluginsGroup = this.pbxGroupByName('Resources');
    pluginsGroup.children.push(pbxGroupChild(file));
}

pbxProject.prototype.removeFromResourcesPbxGroup = function (file) {
    var pluginsGroupChildren = this.pbxGroupByName('Resources').children, i;
    for(i in pluginsGroupChildren) {
        if(pbxGroupChild(file).value == pluginsGroupChildren[i].value &&
           pbxGroupChild(file).comment == pluginsGroupChildren[i].comment) {
            pluginsGroupChildren.splice(i, 1);
            break;
        }
    }
}

pbxProject.prototype.addToFrameworksPbxGroup = function (file) {
    var pluginsGroup = this.pbxGroupByName('Frameworks');
    pluginsGroup.children.push(pbxGroupChild(file));
}

pbxProject.prototype.removeFromFrameworksPbxGroup = function (file) {
    var pluginsGroupChildren = this.pbxGroupByName('Frameworks').children;
    
    for(i in pluginsGroupChildren) {
        if(pbxGroupChild(file).value == pluginsGroupChildren[i].value &&
           pbxGroupChild(file).comment == pluginsGroupChildren[i].comment) {
            pluginsGroupChildren.splice(i, 1);
            break;
        }
    }
}
pbxProject.prototype.addToPbxSourcesBuildPhase = function (file) {
    var sources = this.pbxSourcesBuildPhaseObj(file.target);
    sources.files.push(pbxBuildPhaseObj(file));
}

pbxProject.prototype.removeFromPbxSourcesBuildPhase = function (file) {

    var sources = this.pbxSourcesBuildPhaseObj(file.target), i;
    for(i in sources.files) {
        if(sources.files[i].comment == longComment(file)) {
            sources.files.splice(i, 1);
            break; 
        }
    }
}

pbxProject.prototype.addToPbxResourcesBuildPhase = function (file) {
    var sources = this.pbxResourcesBuildPhaseObj(file.target);
    sources.files.push(pbxBuildPhaseObj(file));
}

pbxProject.prototype.removeFromPbxResourcesBuildPhase = function (file) {
    var sources = this.pbxResourcesBuildPhaseObj(file.target), i;

    for(i in sources.files) {
        if(sources.files[i].comment == longComment(file)) {
            sources.files.splice(i, 1);
            break;
        }
    }
}

pbxProject.prototype.addToPbxFrameworksBuildPhase = function (file) {
    var sources = this.pbxFrameworksBuildPhaseObj(file.target);
    sources.files.push(pbxBuildPhaseObj(file));
}

pbxProject.prototype.removeFromPbxFrameworksBuildPhase = function (file) {
    var sources = this.pbxFrameworksBuildPhaseObj(file.target);
    for(i in sources.files) {
        if(sources.files[i].comment == longComment(file)) {
            sources.files.splice(i, 1);
            break;
        }
    }
}

pbxProject.prototype.addXCConfigurationList = function (configurationObjectsArray, defaultConfigurationName, comment) {
    var pbxBuildConfigurationSection = this.pbxXCBuildConfigurationSection(),
        pbxXCConfigurationListSection = this.pbxXCConfigurationList(), 
        xcConfigurationListUuid = this.generateUuid(),
        commentKey = f("%s_comment", xcConfigurationListUuid),
        xcConfigurationList = { 
            isa: 'XCConfigurationList',
            buildConfigurations: [],
            defaultConfigurationIsVisible: 0,
            defaultConfigurationName: defaultConfigurationName 
        };

    for (var index = 0; index < configurationObjectsArray.length; index++) {
        var configuration = configurationObjectsArray[index],
            configurationUuid = this.generateUuid(),
            configurationCommentKey = f("%s_comment", configurationUuid);

        pbxBuildConfigurationSection[configurationUuid] = configuration;
        pbxBuildConfigurationSection[configurationCommentKey] = configuration.name;
        xcConfigurationList.buildConfigurations.push({value: configurationUuid, comment: configuration.name});
    }
    
    if (pbxXCConfigurationListSection) {
        pbxXCConfigurationListSection[xcConfigurationListUuid] = xcConfigurationList;
        pbxXCConfigurationListSection[commentKey] = comment;
    }
    
    return {uuid: xcConfigurationListUuid, xcConfigurationList: xcConfigurationList};
}

pbxProject.prototype.addTargetDependency = function (target, dependencyTargets) {
    if (!target)
        return undefined;

    var nativeTargets = this.pbxNativeTarget();

    if (typeof nativeTargets[target] == "undefined")
        throw new Error("Invalid target: " + target);
        
    for (var index = 0; index < dependencyTargets.length; index++) {
        var dependencyTarget = dependencyTargets[index];
        if (typeof nativeTargets[dependencyTarget] == "undefined")
            throw new Error("Invalid target: " + dependencyTarget);
    }
    
    var pbxTargetDependency = 'PBXTargetDependency',
        pbxContainerItemProxy = 'PBXContainerItemProxy',
        pbxTargetDependencySection = this.hash.project.objects[pbxTargetDependency],
        pbxContainerItemProxySection = this.hash.project.objects[pbxContainerItemProxy];

    for (var index = 0; index < dependencyTargets.length; index++) {
        var dependencyTargetUuid = dependencyTargets[index],
            dependencyTargetCommentKey = f("%s_comment", dependencyTargetUuid),
            targetDependencyUuid = this.generateUuid(),
            targetDependencyCommentKey = f("%s_comment", targetDependencyUuid),
            itemProxyUuid = this.generateUuid(),
            itemProxyCommentKey = f("%s_comment", itemProxyUuid),
            itemProxy =  {
                isa: pbxContainerItemProxy,
                containerPortal: this.hash.project['rootObject'],
                containerPortal_comment: this.hash.project['rootObject_comment'],
                proxyType: 1,
                remoteGlobalIDString: dependencyTargetUuid,
                remoteInfo: nativeTargets[dependencyTargetUuid].name 
            },
            targetDependency = {
                isa: pbxTargetDependency,
                target: dependencyTargetUuid, 
                target_comment: nativeTargets[dependencyTargetCommentKey],
                targetProxy: itemProxyUuid,
                targetProxy_comment: pbxContainerItemProxy
            };
            
        if (pbxContainerItemProxySection && pbxTargetDependencySection) {
            pbxContainerItemProxySection[itemProxyUuid] = itemProxy;
            pbxContainerItemProxySection[itemProxyCommentKey] = pbxContainerItemProxy;
            pbxTargetDependencySection[targetDependencyUuid] = targetDependency;
            pbxTargetDependencySection[targetDependencyCommentKey] = pbxTargetDependency;
            nativeTargets[target].dependencies.push({value: targetDependencyUuid, comment: pbxTargetDependency})
        }
    }
    
    return {uuid: target, target: nativeTargets[target]};
}

pbxProject.prototype.addBuildPhase = function (filePathsArray, isa, comment) {
    var section = this.hash.project.objects[isa],
        fileReferenceSection = this.pbxFileReferenceSection(),
        buildFileSection = this.pbxBuildFileSection(),
        buildPhaseUuid = this.generateUuid(),
        commentKey = f("%s_comment", buildPhaseUuid),
        buildPhase = {
            isa: isa,
            buildActionMask: 2147483647,
            files: [],
            runOnlyForDeploymentPostprocessing: 0
        },
        filePathToBuildFile = {};
    
    for (var key in buildFileSection) {
        // only look for comments
        if (!COMMENT_KEY.test(key)) continue;
        
        var buildFileKey = key.split(COMMENT_KEY)[0],
            buildFile = buildFileSection[buildFileKey];
            fileReference = fileReferenceSection[buildFile.fileRef];

        if (!fileReference) continue;

        var pbxFileObj = new pbxFile(fileReference.path);
        
        filePathToBuildFile[fileReference.path] = {uuid: buildFileKey, basename: pbxFileObj.basename, group: pbxFileObj.group};
    }

    for (var index = 0; index < filePathsArray.length; index++) {
        var filePath = filePathsArray[index],
            filePathQuoted = "\"" + filePath + "\"",
            file = new pbxFile(filePath);

        if (filePathToBuildFile[filePath]) {
            buildPhase.files.push(pbxBuildPhaseObj(filePathToBuildFile[filePath]));
            continue;
        } else if (filePathToBuildFile[filePathQuoted]) {
            buildPhase.files.push(pbxBuildPhaseObj(filePathToBuildFile[filePathQuoted]));
            continue;
        }
        
        file.uuid = this.generateUuid();
        file.fileRef = this.generateUuid();
        this.addToPbxFileReferenceSection(file);    // PBXFileReference
        this.addToPbxBuildFileSection(file);        // PBXBuildFile
        buildPhase.files.push(pbxBuildPhaseObj(file));
    }
    
    if (section) {
        section[buildPhaseUuid] = buildPhase;
        section[commentKey] = comment;
    }
    
    return {uuid: buildPhaseUuid, buildPhase: buildPhase};
}

// helper access functions
pbxProject.prototype.pbxProjectSection = function () {
    return this.hash.project.objects['PBXProject'];
}
pbxProject.prototype.pbxBuildFileSection = function () {
    return this.hash.project.objects['PBXBuildFile'];
}

pbxProject.prototype.pbxXCBuildConfigurationSection = function () {
    return this.hash.project.objects['XCBuildConfiguration'];
}

pbxProject.prototype.pbxFileReferenceSection = function () {
    return this.hash.project.objects['PBXFileReference'];
}

pbxProject.prototype.pbxNativeTarget = function () {
    return this.hash.project.objects['PBXNativeTarget'];
}

pbxProject.prototype.pbxXCConfigurationList = function () {
    return this.hash.project.objects['XCConfigurationList'];
}

pbxProject.prototype.pbxGroupByName = function (name) {
    return this.pbxItemByComment(name, 'PBXGroup');    
}

pbxProject.prototype.pbxTargetByName = function (name) {
    return this.pbxItemByComment(name, 'PBXNativeTarget');
}

pbxProject.prototype.pbxItemByComment = function (name, pbxSectionName) {
    var section = this.hash.project.objects[pbxSectionName],
        key, itemKey;

    for (key in section) {
        // only look for comments
        if (!COMMENT_KEY.test(key)) continue;

        if (section[key] == name) {
            itemKey = key.split(COMMENT_KEY)[0];
            return section[itemKey];
        }
    }

    return null;
}

pbxProject.prototype.pbxSourcesBuildPhaseObj = function (target) {
    return this.buildPhaseObject('PBXSourcesBuildPhase', 'Sources', target);
}

pbxProject.prototype.pbxResourcesBuildPhaseObj = function (target) {
    return this.buildPhaseObject('PBXResourcesBuildPhase', 'Resources',target);
}

pbxProject.prototype.pbxFrameworksBuildPhaseObj = function (target) {
    return this.buildPhaseObject('PBXFrameworksBuildPhase', 'Frameworks',target);
}

// Find Build Phase from group/target
pbxProject.prototype.buildPhase = function (group,target) {

    if (!target)
        return undefined;

     var nativeTargets = this.pbxNativeTarget();
     if (typeof nativeTargets[target] == "undefined")
        throw new Error("Invalid target: "+target);

     var nativeTarget= nativeTargets[target];
     var buildPhases = nativeTarget.buildPhases;
     for(var i in buildPhases)
     {
        var buildPhase = buildPhases[i];
        if (buildPhase.comment==group)
            return buildPhase.value+"_comment";
     } 
}

pbxProject.prototype.buildPhaseObject = function (name, group,target) {
    var section = this.hash.project.objects[name],
        obj, sectionKey, key;
    var buildPhase = this.buildPhase(group,target);

    for (key in section) {

        // only look for comments
        if (!COMMENT_KEY.test(key)) continue;
  
        // select the proper buildPhase
        if (buildPhase && buildPhase!=key)
            continue;
        if (section[key] == group) {
            sectionKey = key.split(COMMENT_KEY)[0];  
            return section[sectionKey];
        }
     }
    return null;
}


pbxProject.prototype.updateBuildProperty = function(prop, value) {
    var config = this.pbxXCBuildConfigurationSection();
    propReplace(config, prop, value);
}

pbxProject.prototype.updateProductName = function(name) {
    this.updateBuildProperty('PRODUCT_NAME', '"' + name + '"');
}

pbxProject.prototype.removeFromFrameworkSearchPaths = function (file) {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        INHERITED = '"$(inherited)"',
        SEARCH_PATHS = 'FRAMEWORK_SEARCH_PATHS',
        config, buildSettings, searchPaths;
    var new_path = searchPathForFile(file, this);

    for (config in configurations) {
        buildSettings = configurations[config].buildSettings;

        if (unquote(buildSettings['PRODUCT_NAME']) != this.productName)
            continue;

        searchPaths = buildSettings[SEARCH_PATHS];

        if (searchPaths) {
            var matches = searchPaths.filter(function(p) {
                return p.indexOf(new_path) > -1;
            });
            matches.forEach(function(m) {
                var idx = searchPaths.indexOf(m);
                searchPaths.splice(idx, 1);
            });
        }

    }
}

pbxProject.prototype.addToFrameworkSearchPaths = function (file) {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        INHERITED = '"$(inherited)"',
        config, buildSettings, searchPaths;

    for (config in configurations) {
        buildSettings = configurations[config].buildSettings;

        if (unquote(buildSettings['PRODUCT_NAME']) != this.productName)
            continue;

        if (!buildSettings['FRAMEWORK_SEARCH_PATHS']
                || buildSettings['FRAMEWORK_SEARCH_PATHS'] === INHERITED) {
            buildSettings['FRAMEWORK_SEARCH_PATHS'] = [INHERITED];
        }

        buildSettings['FRAMEWORK_SEARCH_PATHS'].push(searchPathForFile(file, this));
    }
}

pbxProject.prototype.removeFromLibrarySearchPaths = function (file) {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        INHERITED = '"$(inherited)"',
        SEARCH_PATHS = 'LIBRARY_SEARCH_PATHS',
        config, buildSettings, searchPaths;
    var new_path = searchPathForFile(file, this);

    for (config in configurations) {
        buildSettings = configurations[config].buildSettings;

        if (unquote(buildSettings['PRODUCT_NAME']) != this.productName)
            continue;

        searchPaths = buildSettings[SEARCH_PATHS];

        if (searchPaths) {
            var matches = searchPaths.filter(function(p) {
                return p.indexOf(new_path) > -1;
            });
            matches.forEach(function(m) {
                var idx = searchPaths.indexOf(m);
                searchPaths.splice(idx, 1);
            });
        }

    }
}

pbxProject.prototype.addToLibrarySearchPaths = function (file) {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        INHERITED = '"$(inherited)"',
        config, buildSettings, searchPaths;

    for (config in configurations) {
        buildSettings = configurations[config].buildSettings;

        if (unquote(buildSettings['PRODUCT_NAME']) != this.productName)
            continue;

        if (!buildSettings['LIBRARY_SEARCH_PATHS']
                || buildSettings['LIBRARY_SEARCH_PATHS'] === INHERITED) {
            buildSettings['LIBRARY_SEARCH_PATHS'] = [INHERITED];
        }

        if (typeof file === 'string') {
            buildSettings['LIBRARY_SEARCH_PATHS'].push(file);
        } else {
            buildSettings['LIBRARY_SEARCH_PATHS'].push(searchPathForFile(file, this));
        }
    }
}

pbxProject.prototype.removeFromHeaderSearchPaths = function (file) {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        INHERITED = '"$(inherited)"',
        SEARCH_PATHS = 'HEADER_SEARCH_PATHS',
        config, buildSettings, searchPaths;
    var new_path = searchPathForFile(file, this);

    for (config in configurations) {
        buildSettings = configurations[config].buildSettings;

        if (unquote(buildSettings['PRODUCT_NAME']) != this.productName)
            continue;

        if (buildSettings[SEARCH_PATHS]) {
            var matches = buildSettings[SEARCH_PATHS].filter(function(p) {
                return p.indexOf(new_path) > -1;
            });
            matches.forEach(function(m) {
                var idx = buildSettings[SEARCH_PATHS].indexOf(m);
                buildSettings[SEARCH_PATHS].splice(idx, 1);
            });
        }

    }
}
pbxProject.prototype.addToHeaderSearchPaths = function (file) {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        INHERITED = '"$(inherited)"',
        config, buildSettings, searchPaths;

    for (config in configurations) {
        buildSettings = configurations[config].buildSettings;

        if (unquote(buildSettings['PRODUCT_NAME']) != this.productName)
            continue;

        if (!buildSettings['HEADER_SEARCH_PATHS']) {
            buildSettings['HEADER_SEARCH_PATHS'] = [INHERITED];
        }

        if (typeof file === 'string') {
            buildSettings['HEADER_SEARCH_PATHS'].push(file);
        } else {
            buildSettings['HEADER_SEARCH_PATHS'].push(searchPathForFile(file, this));
        }
    }
}
// a JS getter. hmmm
pbxProject.prototype.__defineGetter__("productName", function () {
    var configurations = nonComments(this.pbxXCBuildConfigurationSection()),
        config, productName;

    for (config in configurations) {
        productName = configurations[config].buildSettings['PRODUCT_NAME'];

        if (productName) {
            return unquote(productName);
        }
    }
});

// check if file is present
pbxProject.prototype.hasFile = function (filePath) {
    var files = nonComments(this.pbxFileReferenceSection()),
        file, id;
    for (id in files) {
        file = files[id];
        if (file.path == filePath || file.path == ('"' + filePath + '"')) {
            return true;
        }
    }

    return false;
}

// helper recursive prop search+replace
function propReplace(obj, prop, value) {
    var o = {};
    for (var p in obj) {
        if (o.hasOwnProperty.call(obj, p)) {
            if (typeof obj[p] == 'object' && !Array.isArray(obj[p])) {
                propReplace(obj[p], prop, value);
            } else if (p == prop) {
                obj[p] = value;
            }
        }
    }
}

// helper object creation functions
function pbxBuildFileObj(file) {
    var obj = Object.create(null);

    obj.isa = 'PBXBuildFile';
    obj.fileRef = file.fileRef;
    obj.fileRef_comment = file.basename;
    if (file.settings) obj.settings = file.settings;

    return obj;
}

function pbxFileReferenceObj(file) {
    var obj = Object.create(null);

    obj.isa = 'PBXFileReference';
    obj.lastKnownFileType = file.lastType;
    
    obj.name = "\"" + file.basename + "\"";
    obj.path = "\"" + file.path.replace(/\\/g, '/') + "\"";
    
    obj.sourceTree = file.sourceTree;

    if (file.fileEncoding)
        obj.fileEncoding = file.fileEncoding;
        
    if (file.explicitFileType)
        obj.explicitFileType = file.explicitFileType;
        
    if ('includeInIndex' in file)
        obj.includeInIndex = file.includeInIndex;

    return obj;
}

function pbxGroupChild(file) {
    var obj = Object.create(null);

    obj.value = file.fileRef;
    obj.comment = file.basename;

    return obj;
}

function pbxBuildPhaseObj(file) {
    var obj = Object.create(null);

    obj.value = file.uuid;
    obj.comment = longComment(file);

    return obj;
}

function pbxBuildFileComment(file) {
    return longComment(file);
}

function pbxFileReferenceComment(file) {
    return file.basename;
}

function longComment(file) {
    return f("%s in %s", file.basename, file.group);
}

// respect <group> path
function correctForPluginsPath(file, project) {
    return correctForPath(file, project, 'Plugins');
}

function correctForResourcesPath(file, project) {
    return correctForPath(file, project, 'Resources');
}

function correctForFrameworksPath(file, project) {
    return correctForPath(file, project, 'Frameworks');
}

function correctForPath(file, project, group) {
    var r_group_dir = new RegExp('^' + group + '[\\\\/]');

    if (project.pbxGroupByName(group).path)
        file.path = file.path.replace(r_group_dir, '');

    return file;
}

function searchPathForFile(file, proj) {
    var plugins = proj.pbxGroupByName('Plugins'),
        pluginsPath = plugins ? plugins.path : null,
        fileDir = path.dirname(file.path);

    if (fileDir == '.') {
        fileDir = '';
    } else {
        fileDir = '/' + fileDir;
    }

    if (file.plugin && pluginsPath) {
        return '"\\"$(SRCROOT)/' + unquote(pluginsPath) + '\\""';
    } else if (file.customFramework && file.dirname) {
        return '"\\"' + file.dirname + '\\""';
    } else {
        return '"\\"$(SRCROOT)/' + proj.productName + fileDir + '\\""';
    }
}

function nonComments(obj) {
    var keys = Object.keys(obj),
        newObj = {}, i = 0;

    for (i; i < keys.length; i++) {
        if (!COMMENT_KEY.test(keys[i])) {
            newObj[keys[i]] = obj[keys[i]];
        }
    }

    return newObj;
}

function unquote(str) {
    if (str) return str.replace(/^"(.*)"$/, "$1");
}

module.exports = pbxProject;
