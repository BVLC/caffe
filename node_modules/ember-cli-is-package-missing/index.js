'use strict';

/*
 *
 * Given the context and library that's supposed to be looked up in the package.json,
 * this method detects if its already available.
 *
 * @method isPackageMissing
 * @param context The context of the method its called from (ie., this)
 * @param packageName The package that's supposed to be looked up
 * @return Boolean
 *
 */
module.exports = function (context, packageName) {
  var pkgContent = context.project.pkg;
  var isAvailableInDevDependency = pkgContent.devDependencies && pkgContent.devDependencies[packageName];
  var isAvailableInDependency = pkgContent.dependencies && pkgContent.dependencies[packageName];
  return !(isAvailableInDevDependency || isAvailableInDependency);
};
