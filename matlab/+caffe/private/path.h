#ifndef _PATH_H_
#define _PATH_H_

#include <stdlib.h>
#include <string>

using std::string;

class CPath
{
#ifndef MAX_PATH
#define MAX_PATH _MAX_PATH
#endif

public:
	static string GetFileName(string strPath)
	{
		char pDrive[MAX_PATH], pDir[MAX_PATH], pFileName[MAX_PATH], pExt[MAX_PATH];
		char pOutput[MAX_PATH];
		::_splitpath_s(strPath.c_str(), pDrive, MAX_PATH, pDir, MAX_PATH, pFileName, MAX_PATH, pExt, MAX_PATH);
		sprintf_s(pOutput, MAX_PATH, "%s%s", pFileName, pExt);
		return string(pOutput);
	}


	static string GetFileNameWithoutExtension(string strPath)
	{
		char pDrive[MAX_PATH], pDir[MAX_PATH], pFileName[MAX_PATH], pExt[MAX_PATH];
		::_splitpath_s(strPath.c_str(), pDrive, MAX_PATH, pDir, MAX_PATH, pFileName, MAX_PATH, pExt, MAX_PATH);
		return string(pFileName);
	}


	static string GetDirectoryName(string strPath)
	{
		char pDrive[MAX_PATH], pDir[MAX_PATH], pFileName[MAX_PATH], pExt[MAX_PATH];
		char pOutput[MAX_PATH];
		::_splitpath_s(strPath.c_str(), pDrive, MAX_PATH, pDir, MAX_PATH, pFileName, MAX_PATH, pExt, MAX_PATH);
		sprintf_s(pOutput, MAX_PATH, "%s%s", pDrive, pDir);
		return string(pOutput);
	}


	static string GetExtension(string strPath)
	{
		char pDrive[MAX_PATH], pDir[MAX_PATH], pFileName[MAX_PATH], pExt[MAX_PATH];
		::_splitpath_s(strPath.c_str(), pDrive, MAX_PATH, pDir, MAX_PATH, pFileName, MAX_PATH, pExt, MAX_PATH);
		return string(pExt);
	}
};


#endif
