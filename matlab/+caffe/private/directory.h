#ifndef _DIRECTORY_H_
#define _DIRECTORY_H_

#include <windows.h>
#include <vector>
#include <string>
#include <io.h>

using std::vector;
using std::string;

class CDirectory
{
public:
	static vector<string> GetFiles(const char *strFolder, const char *strFilter, bool bAllDirectories)
	{
		vector<string> vec = GetFilesInOneFolder(strFolder, strFilter);
		if (bAllDirectories)
		{
			vector<string> vecSubFolders = GetDirectories(strFolder, "*", true);
			for (size_t i = 0; i < vecSubFolders.size(); i++)
			{
				vector<string> vecFiles = GetFilesInOneFolder(vecSubFolders[i].c_str(), strFilter);
				for (size_t j = 0; j < vecFiles.size(); j++)
					vec.push_back(vecFiles[j]);
			}
		}
		return vec;
	}

	static vector<string> GetDirectories(const char *strFolder, const char *strFilter, bool bAllDirectories)
	{
		vector<string> vec = GetDirectoryInOnFolder(strFolder, strFilter);
		if (vec.size() == 0)
			return vec;
		if (bAllDirectories)
		{
			vector<string> vecSubFolder;
			for (size_t i = 0; i < vec.size(); i++)
			{
				vector<string> vecSub = GetDirectories(vec[i].c_str(), strFilter, bAllDirectories);
				for (size_t j = 0; j < vecSub.size(); j++)
				{
					vecSubFolder.push_back(vecSub[j]);
				}
			}
			for (size_t i = 0; i < vecSubFolder.size(); i++)
				vec.push_back(vecSubFolder[i]);
		}
		return vec;
	}

	static string GetCurrentDirectory()
	{
		char strPath[MAX_PATH];
		::GetCurrentDirectoryA(MAX_PATH, strPath);
		return string(strPath);
	}


	static bool Exist(const char *strPath)
	{
		return (_access(strPath, 0) == 0);
	}


	static bool CreateDirectory(const char *strPath)
	{
		if (Exist(strPath))
			return false;
		char strFolder[MAX_PATH] = {0};
		size_t len = strlen(strPath);
		for (size_t i = 0; i <= len; i++)
		{
			if (strPath[i] == '\\' || strPath[i] == '/' || strPath[i] == '\0')
			{
				if (!Exist(strFolder))
				{
					if(::CreateDirectoryA(strFolder, NULL) == 0)
						return false;
				}
			}
			strFolder[i] = strPath[i];
		}
		return true;
	}


private:
	static vector<string> GetFilesInOneFolder(const char *strFolder, const char *strFilter)
	{
		vector<string> vec;
		char strFile[MAX_PATH] = {'\0'};
		sprintf_s(strFile, MAX_PATH, "%s\\%s", strFolder, strFilter);
		WIN32_FIND_DATAA FindFileData;
		HANDLE hFind = ::FindFirstFileA(strFile, &FindFileData);
		if (INVALID_HANDLE_VALUE == hFind) 
			return vec;
		do
		{
			if (FILE_ATTRIBUTE_DIRECTORY == (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				continue;
			char strName[MAX_PATH];
			sprintf_s(strName, MAX_PATH, "%s\\%s", strFolder, FindFileData.cFileName);
			vec.push_back(strName);
		}while (::FindNextFileA(hFind, &FindFileData));
		::FindClose(hFind);
		return vec;
	}

	static vector<string> GetDirectoryInOnFolder(const char *strFolder, const char *strFilter)
	{
		vector<string> vec;
		char strFile[MAX_PATH] = {'\0'};
		sprintf_s(strFile, MAX_PATH, "%s\\%s", strFolder, strFilter);
		WIN32_FIND_DATAA FindFileData;
		HANDLE hFind = ::FindFirstFileA(strFile, &FindFileData);
		if (INVALID_HANDLE_VALUE == hFind) 
			return vec;
		do
		{
			if (FILE_ATTRIBUTE_DIRECTORY != (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				continue;
			if (strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
				continue;
			char strName[MAX_PATH];
			sprintf_s(strName, MAX_PATH, "%s\\%s", strFolder, FindFileData.cFileName);
			vec.push_back(strName);
		}while (::FindNextFileA(hFind, &FindFileData));
		::FindClose(hFind);
		return vec;
	}
};


#endif