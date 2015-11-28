@echo off
REM Thanks to Jerold Scheulamn <http://www.windowsitpro.com/article/internet/jsi-tip-8332-how-can-i-use-the-command-line-to-determine-the-version-of-internet-explorer-that-is-installed->
setlocal
set qry=reg query "HKEY_LOCAL_MACHINE\Software\Microsoft\Internet Explorer" /v Version
set fnd=findstr /I /L /C:"REG_SZ"
for /f "Tokens=2*" %%u in ('%qry%^|%fnd%') do (
 @echo %%v
)
endlocal