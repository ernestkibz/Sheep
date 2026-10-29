; Inno Setup script for Sheep
; Run BUILD_INSTALLER.ps1 first, then:  iscc installer\sheep.iss
;
; The 3B .ssf downloads on first run, not in the installer.

[Setup]
AppName=Sheep
AppVersion=1.0
AppPublisher=Sheep
DefaultDirName={autopf}\Sheep
DefaultGroupName=Sheep
OutputBaseFilename=Sheep-Setup
Compression=lzma2/max
SolidCompression=yes
UninstallDisplayIcon={app}\sheep-mono-20261010.ico
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\build\Sheep\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Sheep"; Filename: "{app}\Sheep.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\sheep-mono-20261010.ico"
Name: "{commondesktop}\Sheep"; Filename: "{app}\Sheep.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\sheep-mono-20261010.ico"
Name: "{group}\Uninstall Sheep"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Sheep.cmd"; Description: "Launch Sheep now"; Flags: nowait postinstall shellexec skipifsilent
