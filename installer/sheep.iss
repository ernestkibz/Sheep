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

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
  Ok: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Getting what Sheep needs. This can take several minutes and needs the internet...';
    WizardForm.ProgressGauge.Style := npbstMarquee;
    Ok := ExecAsOriginalUser(ExpandConstant('{cmd}'), '/c ""' + ExpandConstant('{app}\Sheep.cmd') + '" prepare"',
                             ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code);
    WizardForm.ProgressGauge.Style := npbstNormal;
    if (not Ok or (Code <> 0)) and not WizardSilent then
      MsgBox('Sheep is installed, but it could not get everything it needs yet. This usually means there is no internet.'
             + #13#10#13#10 + 'Connect to the internet, then run this installer again, or just open Sheep: it finishes the setup by itself.',
             mbInformation, MB_OK);
  end;
end;

[Run]
Filename: "{app}\Sheep.cmd"; Description: "Launch Sheep now"; Flags: nowait postinstall shellexec skipifsilent
