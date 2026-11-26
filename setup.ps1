param([string]$Python = 'py')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Install-OnnxRuntime([string]$Destination) {
    $version = '1.20.1'
    $packageId = 'microsoft.ml.onnxruntime'
    $uri = "https://api.nuget.org/v3-flatcontainer/$packageId/$version/$packageId.$version.nupkg"
    $package = Join-Path ([IO.Path]::GetTempPath()) ("objectkey-ort-{0}.nupkg" -f [guid]::NewGuid())
    $dotnet = Get-Command dotnet.exe -CommandType Application -ErrorAction SilentlyContinue
    if (!$dotnet) { throw 'Install the .NET SDK to verify the ONNX Runtime NuGet package.' }

    try {
        Write-Host "Downloading Microsoft.ML.OnnxRuntime $version from NuGet..."
        Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $package
        & $dotnet.Source nuget verify $package --all
        if ($LASTEXITCODE -ne 0) { throw 'ONNX Runtime NuGet signature verification failed.' }

        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [IO.Compression.ZipFile]::OpenRead($package)
        try {
            $include = @('build/native/include/', 'buildTransitive/native/include/') |
                Where-Object { $archive.GetEntry($_ + 'onnxruntime_cxx_api.h') } |
                Select-Object -First 1
            $native = 'runtimes/win-x64/native/'
            $importLib = $archive.GetEntry($native + 'onnxruntime.lib')
            $dll = $archive.GetEntry($native + 'onnxruntime.dll')
            if (!$include -or !$importLib -or !$dll) { throw 'Windows x64 SDK files are missing from the NuGet package.' }

            $files = [ordered]@{}
            foreach ($entry in $archive.Entries) {
                if ($entry.FullName.StartsWith($include) -and $entry.Name) {
                    $files['include/' + $entry.FullName.Substring($include.Length)] = $entry
                } elseif ($entry.FullName -match '^(LICENSE(\.txt)?|ThirdPartyNotices\.txt)$') {
                    $files[$entry.Name] = $entry
                }
            }
            $files['lib/onnxruntime.lib'] = $importLib
            $files['lib/onnxruntime.dll'] = $dll
            foreach ($relative in $files.Keys) {
                $target = [IO.Path]::GetFullPath((Join-Path $Destination $relative))
                if (!$target.StartsWith($Destination + '\', [StringComparison]::OrdinalIgnoreCase)) {
                    throw "Invalid package path: $relative"
                }
                [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
                $tempFile = "$target.$([guid]::NewGuid()).tmp"
                try {
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($files[$relative], $tempFile)
                    Move-Item -LiteralPath $tempFile -Destination $target -Force
                } finally {
                    if (Test-Path -LiteralPath $tempFile) { Remove-Item -LiteralPath $tempFile }
                }
            }
        } finally { $archive.Dispose() }
    } finally {
        if (Test-Path -LiteralPath $package) { Remove-Item -LiteralPath $package }
    }
}

Push-Location $root
try {
    New-Item -ItemType Directory -Force deps,models | Out-Null
    if (!(Test-Path 'deps/onnxruntime-win-x64-1.20.1/lib/onnxruntime.lib') -or !(Test-Path 'deps/onnxruntime-win-x64-1.20.1/lib/onnxruntime.dll') -or !(Test-Path 'deps/onnxruntime-win-x64-1.20.1/include/onnxruntime_cxx_api.h')) {
        Install-OnnxRuntime (Join-Path $root 'deps\onnxruntime-win-x64-1.20.1')
    }
    if (!(Test-Path '.venv/Scripts/python.exe')) {
        $localPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
        if (!$PSBoundParameters.ContainsKey('Python') -and (Test-Path -LiteralPath $localPython -PathType Leaf)) {
            $Python = $localPython
        }
        if ($Python -eq 'py') { & py -3.10 -m venv .venv } else { & $Python -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 is required; pass -Python with an executable path.' }
    }
    & .venv/Scripts/python.exe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & .venv/Scripts/python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    & .venv/Scripts/python.exe -m pip install onnx==1.17.0 numpy==2.2.6
    if ($LASTEXITCODE -ne 0) { throw 'ONNX installation failed.' }
    & .venv/Scripts/python.exe config/setup/export-resnet18.py
    if ($LASTEXITCODE -ne 0) { throw 'Model export failed.' }
    & .venv/Scripts/python.exe config/setup/export-lightglue.py
    if ($LASTEXITCODE -ne 0) { throw 'LightGlue export failed.' }
} finally { Pop-Location }
