param([string]$Python = 'py')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Push-Location $root
try {
    New-Item -ItemType Directory -Force deps,models | Out-Null
    if (!(Test-Path 'deps/onnxruntime-win-x64-1.20.1/lib/onnxruntime.lib') -or !(Test-Path 'deps/onnxruntime-win-x64-1.20.1/lib/onnxruntime.dll') -or !(Test-Path 'deps/onnxruntime-win-x64-1.20.1/include/onnxruntime_cxx_api.h')) {
        Invoke-WebRequest 'https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-1.20.1.zip' -OutFile deps/onnxruntime.zip
        $digest = (Get-FileHash -LiteralPath deps/onnxruntime.zip -Algorithm SHA256).Hash
        if ($digest -ne '78D447051E48BD2E1E778BBA378BEC4ECE11191C9E538CF7B2C4A4565E8F5581') { throw 'ONNX Runtime archive checksum mismatch.' }
        Expand-Archive deps/onnxruntime.zip -DestinationPath deps -Force
    }
    if (!(Test-Path '.venv/Scripts/python.exe')) {
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
