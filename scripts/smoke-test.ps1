param([string]$Configuration = 'Debug', [string]$Platform = 'x64')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo "bin/$Platform/$Configuration/ObjectKEY.exe"
if (!(Test-Path -LiteralPath $exe)) { throw "Build the project first: $exe" }
$fixtureDir = Join-Path $repo 'results/smoke-fixtures'
New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null
Add-Type -AssemblyName System.Drawing
$bitmap = [System.Drawing.Bitmap]::new(13, 7)
try {
    $bitmap.SetPixel(0, 0, [System.Drawing.Color]::Red)
    $bitmap.Save((Join-Path $fixtureDir '한글 image.png'), [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Save((Join-Path $fixtureDir 'sample.jpg'), [System.Drawing.Imaging.ImageFormat]::Jpeg)
    $bitmap.Save((Join-Path $fixtureDir 'sample.bmp'), [System.Drawing.Imaging.ImageFormat]::Bmp)
} finally { $bitmap.Dispose() }
[IO.File]::WriteAllText((Join-Path $fixtureDir 'corrupt.png'), 'not an image')
[IO.File]::WriteAllText((Join-Path $fixtureDir 'manifest.txt'), "# input fixtures`r`n한글 image.png`r`nsample.jpg`r`nsample.bmp`r`n", [Text.UTF8Encoding]::new($true))
[IO.File]::WriteAllText((Join-Path $fixtureDir 'empty.txt'), '')
[IO.File]::WriteAllText((Join-Path $fixtureDir 'invalid.txt'), "missing.png`n")
function Assert-Run([string[]]$Arguments, [int]$ExpectedCode, [string]$ExpectedText) {
    # Windows PowerShell treats native stderr as ErrorRecord; expected failures are checked below.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = (& $exe @Arguments 2>&1 | Out-String)
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($code -ne $ExpectedCode -or !$output.Contains($ExpectedText)) {
        throw "Failed [$Arguments]: code=$code, expected=$ExpectedCode`n$output"
    }
    Write-Host "PASS: $Arguments"
}
Assert-Run @('--help') 0 'Usage:'
foreach ($name in @('한글 image.png','sample.jpg','sample.bmp')) {
    Assert-Run @('inspect', (Join-Path $fixtureDir $name)) 0 'width=13 height=7 channels=3 format=BGR8 bytes=273'
}
Assert-Run @('inspect', (Join-Path $fixtureDir 'missing.png')) 2 'INPUT_ERROR'
Assert-Run @('inspect', (Join-Path $fixtureDir 'corrupt.png')) 2 'INPUT_ERROR'
Assert-Run @('enroll', (Join-Path $fixtureDir 'sample.bmp')) 3 'No template saved'
Assert-Run @('verify', (Join-Path $fixtureDir 'sample.bmp')) 3 'decision=NOT_EVALUATED'
Assert-Run @('evaluate', (Join-Path $fixtureDir 'manifest.txt')) 3 'samples=3 failures=0'
Assert-Run @('evaluate', (Join-Path $fixtureDir 'empty.txt')) 2 'INPUT_ERROR'
Assert-Run @('evaluate', (Join-Path $fixtureDir 'invalid.txt')) 2 'failures=1'
Assert-Run @('unknown', 'dummy') 2 'unknown command'
Assert-Run @('verify') 2 'Usage:'
Write-Host 'All 13 smoke checks passed.'
