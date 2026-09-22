#Requires -Version 5.1
<#
.SYNOPSIS
    在 Windows 上打服务器发布包：speech-assist-YYYYMMDD.tar.gz

.DESCRIPTION
    本仓库不是 git 仓库（没有 .git），服务器上无 clone 来源，
    所以发布链路只能是「本地打包 -> scp 上传 -> 解压 -> install.sh/update.sh」。

    白名单打包，只带运行需要的东西：
        app/  deploy/  docs/  tests/(可选)  requirements.txt  评价标准.txt  .env.example

    刻意排除（这些绝不能进包）：
        api_key.txt    千问密钥，进包等于把密钥上传到公网服务器
        .env           本地开发配置（含 dev SECRET_KEY / admin123）
        data/          本地用户数据、视频、speech.db
        __pycache__/ *.pyc *.log logs_*.txt 以及临时脚本

    另外做两件容易翻车的事：
        1) 把 deploy/ 下的 .sh / .service / .conf / env.production 换行符归一为 LF。
           Windows 上写的 bash 脚本带 CRLF，在 Linux 上会直接报
           "/usr/bin/env: bash\r: No such file or directory"。
        2) 打完包再解压一次做回环校验，确认中文文件名（评价标准.txt）没被编错码。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\package.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\package.ps1 -OutDir D:\release -SkipTests
#>
[CmdletBinding()]
param(
    [string]$OutDir = '',
    [switch]$SkipTests,
    [string]$Stamp = ''
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist' }
if (-not $Stamp) { $Stamp = Get-Date -Format 'yyyyMMdd' }

$leaf      = "speech-assist-$Stamp"
$stageRoot = Join-Path $env:TEMP ("speech-assist-pkg-" + [Guid]::NewGuid().ToString('N'))
$stage     = Join-Path $stageRoot $leaf

$Dirs  = @('app', 'deploy', 'docs')
$Files = @('requirements.txt', '评价标准.txt', '.env.example')
if (-not $SkipTests) { $Dirs += 'tests' }

try {
    Write-Host "[打包] 源目录 $repoRoot" -ForegroundColor Green
    New-Item -ItemType Directory -Path $stage | Out-Null

    foreach ($d in $Dirs) {
        $src = Join-Path $repoRoot $d
        if (-not (Test-Path $src)) { throw "缺少 $d，源目录看起来不完整，先别打包。" }
        Copy-Item -Path $src -Destination $stage -Recurse -Force
    }
    foreach ($f in $Files) {
        $src = Join-Path $repoRoot $f
        if (-not (Test-Path $src)) { throw "缺少 $f —— 它是运行必需的。" }
        Copy-Item -Path $src -Destination (Join-Path $stage $f) -Force
    }

    # -------------------------------------------------- 清除编译产物与本地垃圾
    Get-ChildItem $stage -Recurse -Force -Directory |
        Where-Object { $_.Name -in @('__pycache__', '.pytest_cache', '.mypy_cache', '.idea', '.vscode') } |
        Remove-Item -Recurse -Force

    Get-ChildItem $stage -Recurse -Force -File |
        Where-Object { $_.Extension -in @('.pyc', '.pyo', '.log', '.tmp') -or $_.Name -like 'logs_*.txt' } |
        Remove-Item -Force

    # -------------------------------------------------- deploy/ 模板换行符归一为 LF
    $deployStage = Join-Path $stage 'deploy'
    $noBom = New-Object System.Text.UTF8Encoding($false)
    Get-ChildItem $deployStage -File |
        Where-Object { $_.Extension -eq '.sh' -or $_.Name -in @('speech-assist.service', 'nginx-speech-assist.conf', 'env.production') } |
        ForEach-Object {
            $txt = [System.IO.File]::ReadAllText($_.FullName)
            $txt = $txt -replace "`r`n", "`n"
            [System.IO.File]::WriteAllText($_.FullName, $txt, $noBom)
            Write-Host "[LF] deploy/$($_.Name)"
        }

    # -------------------------------------------------- 密钥/数据外泄断言
    $leaked = Get-ChildItem $stage -Recurse -Force -File | Where-Object {
        $_.Name -eq 'api_key.txt' -or $_.Name -eq '.env' -or $_.Name -eq 'speech.db' -or
        $_.Name -like '*.wav' -or $_.Name -like 'speech.db-*'
    }
    if ($leaked) {
        throw "包里混进了不该上传的文件：`n" + (($leaked | ForEach-Object { '  ' + $_.FullName.Substring($stage.Length) }) -join "`n")
    }
    if (Test-Path (Join-Path $stage 'data')) { throw '包里含 data/ 目录，服务器上会盖掉真实用户数据。' }

    # -------------------------------------------------- 生成 tar.gz
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $tar) { throw '找不到 tar.exe（Win10 1803+ 自带）。也可以手动压成 zip 后改用 unzip 部署。' }

    if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
    $archive = Join-Path $OutDir "$leaf.tar.gz"
    if (Test-Path $archive) { Remove-Item $archive -Force }

    Write-Host "[打包] -> $archive" -ForegroundColor Green
    & $tar.Source -czf $archive -C $stageRoot $leaf
    if ($LASTEXITCODE -ne 0) { throw "tar 打包失败，退出码 $LASTEXITCODE" }

    # -------------------------------------------------- 回环校验：解压一次确认内容完整
    $verify = Join-Path $stageRoot '_verify'
    New-Item -ItemType Directory -Path $verify | Out-Null
    & $tar.Source -xzf $archive -C $verify
    if ($LASTEXITCODE -ne 0) { throw 'tar 解压校验失败。' }

    foreach ($need in @('app/main.py', 'app/static/style.css', 'deploy/install.sh', 'requirements.txt')) {
        if (-not (Test-Path (Join-Path $verify (Join-Path $leaf $need)))) { throw "包内缺少 $need" }
    }
    $rubricInPkg = Get-ChildItem $verify -Recurse -File | Where-Object { $_.Name -eq '评价标准.txt' }
    if (-not $rubricInPkg) { throw '包内找不到 评价标准.txt —— 中文文件名在打包过程中被编码破坏了，服务器上会以 mock 之外的方式直接启动失败。' }

    $shWithCr = @()
    Get-ChildItem (Join-Path $verify (Join-Path $leaf 'deploy')) -File | Where-Object { $_.Extension -eq '.sh' } | ForEach-Object {
        if ([System.IO.File]::ReadAllBytes($_.FullName) -contains 13) { $shWithCr += $_.Name }
    }
    if ($shWithCr) { throw "这些脚本仍带 CR，bash 会拒绝执行：$($shWithCr -join ', ')" }

    $count = (Get-ChildItem $verify -Recurse -File | Measure-Object).Count
    $size  = [math]::Round((Get-Item $archive).Length / 1MB, 2)
    Write-Host ''
    Write-Host '[完成] 发布包已生成并通过回环校验' -ForegroundColor Green
    Write-Host "  文件    : $archive"
    Write-Host "  大小    : $size MB，共 $count 个文件"
    Write-Host "  顶层目录: $leaf"
    Write-Host ''
    Write-Host '下一步（上传 + 安装）：' -ForegroundColor Yellow
    Write-Host "  scp `"$archive`" root@<公网IP>:/root/"
    Write-Host '  ssh root@<公网IP>'
    Write-Host "  tar xzf $leaf.tar.gz && sudo bash $leaf/deploy/install.sh"
    Write-Host '完整步骤见 deploy/部署手册.md'
}
finally {
    if (Test-Path $stageRoot) { Remove-Item $stageRoot -Recurse -Force -ErrorAction SilentlyContinue }
}
