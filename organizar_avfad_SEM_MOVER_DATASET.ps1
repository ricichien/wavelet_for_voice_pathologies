#requires -Version 5.1
<#
Organiza os arquivos existentes SEM copiar dados nem mexer nos ZIPs originais.
  PREVIA:   powershell -ExecutionPolicy Bypass -File .\organizar_avfad_SEM_MOVER_DATASET.ps1
  LEVE:     powershell -ExecutionPolicy Bypass -File .\organizar_avfad_SEM_MOVER_DATASET.ps1 -Leve
  COMPLETA (somente apos a extracao finalizar):
            powershell -ExecutionPolicy Bypass -File .\organizar_avfad_SEM_MOVER_DATASET.ps1 -Aplicar
IMPORTANTE: execute a partir da copia local do repositorio.
#>
[CmdletBinding()]
param([switch]$Aplicar, [switch]$Leve)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$Root=[System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $Root '.git'))) {
    throw 'Por seguranca, execute este script a partir da raiz do repositorio.'
}
if ($Leve -and $Aplicar) { throw 'Escolha somente -Leve ou -Aplicar.' }

# Verificar apenas os NOMES dos cinco ZIPs originais; nao abre, move nem calcula hash deles.
$ProtectedZips=@(
 'AVFAD_01_00_00_1_README.zip',
 'AVFAD_01_00_00_2_A_to_C.zip',
 'AVFAD_01_00_00_3_D_to_L.zip',
 'AVFAD_01_00_00_4_M.zip',
 'AVFAD_01_00_00_5_N_to_Z.zip'
)
foreach ($name in $ProtectedZips) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $name) -PathType Leaf)) {
        throw "ZIP protegido nao encontrado na raiz: $name. Nada sera movido."
    }
}

$Payload=Join-Path $Root 'codigo_novo\src'
$script:Plan=New-Object System.Collections.ArrayList
$script:Moved=New-Object System.Collections.ArrayList
function Add-Move([string]$src,[string]$dest) {
    if ((Test-Path -LiteralPath $src) -and -not $src.Equals($dest,[StringComparison]::OrdinalIgnoreCase)) {
        [void]$script:Plan.Add([pscustomobject]@{Source=$src;Destination=$dest})
    }
}
function File-Target([System.IO.FileInfo]$file) {
    $name=$file.Name
    $ext=$file.Extension.ToLowerInvariant()
    if ($ProtectedZips -contains $name -or $name -match '^AVFAD_01_00_00_.*\.zip$') {return $null}
    if ($name -in @('.gitignore','README.md','requirements.txt','requirements_scattering.txt','organizar_avfad_SEM_MOVER_DATASET.ps1')) {return $null}
    if ($ext -eq '.py') {return (Join-Path $Root ('backups\codigo_original\'+$name))}
    if ($ext -eq '.ps1') {return (Join-Path $Root ('backups\scripts_antigos\'+$name))}
    if ($ext -eq '.zip') {return (Join-Path $Root ('backups\pacotes_zip\'+$name))}
    if ($ext -eq '.npz' -or ($ext -eq '.csv' -and $name -match '^(features_|metadata_|validacao_)')) {
        return (Join-Path $Root ('data\processed\'+$name))
    }
    if ($ext -in @('.sqlite','.sqlite-wal','.sqlite-shm')) {
        return (Join-Path $Root ('checkpoints\'+$name))
    }
    if ($ext -in @('.csv','.txt','.md','.png','.json')) {
        return (Join-Path $Root ('results\legacy\'+$name))
    }
    return $null # Tipos desconhecidos: preservados onde estao
}
function Folder-Target([System.IO.DirectoryInfo]$dir) {
    $name=$dir.Name
    if ($name -in @('.venv','.git','backups','docs','data','results','checkpoints','src','codigo_novo','codigo','__pycache__')) {return $null}
    if ($name -like 'resultados_*') {return (Join-Path $Root ('results\experiments\'+$name))}
    if ($name -like 'checkpoint_*') {return (Join-Path $Root ('checkpoints\'+$name))}
    if ($name -eq 'metadata') {return (Join-Path $Root 'data\metadata')}
    if ($name -eq 'saida') {return (Join-Path $Root 'results\legacy\saida')}
    return $null # Diretorios desconhecidos nao sao movidos
}

if ($Leve) {
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -File)) {
        if ($ProtectedZips -contains $file.Name) {continue}
        if ($file.Extension.ToLowerInvariant() -eq '.zip' -and $file.Name -notmatch '^AVFAD_01_00_00_.*\.zip$') {
            Add-Move $file.FullName (Join-Path $Root ('backups\pacotes_zip\'+$file.Name))
        }
        if ($file.Extension.ToLowerInvariant() -eq '.md' -and $file.Name -match '^(LEIA_|README_|05_DIAGNOSTICO_LEIA)') {
            Add-Move $file.FullName (Join-Path $Root ('docs\guias\'+$file.Name))
        }
    }
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $Payload 'avfad_paths.py'))) {
        throw 'Nao encontrei codigo_novo/src/avfad_paths.py. Extraia TODO o pacote na raiz AVFAD.'
    }
    if (Test-Path -LiteralPath (Join-Path $Root 'src')) {
        throw 'src/ ja existe. Nao sobrescreverei uma migracao anterior.'
    }
    $origins=@(Get-ChildItem -LiteralPath $Root -File -Filter '*.py' |
        Where-Object {$_.Name -notlike '*.bak.py'} | Select-Object -ExpandProperty Name)
    $replacements=@(Get-ChildItem -LiteralPath $Payload -File -Filter '*.py' |
        Select-Object -ExpandProperty Name)
    $notCovered=@($origins | Where-Object {$replacements -notcontains $_})
    if ($notCovered.Count -gt 0) {
        throw "Codigo novo nao contem os scripts atuais: $($notCovered -join ', '). Nada foi movido."
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -File)) {
        $dest=File-Target $file
        if ($null -ne $dest) {Add-Move $file.FullName $dest}
    }
    foreach ($dir in @(Get-ChildItem -LiteralPath $Root -Directory)) {
        $dest=Folder-Target $dir
        if ($null -ne $dest) {Add-Move $dir.FullName $dest}
    }
    Add-Move $Payload (Join-Path $Root 'src')
    Add-Move (Join-Path $Root 'codigo_novo') (Join-Path $Root 'backups\refatoracao_aplicada')
    if (Test-Path -LiteralPath (Join-Path $Root 'codigo')) {
        Add-Move (Join-Path $Root 'codigo') (Join-Path $Root 'backups\refatoracao_anterior')
    }
}

# Antes de mudar qualquer coisa, checar colisoes de nomes e de subarvores.
$seen=@{}
foreach ($op in $script:Plan) {
    if (Test-Path -LiteralPath $op.Destination) {
        throw "Conflito: $($op.Destination) ja existe. Nada movido."
    }
    $lower=$op.Destination.ToLowerInvariant()
    if ($seen.ContainsKey($lower)) {throw "Duas origens para o mesmo destino: $($op.Destination)"}
    $seen[$lower]=$true
    foreach ($protected in $ProtectedZips) {
        if ($op.Source.Equals((Join-Path $Root $protected),[StringComparison]::OrdinalIgnoreCase) -or
            $op.Destination.Equals((Join-Path $Root $protected),[StringComparison]::OrdinalIgnoreCase)) {
            throw 'Plano tentou alterar um ZIP protegido. Migracao bloqueada.'
        }
    }
}

Write-Host ''
Write-Host '=== AVFAD ORGANIZACAO SEM MOVER DATASET ==='
Write-Host "Raiz: $Root"
Write-Host 'Os CINCO ZIPs originais permanecem na raiz, nos mesmos caminhos.'
Write-Host 'Nenhum ZIP sera extraido. Nenhuma matriz sera duplicada.'
Write-Host "Operacoes planejadas: $($script:Plan.Count)"
foreach ($op in $script:Plan) {
    $a=$op.Source.Substring($Root.Length).TrimStart('\')
    $b=$op.Destination.Substring($Root.Length).TrimStart('\')
    Write-Host "  $a -> $b"
}
if (-not $Aplicar -and -not $Leve) {
    Write-Host '\nPREVIA SOMENTE LEITURA. NADA ALTERADO.'
    Write-Host 'So rode -Aplicar depois que Scattering terminar e nenhum Python estiver executando.'
    exit 0
}
if ($Aplicar) {
    # Conservador: bloquear caso exista QUALQUER processo Python, mesmo de outro projeto.
    $active=@(Get-CimInstance Win32_Process -ErrorAction Stop |
        Where-Object {$_.Name -in @('python.exe','pythonw.exe','py.exe','python3.exe')})
    if ($active.Count -gt 0) {
        $active | Select-Object ProcessId,Name,CommandLine | Format-Table -Wrap
        throw "MIGRACAO BLOQUEADA: ha $($active.Count) processo(s) Python ativo(s)."
    }
}

try {
    foreach ($op in $script:Plan) {
        $parent=Split-Path -Parent $op.Destination
        if (-not (Test-Path -LiteralPath $parent)) {New-Item -ItemType Directory -Path $parent -Force | Out-Null}
        Move-Item -LiteralPath $op.Source -Destination $op.Destination -ErrorAction Stop
        [void]$script:Moved.Add($op)
    }
    if ($Aplicar) {
        $ignore=Join-Path $Root '.gitignore'
        $rules=@(
            '# Dados, audios, matrizes e resultados: NAO COMMITAR',
            '/.venv/','/.git/','/backups/','/codigo/','/codigo_novo/',
            '/data/','/results/','/checkpoints/','/metadata/','/saida/',
            '/resultados_*/','/checkpoint_*/','/docs/guias/',
            '**/__pycache__/','*.pyc','*.zip','*.npz','*.sqlite*',
            '*.csv','*.wav','*.prt','*.xlsx','*.pkl','*.joblib',
            '*.db','*.key','*.pem','.env','*.log','*.tmp','*.bak','*.bak.py'
        )
        $existing=if (Test-Path -LiteralPath $ignore) {@(Get-Content -LiteralPath $ignore)} else {@()}
        foreach ($r in $rules) {
            if ($existing -cnotcontains $r) {Add-Content -LiteralPath $ignore -Value $r -Encoding UTF8}
        }
        Write-Host '\nMIGRACAO COMPLETA. Os cinco ZIPs de AVFAD NAO foram tocados.'
        Write-Host 'Teste: python .\src\00_verificar_entradas.py'
        Write-Host 'Teste: python .\src\10_verificar_scattering.py'
        Write-Host 'Antes de publicar: git status --short e git check-ignore dos ZIPs.'
    } else {
        Write-Host '\nARRUMACAO LEVE CONCLUIDA. Scattering, CSVs, NPZs e ZIPs do dataset INTACTOS.'
    }
} catch {
    $problem=$_
    Write-Warning "Falha: $problem. Iniciando reversao de Move-Item ja executados."
    for ($i=$script:Moved.Count-1;$i -ge 0;$i--) {
        $op=$script:Moved[$i]
        try {
            if (Test-Path -LiteralPath $op.Destination) {
                Move-Item -LiteralPath $op.Destination -Destination $op.Source -ErrorAction Stop
            }
        } catch {Write-Warning "Nao foi possivel restaurar $($op.Source): $_"}
    }
    throw $problem
}
