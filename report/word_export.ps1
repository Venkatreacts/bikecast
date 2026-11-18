# Opens the built report in Microsoft Word, fills the TOC / list fields and page numbers, saves, and exports a PDF.
param([string]$docx = (Join-Path $PSScriptRoot "Team8_ML_Project_Report.docx"))
$ErrorActionPreference = "Stop"
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.Update() }  # second pass: page numbers settle after the lists grow
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)  # 17 = wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close([ref]0)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
"saved $pdf"
