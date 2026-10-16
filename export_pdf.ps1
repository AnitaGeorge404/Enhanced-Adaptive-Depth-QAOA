$docPath = 'd:\Projects\Enhanced-Adaptive-Depth-QAOA\IEG313_Enhanced_Adaptive_Depth_QAOA_Project_Report.docx'
$pdfPath = 'd:\Projects\Enhanced-Adaptive-Depth-QAOA\IEG313_Enhanced_Adaptive_Depth_QAOA_Project_Report.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docPath)
    $doc.Fields.Update()
    $doc.ExportAsFixedFormat($pdfPath, 17)
    $doc.Close([ref]$false)
    Write-Host "Word COM export completed successfully."
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
