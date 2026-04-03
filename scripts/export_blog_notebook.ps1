param(
    [string]$NotebookPath = "notebooks/05_blog_tumor_detection.ipynb",
    [string]$OutputDir = "docs",
    [string]$OutputFile = "index.html"
)

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
jupyter nbconvert $NotebookPath --to html --output $OutputFile --output-dir $OutputDir
