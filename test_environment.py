# Test de l'environnement
import pandas as pd
import pdfplumber
from pathlib import Path

print("✅ Pandas version:", pd.__version__)
print("✅ pdfplumber importé avec succès")
print("✅ Environnement prêt!")

# Vérifier les PDFs
pdf_dir = Path("data/raw")
pdfs = list(pdf_dir.glob("*.pdf"))
print(f"\n📄 Nombre de PDFs trouvés : {len(pdfs)}")
for pdf in pdfs:
    print(f"  - {pdf.name}")
