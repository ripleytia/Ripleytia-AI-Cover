@echo off
echo [Ripleytia AI] Kodlar GitHub'a gonderiliyor...
git init
git add .
git commit -m "feat: v1.0.0 Stable - Ripleytia AI Automated Cover Launch (Complete MVP)"
echo --------------------------------------------------
echo Lutfen asagidaki satira kendi GitHub depo (repository) linkini yapistir ve Enter'a bas:
set /p repo_url=Repo Linki: 
git remote add origin %repo_url%
git branch -M main
git push -u origin main --force
echo [Basarili] Tum kodlar ve mimari GitHub deponuza yuklendi!
pause
