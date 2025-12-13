# הוראות התקנה ל-Windows

## פתרון בעיות נפוצות ב-Windows

### בעיה 1: Long Path Support

אם אתה מקבל שגיאה על נתיבים ארוכים:

**פתרון מהיר - הפעל Long Paths:**

1. פתח PowerShell **כמנהל** (קליק ימני → Run as Administrator)

2. הרץ את הפקודה:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

3. **הפעל מחדש את המחשב**

4. נסה שוב להתקין:
```powershell
pip install torch transformers sentencepiece sacremoses numpy accelerate
```

**או פתרון חלופי - התקן בנפרד:**
```powershell
pip install torch --no-cache-dir
pip install transformers sentencepiece sacremoses numpy accelerate
```

### בעיה 2: uvicorn לא מזוהה

אם אתה מקבל `uvicorn is not recognized`:

**פתרון 1 - התקן את התלויות הבסיסיות:**
```powershell
cd backend
pip install fastapi uvicorn[standard] pydantic python-multipart
```

**פתרון 2 - השתמש ב-python -m:**
```powershell
python -m uvicorn main:app --reload --port 8000
```

**פתרון 3 - בדוק את ה-PATH:**
```powershell
# בדוק איפה Python מותקן
where python

# הוסף ל-PATH אם צריך (במשתני סביבה)
```

### בעיה 3: Scripts לא ב-PATH

אם אתה רואה אזהרות על Scripts לא ב-PATH:

**פתרון - הוסף את התיקייה ל-PATH:**

1. פתח "Environment Variables" (משתני סביבה)
2. מצא את `Path` ב-"User variables"
3. הוסף: `C:\Users\אסיף פרץ\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts`
4. הפעל מחדש את PowerShell

## התקנה מומלצת - שלב אחר שלב

### שלב 1: צור סביבה וירטואלית (מומלץ)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

אם אתה מקבל שגיאה על ExecutionPolicy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### שלב 2: התקן תלויות

```powershell
# תלויות בסיסיות
pip install fastapi uvicorn[standard] pydantic python-multipart

# תלויות ML (אם יש בעיית Long Paths, התקן בנפרד)
pip install torch --no-cache-dir
pip install transformers sentencepiece sacremoses numpy accelerate
```

### שלב 3: הפעל את השרת

```powershell
# ודא שאתה בתיקיית backend
cd backend

# הפעל את השרת
python -m uvicorn main:app --reload --port 8000
```

## בדיקת התקנה

בדוק שהכל מותקן:

```powershell
python -c "import fastapi; import uvicorn; import torch; import transformers; print('✅ הכל מותקן!')"
```

אם יש שגיאה, התקן את החבילה החסרה.

## פתרון בעיות נוספות

### אם torch לא מתקין:

נסה גרסה ספציפית:
```powershell
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### אם transformers לא מתקין:

```powershell
pip install transformers --no-cache-dir
```

### אם יש בעיות זיכרון:

התקן רק את מה שצריך:
```powershell
# רק לתרגום וזיהוי בסיסי
pip install transformers torch sentencepiece
```

## תמיכה

אם עדיין יש בעיות:
1. בדוק את גרסת Python: `python --version` (צריך 3.8+)
2. בדוק שיש חיבור לאינטרנט (המודלים מורידים מ-Hugging Face)
3. נסה להתקין חבילה אחת בכל פעם כדי לזהות את הבעיה

