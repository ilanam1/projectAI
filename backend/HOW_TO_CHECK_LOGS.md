# איך לבדוק אם המודלים עובדים - מדריך

## 🔍 מה זה "לוגים" ואיפה לראות אותם?

### הלוגים = מה שמופיע בטרמינל של השרת

כשאתה מריץ את השרת:
```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

**הטרמינל הזה מראה את הלוגים!**

## 📊 מה אתה אמור לראות אם המודלים עובדים:

### שלב 1: כשהשרת מתחיל
אם המודלים נטענים אוטומטית, תראה:
```
🔄 Loading translation model: Helsinki-NLP/opus-mt-he-en
✅ Translation model loaded successfully!
🔄 Loading review classifier: debojit01/fake-review-detector
✅ Review classifier loaded successfully!
🔄 Loading AI detector: roberta-base-openai-detector
✅ AI detector loaded successfully!
🎉 All models loaded and ready!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### שלב 2: כשאתה מנתח ביקורת
תראה:
```
🔍 Model 1 Raw Results: [{'label': 'CG', 'score': 0.85}, {'label': 'REAL', 'score': 0.15}]
  - Label: 'CG' (upper: 'CG'), Score: 0.8500
    → Identified as FAKE/CG
  - Label: 'REAL' (upper: 'REAL'), Score: 0.1500
    → Identified as REAL
🔍 Model 1 Results - CG: 0.850, REAL: 0.150, Classification: FAKE, Score: 0.850
📊 Final: FAKE with probability 0.850
```

## ❌ מה זה אומר אם אתה לא רואה את זה:

### אם אתה רואה:
```
⚠️ ML libraries not installed. Using placeholder mode.
```
**זה אומר:** המודלים לא מותקנים - צריך להתקין:
```powershell
pip install transformers torch sentencepiece sacremoses numpy accelerate
```

### אם אתה רואה:
```
❌ Error loading models: [שגיאה]
   Falling back to placeholder mode.
```
**זה אומר:** יש שגיאה בטעינת המודלים - בדוק את השגיאה.

### אם אתה לא רואה כלום:
**זה אומר:** המודלים לא נטענים - הם ייטענו בפעם הראשונה שתנתח ביקורת.

## 🔄 מה ההבדל בין Swagger UI ללוגים?

### Swagger UI (`http://localhost:8000/docs`):
- זה רק ממשק לבדיקת ה-API
- זה מראה שהשרת **רץ**
- זה **לא** מראה אם המודלים נטענו

### הלוגים (בטרמינל):
- זה מראה מה **קורה בפועל**
- זה מראה אם המודלים **נטענו**
- זה מראה מה המודלים **מחזירים** כשאתה מנתח

## ✅ איך לבדוק שהכל עובד:

### שלב 1: בדוק את הטרמינל
פתח את הטרמינל שבו השרת רץ ובדוק:
- האם יש הודעות של טעינת מודלים?
- האם יש שגיאות?

### שלב 2: נסה לנתח ביקורת
1. פתח את האתר
2. הכנס ביקורת
3. לחץ "נתח ביקורת"
4. **עכשיו תסתכל על הטרמינל** - מה כתוב שם?

### שלב 3: בדוק את הלוגים
חפש בטרמינל:
- `🔍 Model 1 Raw Results` - זה אומר שהמודל עובד
- `📊 Final:` - זה התוצאה הסופית
- אם אתה רואה `Placeholder` - המודלים לא עובדים

## 🐛 אם המודלים לא נטענים:

### פתרון 1: טען ידנית
ערוך את `backend/ml_model.py` ומצא בסוף:
```python
try:
    load_models()
except Exception as e:
    print(f"⚠️ Could not auto-load models: {e}")
```

אם יש שגיאה, תראה אותה כאן.

### פתרון 2: בדוק התקנות
```powershell
python -c "import transformers; import torch; print('✅ OK')"
```

אם יש שגיאה, צריך להתקין:
```powershell
pip install transformers torch sentencepiece sacremoses
```

### פתרון 3: בדוק חיבור לאינטרנט
המודלים מורידים מ-Hugging Face - צריך חיבור לאינטרנט.

## 📝 דוגמה ללוגים תקינים:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
🔄 Loading translation model: Helsinki-NLP/opus-mt-he-en
Downloading: 100%|████████| 123M/123M [00:30<00:00, 4.1MB/s]
✅ Translation model loaded successfully!
🔄 Loading review classifier: debojit01/fake-review-detector
Downloading: 100%|████████| 456M/456M [01:15<00:00, 6.0MB/s]
✅ Review classifier loaded successfully!
🔄 Loading AI detector: roberta-base-openai-detector
✅ AI detector loaded successfully!
🎉 All models loaded and ready!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:54321 - "POST /predict HTTP/1.1" 200 OK
🔍 Model 1 Raw Results: [{'label': 'CG', 'score': 0.92}]
📊 Final: FAKE with probability 0.920
```

## 🎯 סיכום:

**הלוגים = מה שמופיע בטרמינל של השרת**

אם אתה רואה:
- ✅ הודעות של טעינת מודלים → המודלים עובדים
- ❌ "placeholder mode" → המודלים לא עובדים
- 🔍 "Model 1 Raw Results" → המודלים עובדים ומחזירים תוצאות

**שלח לי את מה שמופיע בטרמינל כשאתה מנתח ביקורת!**

