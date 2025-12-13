const Footer = () => {
  return (
    <footer className="bg-gray-900 text-gray-300 mt-20">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-right">
            <h3 className="text-white font-bold text-lg mb-4" dir="ltr">ReviewGuard</h3>
            <p className="text-sm text-gray-400 text-right">
              מערכת מתקדמת לזיהוי ביקורות מזויפות המבוססת על בינה מלאכותית, למידת מכונה ועיבוד שפה טבעית.
            </p>
          </div>
          <div className="text-right">
            <h3 className="text-white font-bold text-lg mb-4">טכנולוגיה</h3>
            <ul className="space-y-2 text-sm text-right">
              <li dir="ltr" className="text-left">• FastAPI Backend</li>
              <li dir="ltr" className="text-left">• React + Vite Frontend</li>
              <li>• מודלי למידת מכונה</li>
              <li>• עיבוד שפה טבעית</li>
            </ul>
          </div>
          <div className="text-right">
            <h3 className="text-white font-bold text-lg mb-4">פרויקט</h3>
            <p className="text-sm text-gray-400 text-right">
              פרויקט אקדמי להצגה בקורס. נבנה עם טכנולוגיות ווב מודרניות ויכולות AI/ML.
            </p>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
          <p dir="rtl">© 2024 <span dir="ltr">ReviewGuard</span> - מערכת זיהוי ביקורות מזויפות | פרויקט אקדמי | ישראל 🇮🇱</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer

