const About = () => {
  const teamMembers = [
    {
      name: "אילן אמוייב",
      role: "Tester",
      roleHebrew: "בודק תוכנה",
      description: "אחראי על בדיקות איכות, אימות פונקציונליות וזיהוי באגים במערכת",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    {
      name: "שילת יוספי",
      role: "UI/UX Designer",
      roleHebrew: "מעצב ממשק משתמש",
      description: "מעצב את חוויית המשתמש והממשק הגרפי של המערכת",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
        </svg>
      )
    },
    {
      name: "שובל אש",
      role: "Data Engineer",
      roleHebrew: "מהנדס נתונים",
      description: "מתמחה בעיבוד נתונים, בניית pipelines וניהול מאגרי מידע",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
        </svg>
      )
    },
    {
      name: "עומר דניאל",
      role: "Software Architect",
      roleHebrew: "ארכיטקט תוכנה",
      description: "מתכנן את הארכיטקטורה הכללית של המערכת ומבנה הקוד",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      )
    },
    {
      name: "רון כהן",
      role: "Regulatory Expert",
      roleHebrew: "מומחה רגולציה",
      description: "מוודא שהמערכת עומדת בתקנים ובתקנות הרלוונטיות",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      )
    },
    {
      name: "אסיף פרץ",
      role: "AI Engineer",
      roleHebrew: "מהנדס בינה מלאכותית",
      description: "מפתח ומטמיע את מודלי למידת המכונה ופועל על שיפור הדיוק",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      )
    }
  ]

  return (
    <section id="about" className="mt-20 mb-12" dir="rtl">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold text-gray-800 mb-4">אודות הצוות</h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          צוות מקצועי ומנוסה שפועל יחדיו כדי לספק פתרון מתקדם לזיהוי ביקורות מזויפות
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {teamMembers.map((member, index) => (
          <div
            key={index}
            className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-indigo-200 group"
          >
            <div className="flex items-start gap-4 mb-4">
              <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center text-white shadow-md group-hover:scale-110 transition-transform flex-shrink-0">
                {member.icon}
              </div>
              <div className="flex-1 text-right">
                <h3 className="text-xl font-bold text-gray-800 mb-1">{member.name}</h3>
                <p className="text-sm text-indigo-600 font-semibold mb-1">{member.roleHebrew}</p>
                <p className="text-xs text-gray-500" dir="ltr">{member.role}</p>
              </div>
            </div>
            <p className="text-gray-600 text-sm leading-relaxed text-right">
              {member.description}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-12 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-2xl p-8 border border-indigo-100">
        <div className="text-center">
          <h3 className="text-2xl font-bold text-gray-800 mb-4">על הפרויקט</h3>
          <p className="text-gray-700 max-w-3xl mx-auto leading-relaxed text-right">
            פרויקט זה הוא תוצאה של עבודה משותפת של צוות מקצועי המשלב מומחיות בתחומי בדיקות תוכנה, עיצוב ממשק משתמש, הנדסת נתונים, ארכיטקטורת תוכנה, רגולציה והנדסת בינה מלאכותית. 
            המערכת משתמשת במודלי למידת מכונה מתקדמים לזיהוי ביקורות מזויפות, תוך שימוש בטכנולוגיות עדכניות ועמידה בסטנדרטים הגבוהים ביותר של איכות ואמינות.
          </p>
        </div>
      </div>
    </section>
  )
}

export default About

