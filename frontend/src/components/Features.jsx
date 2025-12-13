const Features = () => {
  const features = [
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      title: "ניתוח מבוסס בינה מלאכותית",
      description: "מודלי למידת מכונה מתקדמים מנתחים דפוסי ביקורות, שפה וסנטימנט כדי לזהות אותנטיות."
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      title: "זיהוי בזמן אמת",
      description: "קבל תוצאות מיידיות עם ציוני ביטחון ופירוט הסתברויות מפורט לכל ביקורת."
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      title: "מדדי ביטחון",
      description: "אנליטיקה מפורטת המציגה הסתברות מזויף, הסתברות אמיתי ורמות ביטחון כוללות."
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      ),
      title: "מאובטח ופרטי",
      description: "הביקורות שלך מעובדות בצורה מאובטחת. אין אחסון או שיתוף נתונים עם צדדים שלישיים."
    }
  ]

  return (
    <section id="features" className="mt-20 mb-12" dir="rtl">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold text-gray-800 mb-4">תכונות עיקריות</h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          יכולות חזקות שהופכות את זיהוי הביקורות המזויפות למדויק ואמין
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {features.map((feature, index) => (
          <div
            key={index}
            className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-indigo-200 group"
            dir="rtl"
          >
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform ml-auto">
              {feature.icon}
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">{feature.title}</h3>
            <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Features

