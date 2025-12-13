const Header = () => {
  return (
    <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between" dir="rtl">
          <nav className="hidden md:flex items-center gap-6">
            <a href="#analyzer" className="text-gray-600 hover:text-indigo-600 transition-colors font-medium">מנתח</a>
            <a href="#features" className="text-gray-600 hover:text-indigo-600 transition-colors font-medium">תכונות</a>
            <a href="#about" className="text-gray-600 hover:text-indigo-600 transition-colors font-medium">אודות</a>
          </nav>
          <div className="flex items-center gap-3" dir="ltr">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-800">ReviewGuard</h2>
              <p className="text-xs text-gray-500" dir="rtl">מערכת זיהוי ביקורות מזויפות</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header

