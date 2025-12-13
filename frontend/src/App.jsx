import ReviewAnalyzer from './components/ReviewAnalyzer'
import Header from './components/Header'
import Footer from './components/Footer'
import Features from './components/Features'
import About from './components/About'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <Header />
      <main className="container mx-auto px-4 py-12" dir="rtl">
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4 animate-fade-in">
            זיהוי ביקורות מזויפות
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            ניתוח מתקדם מבוסס בינה מלאכותית לזיהוי ביקורות אותנטיות ומזויפות באמצעות למידת מכונה מתקדמת
          </p>
        </div>
        <ReviewAnalyzer />
        <Features />
        <About />
      </main>
      <Footer />
    </div>
  )
}

export default App

