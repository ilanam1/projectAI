import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8000/predict'
const API_HEALTH_URL = 'http://localhost:8000/health'

const ReviewAnalyzer = () => {
  const [reviewText, setReviewText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDemoMode, setIsDemoMode] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking') // 'checking', 'connected', 'disconnected'

  // Check backend connection on component mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
        
        // First check health endpoint
        const healthResponse = await fetch(API_HEALTH_URL, {
          method: 'GET',
          signal: controller.signal,
        })
        
        if (!healthResponse.ok) {
          setBackendStatus('disconnected')
          setIsDemoMode(true)
          clearTimeout(timeoutId)
          return
        }
        
        // Health check passed, mark as connected
        // We'll verify /predict works when user actually tries to analyze
        setBackendStatus('connected')
        setIsDemoMode(false)
        clearTimeout(timeoutId)
      } catch (err) {
        setBackendStatus('disconnected')
        setIsDemoMode(true)
        console.log('Backend health check failed:', err.message)
      }
    }

    checkBackend()
    // Check every 10 seconds
    const interval = setInterval(checkBackend, 10000)
    return () => clearInterval(interval)
  }, [])

  // Mock function for demo mode (when backend is not available)
  const getMockResult = (text) => {
    // Simple heuristic-based mock: longer reviews with positive words are more likely to be real
    const textLength = text.length
    const positiveWords = ['great', 'excellent', 'amazing', 'love', 'recommend', 'perfect', 'wonderful', 'good', 'nice', 'satisfied']
    const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'poor', 'waste']
    
    const lowerText = text.toLowerCase()
    const positiveCount = positiveWords.filter(word => lowerText.includes(word)).length
    const negativeCount = negativeWords.filter(word => lowerText.includes(word)).length
    
    // Calculate a mock probability (0-1)
    let fakeProbability = 0.3 // Base probability
    
    // Adjust based on length (very short reviews might be fake)
    if (textLength < 20) fakeProbability += 0.3
    else if (textLength > 200) fakeProbability -= 0.2
    
    // Adjust based on sentiment words
    if (positiveCount > negativeCount) fakeProbability -= 0.2
    if (negativeCount > positiveCount) fakeProbability += 0.1
    
    // Add some randomness
    fakeProbability += (Math.random() - 0.5) * 0.2
    
    // Clamp between 0 and 1
    fakeProbability = Math.max(0, Math.min(1, fakeProbability))
    
    const isFake = fakeProbability >= 0.5
    const confidence = Math.abs(fakeProbability - 0.5) * 2
    
    return {
      is_fake: isFake,
      confidence: confidence,
      probability: fakeProbability
    }
  }

  const handleAnalyze = async () => {
    if (!reviewText.trim()) {
      setError('אנא הזן ביקורת לניתוח')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setIsDemoMode(false)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: reviewText }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Server error: ${response.status} - ${errorText}`)
      }

      const data = await response.json()
      
      // Success! Clear all errors and update status
      setResult(data)
      setIsDemoMode(false)
      setBackendStatus('connected')
      setError(null) // Clear any previous errors
      
      console.log('✅ Analysis successful:', data)
    } catch (err) {
      console.error('❌ Analysis error:', err)
      
      // Check if it's a network error (backend not running)
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.message.includes('ERR_CONNECTION_REFUSED') || err.name === 'TypeError') {
        console.warn('Backend not available, using demo mode:', err.message)
        setIsDemoMode(true)
        setBackendStatus('disconnected')
        const mockData = getMockResult(reviewText)
        setResult(mockData)
        setError('⚠️ שרת ה-backend לא פועל. מציג תוצאות דמו. הפעל את שרת ה-backend לשימוש במודל ML האמיתי.')
      } else {
        // Other errors (server errors, validation errors, etc.)
        // Server is running but there's an error - show error but don't mark as disconnected
        console.error('Server error (but server is running):', err.message)
        setError(`⚠️ שגיאה בניתוח: ${err.message}. בדוק את לוגי השרת.`)
        // Don't set result or demo mode - let user see the error
        // Keep backendStatus as 'connected' since server responded
      }
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setReviewText('')
    setResult(null)
    setError(null)
  }

  // Handle both old format (confidence, probability) and new format (confidence_score, fake_probability)
  // confidence_score is already in percentage (0-100), confidence is 0-1
  const confidencePercentage = result 
    ? Math.round(result.confidence_score !== undefined ? result.confidence_score : (result.confidence ? result.confidence * 100 : 0))
    : 0
  // fake_probability is 0-1, probability is also 0-1, so multiply by 100 for percentage
  const fakeProbability = result 
    ? parseFloat(((result.fake_probability !== undefined ? result.fake_probability : (result.probability || 0)) * 100).toFixed(2))
    : 0

  return (
    <div id="analyzer" className="max-w-5xl mx-auto animate-fade-in">
      {/* Backend Status Indicator */}
      {backendStatus === 'checking' && (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl shadow-sm animate-slide-in">
          <p className="text-blue-800 text-sm flex items-center font-medium">
            <svg className="animate-spin w-5 h-5 ml-3" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            בודק חיבור לשרת...
          </p>
        </div>
      )}
      {backendStatus === 'connected' && !isDemoMode && (
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl shadow-sm animate-slide-in">
          <p className="text-green-800 text-sm flex items-center font-medium">
            <svg className="w-5 h-5 ml-3" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            ✅ השרת מחובר - משתמש במודל ML אמיתי
          </p>
        </div>
      )}
      {backendStatus === 'disconnected' && isDemoMode && !result && (
        <div className="mb-6 p-4 bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-200 rounded-xl shadow-sm animate-slide-in">
          <p className="text-yellow-800 text-sm flex items-center font-medium">
            <svg className="w-5 h-5 ml-3" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            ⚠️ מצב דמו: השרת לא מחובר. מציג תוצאות מדומות. 
            <span className="mr-2 font-semibold">לשימוש במודל האמיתי, הפעל את שרת ה-backend.</span>
          </p>
        </div>
      )}
      <div className="bg-white rounded-2xl shadow-2xl p-8 md:p-10 border border-gray-100 hover:shadow-3xl transition-shadow duration-300" dir="rtl">
        {/* Input Section */}
        <div className="mb-8">
          <label
            htmlFor="review-input"
            className="block text-lg font-semibold text-gray-800 mb-4 flex items-center"
          >
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center mr-3 shadow-md">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </div>
            הזן טקסט ביקורת
          </label>
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 rounded-2xl blur-sm opacity-50"></div>
            <textarea
              id="review-input"
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              placeholder="הדבק או הקלד את הביקורת שברצונך לנתח כאן...&#10;&#10;דוגמה: 'המוצר הזה עלה על כל הציפיות שלי! האיכות מצוינת ואני בהחלט ממליץ עליו לאחרים.'"
              className="relative w-full h-64 px-6 py-5 bg-white/90 backdrop-blur-sm border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-indigo-200 focus:border-indigo-400 resize-none transition-all duration-300 text-gray-800 placeholder-gray-400 shadow-lg hover:shadow-xl hover:border-indigo-300 text-base leading-relaxed"
              disabled={loading}
              style={{
                fontFamily: 'inherit',
                lineHeight: '1.7'
              }}
            />
            <div className="absolute top-4 right-4 opacity-30">
              <svg className="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
          </div>
          <div className="mt-4 flex items-center justify-between bg-gradient-to-r from-gray-50 to-indigo-50 rounded-xl px-4 py-3 border border-gray-100">
            <div className="text-sm text-gray-600 flex items-center font-medium">
              <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center mr-2 shadow-sm">
                <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                </svg>
              </div>
              <span className="text-gray-700 font-semibold">{reviewText.length}</span>
              <span className="mr-2">תווים</span>
            </div>
            {reviewText.length > 0 && (
              <div className="flex items-center bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-4 py-1.5 rounded-lg shadow-md">
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm font-semibold">מוכן לניתוח</span>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 mb-8">
          <button
            onClick={handleAnalyze}
            disabled={loading || !reviewText.trim()}
            className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-8 py-4 rounded-xl font-bold text-lg hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-4 focus:ring-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-[1.02] active:scale-[0.98]"
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin -mr-1 ml-3 h-6 w-6 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                מנתח...
              </span>
            ) : (
              <span className="flex items-center justify-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                נתח ביקורת
              </span>
            )}
          </button>
          <button
            onClick={handleClear}
            disabled={loading}
            className="px-8 py-4 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 hover:border-gray-400 focus:outline-none focus:ring-4 focus:ring-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            <span className="flex items-center">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              נקה
            </span>
          </button>
        </div>

        {/* Error Message - Only show if there's an error AND no successful result */}
        {error && !result && (
          <div className="mb-6 p-4 bg-gradient-to-r from-red-50 to-pink-50 border-2 border-red-200 rounded-xl shadow-sm animate-slide-in">
            <p className="text-red-800 text-sm flex items-center font-medium">
              <svg className="w-5 h-5 ml-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              {error}
            </p>
          </div>
        )}

        {/* Result Section */}
        {result && (
          <div className="mt-8 p-8 bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl border-2 border-gray-200 shadow-lg animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800 flex items-center">
                <svg className="w-6 h-6 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                תוצאות הניתוח
              </h2>
              <div className="text-xs text-gray-500 bg-white px-3 py-1 rounded-full border border-gray-200" dir="ltr">
                {new Date().toLocaleTimeString('he-IL')}
              </div>
            </div>

            {/* Result Badge */}
            <div className="mb-6">
              <div
                className={`inline-flex items-center px-6 py-3 rounded-full font-bold text-xl shadow-lg ${
                  result.is_fake
                    ? 'bg-gradient-to-r from-red-500 to-pink-500 text-white'
                    : 'bg-gradient-to-r from-green-500 to-emerald-500 text-white'
                } animate-slide-in`}
              >
                {result.is_fake ? (
                  <>
                    <svg
                      className="w-5 h-5 ml-2"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clipRule="evenodd"
                      />
                    </svg>
                    ביקורת מזויפת
                  </>
                ) : (
                  <>
                    <svg
                      className="w-5 h-5 ml-2"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                    ביקורת אמיתית
                  </>
                )}
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="mb-6">
              <div className="flex justify-between items-center mb-3">
                <span className="text-base font-semibold text-gray-700 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  רמת ביטחון
                </span>
                <span className="text-lg font-bold text-gray-800 bg-white px-3 py-1 rounded-lg border border-gray-200" dir="ltr">
                  {confidencePercentage}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 shadow-inner overflow-hidden">
                <div
                  className={`h-4 rounded-full transition-all duration-1000 ease-out shadow-lg ${
                    result.is_fake 
                      ? 'bg-gradient-to-r from-red-500 to-pink-500' 
                      : 'bg-gradient-to-r from-green-500 to-emerald-500'
                  }`}
                  style={{ width: `${confidencePercentage}%` }}
                ></div>
              </div>
            </div>

            {/* Probability Details */}
            <div className="grid grid-cols-2 gap-6 mt-6">
              <div className="p-5 bg-white rounded-xl border-2 border-red-200 shadow-md hover:shadow-lg transition-shadow">
                <div className="text-sm font-semibold text-gray-600 mb-2 flex items-center">
                  <svg className="w-4 h-4 ml-2 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  הסתברות מזויף
                </div>
                <div className="text-4xl font-bold bg-gradient-to-r from-red-600 to-pink-600 bg-clip-text text-transparent">
                  {fakeProbability < 0.01 ? fakeProbability.toFixed(4) : fakeProbability.toFixed(2)}%
                </div>
              </div>
              <div className="p-5 bg-white rounded-xl border-2 border-green-200 shadow-md hover:shadow-lg transition-shadow">
                <div className="text-sm font-semibold text-gray-600 mb-2 flex items-center">
                  <svg className="w-4 h-4 ml-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  הסתברות אמיתי
                </div>
                <div className="text-4xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                  {(100 - fakeProbability) < 0.01 ? (100 - fakeProbability).toFixed(4) : (100 - fakeProbability).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ReviewAnalyzer

