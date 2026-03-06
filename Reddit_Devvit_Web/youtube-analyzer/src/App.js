import { useState } from 'react';
import { AlertCircle, CheckCircle, Video, TrendingUp, AlertTriangle, Info } from 'lucide-react';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const extractVideoId = (url) => {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
      /^([a-zA-Z0-9_-]{11})$/
    ];
    
    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }
    return null;
  };

  const analyzeVideo = async () => {
    setError('');
    setResults(null);
    
    const videoId = extractVideoId(url);
    if (!videoId) {
      setError('Invalid YouTube URL. Please enter a valid YouTube video link.');
      return;
    }

    setLoading(true);

    try {
      /*const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ video_id: videoId })
      });*/
      const response = await fetch('https://your-devvit-web-url/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze video');
      }
      
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze video. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity?.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': case 'moderate': return 'text-orange-600 bg-orange-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 7) return 'text-red-600';
    if (score >= 4) return 'text-orange-600';
    return 'text-green-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 p-4" style={{ marginLeft: '1in' }}>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center mb-4">
            <Video className="w-12 h-12 text-purple-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-800">
              YouTube Content Analyzer
            </h1>
          </div>
          <p className="text-gray-600 text-lg">
            Analyze videos for toxicity, bias, and misinformation
          </p>
          <div className="mt-2 text-sm text-gray-500">
            Built for Reddit Community Games Challenge 2025
          </div>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <label className="block text-gray-700 font-semibold mb-3 text-lg">
            Enter YouTube URL or Video ID
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=... or video ID"
              className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none text-lg"
              onKeyPress={(e) => e.key === 'Enter' && analyzeVideo()}
            />
            <button
              onClick={analyzeVideo}
              disabled={loading || !url}
              className="px-8 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-lg"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
              <span className="text-red-700">{error}</span>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-white rounded-2xl shadow-xl p-12 text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-purple-600 mx-auto mb-4"></div>
            <p className="text-gray-600 text-lg">Fetching transcript and analyzing content...</p>
            <p className="text-gray-500 text-sm mt-2">This may take 10-30 seconds</p>
          </div>
        )}

        {/* Results Section */}
        {results && !loading && (
          <div className="space-y-6">
            {/* Video Info */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <Video className="w-6 h-6 text-purple-600" />
                <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
              </div>
              <p className="text-gray-600">
                Video ID: <span className="font-mono font-semibold">{results.videoId}</span>
              </p>
              {results.transcriptLength && (
                <p className="text-gray-600">
                  Transcript Length: <span className="font-semibold">{results.transcriptLength.toLocaleString()}</span> characters
                </p>
              )}
            </div>

            {/* Toxicity Score */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-6 h-6 text-orange-600" />
                <h3 className="text-xl font-bold text-gray-800">Toxicity Analysis</h3>
              </div>
              <div className="flex items-center gap-4 mb-4">
                <div className="text-5xl font-bold">
                  <span className={getScoreColor(results.toxicity.score)}>
                    {results.toxicity.score}
                  </span>
                  <span className="text-gray-400 text-3xl">/10</span>
                </div>
                <div className="flex-1">
                  <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${results.toxicity.score >= 7 ? 'bg-red-500' : results.toxicity.score >= 4 ? 'bg-orange-500' : 'bg-green-500'}`}
                      style={{ width: `${results.toxicity.score * 10}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              <div className={`inline-block px-4 py-2 rounded-full text-sm font-semibold ${getSeverityColor(results.toxicity.severity)}`}>
                {results.toxicity.severity.toUpperCase()} SEVERITY
              </div>
              <p className="text-gray-600 mt-4">{results.toxicity.details}</p>
            </div>

            {/* Bias Detection */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-6 h-6 text-blue-600" />
                <h3 className="text-xl font-bold text-gray-800">Bias Detection</h3>
              </div>
              {results.bias.detected ? (
                <div>
                  <div className={`inline-block px-4 py-2 rounded-full text-sm font-semibold mb-4 ${getSeverityColor(results.bias.severity)}`}>
                    BIAS DETECTED - {results.bias.severity.toUpperCase()}
                  </div>
                  {results.bias.types && results.bias.types.length > 0 && (
                    <div className="mb-4">
                      <p className="font-semibold text-gray-700 mb-2">Types of bias found:</p>
                      <div className="flex flex-wrap gap-2">
                        {results.bias.types.map((type, idx) => (
                          <span key={idx} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                            {type}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  <p className="text-gray-600">{results.bias.details}</p>
                </div>
              ) : (
                <div className="flex items-center gap-3 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="font-semibold">No significant bias detected</span>
                </div>
              )}
            </div>

            {/* Misinformation Detection */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <Info className="w-6 h-6 text-purple-600" />
                <h3 className="text-xl font-bold text-gray-800">Misinformation Check</h3>
              </div>
              {results.misinformation.detected ? (
                <div>
                  <div className={`inline-block px-4 py-2 rounded-full text-sm font-semibold mb-4 ${getSeverityColor(results.misinformation.severity)}`}>
                    POTENTIAL MISINFORMATION - {results.misinformation.severity.toUpperCase()}
                  </div>
                  {results.misinformation.claims && results.misinformation.claims.length > 0 && (
                    <div className="mb-4">
                      <p className="font-semibold text-gray-700 mb-2">Flagged claims:</p>
                      <ul className="space-y-2">
                        {results.misinformation.claims.map((claim, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-gray-700">
                            <span className="text-purple-600 font-bold">•</span>
                            <span>{claim}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <p className="text-gray-600">{results.misinformation.details}</p>
                </div>
              ) : (
                <div className="flex items-center gap-3 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="font-semibold">No obvious misinformation detected</span>
                </div>
              )}
            </div>

            {/* Keyword Analysis (if available) */}
            {results.keyword_analysis && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Info className="w-6 h-6 text-indigo-600" />
                  <h3 className="text-xl font-bold text-gray-800">Keyword Analysis</h3>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Science Credibility</p>
                    <p className="text-lg font-bold">
                      {results.keyword_analysis.science_credibility ? 
                        <span className="text-green-600">✓ High</span> : 
                        <span className="text-orange-600">⚠ Low</span>
                      }
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Misinfo Flags</p>
                    <p className="text-lg font-bold text-red-600">
                      {results.keyword_analysis.misinfo_flags_count}
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Deepfake Risk</p>
                    <p className="text-lg font-bold">
                      {results.keyword_analysis.deepfake_risk ? 
                        <span className="text-red-600">⚠ Yes</span> : 
                        <span className="text-green-600">✓ None</span>
                      }
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Sentiment</p>
                    <p className="text-lg font-bold">
                      {results.keyword_analysis.sentiment.compound > 0.05 ? 
                        <span className="text-green-600">Positive</span> : 
                        results.keyword_analysis.sentiment.compound < -0.05 ?
                        <span className="text-red-600">Negative</span> :
                        <span className="text-gray-600">Neutral</span>
                      }
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Summary */}
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl shadow-xl p-6 text-white">
              <h3 className="text-xl font-bold mb-3">Overall Summary</h3>
              <p className="text-purple-100 leading-relaxed">{results.summary}</p>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-12 pb-8 text-gray-500 text-sm">
          <p>Built with React + Vite and AI-powered content analysis</p>
          <p className="mt-1">Reddit Community Games Challenge 2025</p>
        </div>
      </div>
    </div>
  );
}

export default App;