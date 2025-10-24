import { navigateTo } from '@devvit/web/client';
import { useYouTubeAnalysis } from './hooks/useYouTubeAnalysis';

export const App = () => {
  const { 
    youtubeUrl, 
    setYoutubeUrl, 
    analysisResult, 
    isAnalyzing, 
    analyzeVideo, 
    error 
  } = useYouTubeAnalysis();

  return (
    <div className="flex relative flex-col justify-center items-center min-h-screen gap-4 p-4">
      <img className="object-contain w-1/2 max-w-[250px] mx-auto" src="/snoo.png" alt="Snoo" />
      <div className="flex flex-col items-center gap-2">
        <h1 className="text-2xl font-bold text-center text-gray-900 ">
          YouTube Video Analyzer
        </h1>
        <p className="text-base text-center text-gray-600 ">
          Analyze YouTube videos for toxicity and content
        </p>
      </div>

      {/* YouTube URL Input Section */}
      <div className="w-full max-w-md flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <label htmlFor="youtube-url" className="text-sm font-medium text-gray-700">
            YouTube Video URL:
          </label>
          <input
            id="youtube-url"
            type="url"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#d93900] focus:border-transparent"
            disabled={isAnalyzing}
          />
        </div>
        
        <button
          onClick={analyzeVideo}
          disabled={!youtubeUrl || isAnalyzing}
          className="w-full bg-[#d93900] text-white py-2 px-4 rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#c23300] transition-colors"
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
        </button>

        {error && (
          <div className="w-full p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        {analysisResult && (
          <div className="w-full p-4 bg-gray-50 border border-gray-200 rounded-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Analysis Result:</h3>
            <div className="space-y-2">
              <p><span className="font-medium">Title:</span> {analysisResult.title}</p>
              <p><span className="font-medium">Channel:</span> {analysisResult.channel}</p>
              <p><span className="font-medium">Duration:</span> {analysisResult.duration}</p>
              <p><span className="font-medium">Views:</span> {analysisResult.views}</p>
              <p><span className="font-medium">Toxicity Score:</span> 
                <span className={`ml-2 px-2 py-1 rounded text-sm ${
                  analysisResult.toxicityScore < 0.3 ? 'bg-green-100 text-green-800' :
                  analysisResult.toxicityScore < 0.7 ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {(analysisResult.toxicityScore * 100).toFixed(1)}%
                </span>
              </p>
              {analysisResult.summary && (
                <div>
                  <p className="font-medium">Summary:</p>
                  <p className="text-sm text-gray-600 mt-1">{analysisResult.summary}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <footer className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3 text-[0.8em] text-gray-600">
        <button
          className="cursor-pointer"
          onClick={() => navigateTo('https://developers.reddit.com/docs')}
        >
          Docs
        </button>
        <span className="text-gray-300">|</span>
        <button
          className="cursor-pointer"
          onClick={() => navigateTo('https://www.reddit.com/r/Devvit')}
        >
          r/Devvit
        </button>
        <span className="text-gray-300">|</span>
        <button
          className="cursor-pointer"
          onClick={() => navigateTo('https://discord.com/invite/R7yu2wh9Qz')}
        >
          Discord
        </button>
      </footer>
    </div>
  );
};
