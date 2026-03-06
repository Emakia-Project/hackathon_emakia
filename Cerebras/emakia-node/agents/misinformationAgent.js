class MisinformationAgent {
    constructor() {
        this.name = 'MisinformationAgent';
        this.description = 'Real AI misinformation detection using Cerebras Qwen-3-32B model';
        this.cerebrasApiKey = process.env.CEREBRAS_API_KEY;
        this.cerebrasApiUrl = process.env.CEREBRAS_API_URL || 'https://api.cerebras.com';
        this.model = 'Qwen-3-32B';
        this.isReady = !!this.cerebrasApiKey;
        
        // Fallback patterns for when API is unavailable
        this.fallbackPatterns = {
            conspiracy_theories: ['conspiracy', 'cover up', 'secret agenda', 'hidden truth'],
            pseudoscience: ['quantum healing', 'energy crystals', 'alternative facts'],
            clickbait: ['you won\'t believe', 'shocking truth', 'number one secret'],
            sensationalism: ['breaking news', 'urgent warning', 'must see'],
            unverified_claims: ['studies show', 'research proves', 'experts agree']
        };
        
        // Factual error patterns for common misinformation
        this.factualErrors = {
            geography: {
                'paris is in spain': { correct: 'Paris is in France', score: 0.9 },
                'london is in france': { correct: 'London is in England/UK', score: 0.9 },
                'tokyo is in china': { correct: 'Tokyo is in Japan', score: 0.9 },
                'moscow is in ukraine': { correct: 'Moscow is in Russia', score: 0.9 },
                'berlin is in austria': { correct: 'Berlin is in Germany', score: 0.9 }
            },
            science: {
                'earth is flat': { correct: 'Earth is spherical', score: 0.95 },
                'vaccines cause autism': { correct: 'No scientific evidence supports this claim', score: 0.9 },
                'climate change is a hoax': { correct: 'Climate change is supported by scientific consensus', score: 0.9 }
            },
            history: {
                'world war 2 ended in 1944': { correct: 'WW2 ended in 1945', score: 0.9 },
                'nixon was president in 1960': { correct: 'Nixon was president 1969-1974', score: 0.8 }
            }
        };
    }

    /**
     * Analyze text for misinformation using Cerebras Qwen-3-32B
     * @param {string} text - Text to analyze
     * @returns {Promise<Object>} - Misinformation analysis results
     */
    async analyze(text) {
        try {
            if (!this.cerebrasApiKey) {
                console.log('No Cerebras API key configured, using fallback analysis...');
                return this.fallbackAnalysis(text);
            }

            // Create prompt for misinformation analysis
            const prompt = this.createMisinformationPrompt(text);
            
            // Call Cerebras API with Qwen-3-32B
            const response = await fetch(`${this.cerebrasApiUrl}/v1/chat/completions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.cerebrasApiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: this.model,
                    messages: [
                        {
                            role: 'system',
                            content: 'You are a misinformation detection expert. Analyze the given text and determine if it contains false claims, misleading content, or misinformation. Respond with a JSON object containing: hasMisinformation (boolean), misinformationScore (0-1), severity (none/low/medium/high/critical), categories (array of detected issues), and recommendations (array of suggestions).'
                        },
                        {
                            role: 'user',
                            content: prompt
                        }
                    ],
                    temperature: 0.1,
                    max_tokens: 500
                })
            });

            if (!response.ok) {
                throw new Error(`Cerebras API error: ${response.status}`);
            }

            const data = await response.json();
            const aiResponse = data.choices[0].message.content;
            
            // Parse AI response
            const analysis = this.parseAIResponse(aiResponse, text);
            
            return analysis;
        } catch (error) {
            console.error('Cerebras misinformation analysis error:', error);
            console.log('Falling back to pattern-based analysis...');
            
            // Fallback to pattern-based analysis
            return this.fallbackAnalysis(text);
        }
    }

    /**
     * Create prompt for misinformation analysis
     * @param {string} text - Text to analyze
     * @returns {string} - Formatted prompt
     */
    createMisinformationPrompt(text) {
        return `Analyze the following text for misinformation, false claims, and misleading content:

Text: "${text}"

Please evaluate for:
1. Conspiracy theories or unproven claims
2. Pseudoscience or false scientific claims
3. Clickbait or sensationalist language
4. Unverified claims or lack of sources
5. Misleading or deceptive content
6. Other forms of misinformation

Respond with a JSON object like this:
{
  "hasMisinformation": true/false,
  "misinformationScore": 0.0-1.0,
  "severity": "none/low/medium/high/critical",
  "categories": ["conspiracy_theories", "pseudoscience", "clickbait"],
  "flaggedPhrases": ["phrase1", "phrase2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}`;
    }

    /**
     * Parse AI response into structured format
     * @param {string} aiResponse - Raw AI response
     * @param {string} originalText - Original text analyzed
     * @returns {Object} - Parsed analysis
     */
    parseAIResponse(aiResponse, originalText) {
        try {
            // Extract JSON from AI response
            const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
            if (!jsonMatch) {
                throw new Error('No JSON found in AI response');
            }

            const parsed = JSON.parse(jsonMatch[0]);
            
            return {
                text: originalText,
                hasMisinformation: parsed.hasMisinformation || false,
                misinformationScore: parsed.misinformationScore || 0,
                severity: parsed.severity || 'none',
                categories: parsed.categories || [],
                flaggedPhrases: parsed.flaggedPhrases || [],
                recommendations: parsed.recommendations || [],
                model: this.model,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Failed to parse AI response:', error);
            return {
                text: originalText,
                error: 'Failed to parse AI response',
                hasMisinformation: false,
                misinformationScore: 0,
                severity: 'error',
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Fallback analysis using pattern matching when API is unavailable
     * @param {string} text - Text to analyze
     * @returns {Object} - Fallback analysis results
     */
    fallbackAnalysis(text) {
        try {
            const lowerText = text.toLowerCase();
            const words = lowerText.split(/\s+/);
            
            const analysis = {
                text: text,
                hasMisinformation: false,
                misinformationScore: 0,
                severity: 'none',
                categories: {},
                flaggedPhrases: [],
                recommendations: [],
                model: 'fallback-pattern-matcher',
                timestamp: new Date().toISOString()
            };

            // Check for factual errors first (highest priority)
            let factualErrorFound = false;
            for (const [category, errors] of Object.entries(this.factualErrors)) {
                for (const [errorPattern, correction] of Object.entries(errors)) {
                    if (lowerText.includes(errorPattern)) {
                        factualErrorFound = true;
                        analysis.hasMisinformation = true;
                        analysis.misinformationScore = Math.max(analysis.misinformationScore, correction.score);
                        analysis.categories[category] = {
                            score: correction.score,
                            matches: [errorPattern],
                            count: 1,
                            correction: correction.correct
                        };
                        analysis.flaggedPhrases.push(errorPattern);
                        analysis.recommendations.push(`Factual error detected: ${correction.correct}`);
                    }
                }
            }

            // Analyze each category for pattern-based misinformation
            for (const [category, keywords] of Object.entries(this.fallbackPatterns)) {
                const matches = [];
                let categoryScore = 0;

                for (const word of words) {
                    if (keywords.some(keyword => word.includes(keyword))) {
                        matches.push(word);
                        categoryScore += 0.2;
                    }
                }

                if (matches.length > 0) {
                    analysis.categories[category] = {
                        score: Math.min(categoryScore, 1.0),
                        matches: matches,
                        count: matches.length
                    };
                    analysis.flaggedPhrases.push(...matches);
                }
            }

            // Calculate overall misinformation score
            const categoryScores = Object.values(analysis.categories).map(cat => cat.score);
            analysis.misinformationScore = categoryScores.length > 0 
                ? Math.min(categoryScores.reduce((sum, score) => sum + score, 0) / categoryScores.length, 1.0)
                : 0;

            // Determine severity level
            if (analysis.misinformationScore >= 0.8) {
                analysis.severity = 'critical';
                analysis.hasMisinformation = true;
            } else if (analysis.misinformationScore >= 0.6) {
                analysis.severity = 'high';
                analysis.hasMisinformation = true;
            } else if (analysis.misinformationScore >= 0.4) {
                analysis.severity = 'medium';
                analysis.hasMisinformation = true;
            } else if (analysis.misinformationScore >= 0.2) {
                analysis.severity = 'low';
                analysis.hasMisinformation = true;
            }

            // Generate recommendations
            if (analysis.hasMisinformation) {
                if (!factualErrorFound) {
                    analysis.recommendations.push('Misinformation detected. Verify claims with reliable sources.');
                }
            }

            return analysis;
        } catch (error) {
            console.error('Fallback analysis error:', error);
            return {
                text: text,
                error: 'Both API and fallback analysis failed',
                hasMisinformation: false,
                misinformationScore: 0,
                severity: 'error',
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Get agent information
     * @returns {Object} - Agent metadata
     */
    getName() {
        return this.name;
    }

    getDescription() {
        return this.description;
    }

    isReady() {
        return this.isReady;
    }

    getCapabilities() {
        return {
            model: this.model,
            provider: 'Cerebras',
            features: ['ai_analysis', 'misinformation_detection', 'fact_checking_guidance'],
            accuracy: 'High (32B parameter model)',
            context: 'Understands false claims and misleading content'
        };
    }

    /**
     * Update misinformation patterns
     * @param {Object} newPatterns - New misinformation patterns
     */
    updatePatterns(newPatterns) {
        this.misinformationPatterns = { ...this.misinformationPatterns, ...newPatterns };
    }

    /**
     * Update red flags
     * @param {Object} newRedFlags - New red flag indicators
     */
    updateRedFlags(newRedFlags) {
        this.redFlags = { ...this.redFlags, ...newRedFlags };
    }

    /**
     * Update severity thresholds
     * @param {Object} newThresholds - New severity thresholds
     */
    updateSeverityLevels(newThresholds) {
        this.severityLevels = { ...this.severityLevels, ...newThresholds };
    }
}

module.exports = MisinformationAgent; 