class CoordinationAgent {
    constructor() {
        this.name = 'CoordinationAgent';
        this.description = 'Real AI coordination detection using Cerebras Qwen-3-32B model';
        this.cerebrasApiKey = process.env.CEREBRAS_API_KEY;
        this.cerebrasApiUrl = process.env.CEREBRAS_API_URL || 'https://api.cerebras.com';
        this.model = 'Qwen-3-32B';
        this.isReady = !!this.cerebrasApiKey;
        
        // Fallback patterns for when API is unavailable
        this.fallbackPatterns = {
            bot_like_behavior: ['automated', 'bot', 'script', 'program', 'algorithm'],
            coordinated_campaigns: ['campaign', 'movement', 'initiative', 'drive', 'push'],
            manipulation_tactics: ['astroturfing', 'brigading', 'vote manipulation'],
            amplification: ['share this', 'retweet', 'repost', 'forward', 'viral'],
            echo_chambers: ['echo chamber', 'bubble', 'filter', 'algorithm']
        };
    }

    /**
     * Analyze text for coordination using Cerebras Qwen-3-32B
     * @param {string} text - Text to analyze
     * @returns {Promise<Object>} - Coordination analysis results
     */
    async analyze(text) {
        try {
            if (!this.cerebrasApiKey) {
                console.log('No Cerebras API key configured, using fallback analysis...');
                return this.fallbackAnalysis(text);
            }

            // Create prompt for coordination analysis
            const prompt = this.createCoordinationPrompt(text);
            
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
                            content: 'You are a coordination detection expert. Analyze the given text and determine if it contains coordinated behavior, manipulation tactics, or inauthentic activity patterns. Respond with a JSON object containing: hasCoordination (boolean), coordinationScore (0-1), severity (none/low/medium/high/critical), categories (array of detected issues), and recommendations (array of suggestions).'
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
            console.error('Cerebras coordination analysis error:', error);
            console.log('Falling back to pattern-based analysis...');
            
            // Fallback to pattern-based analysis
            return this.fallbackAnalysis(text);
        }
    }

    /**
     * Create prompt for coordination analysis
     * @param {string} text - Text to analyze
     * @returns {string} - Formatted prompt
     */
    createCoordinationPrompt(text) {
        return `Analyze the following text for coordinated behavior, manipulation, and inauthentic activity:

Text: "${text}"

Please evaluate for:
1. Bot-like behavior or automated content
2. Coordinated campaigns or organized manipulation
3. Manipulation tactics (astroturfing, brigading, etc.)
4. Amplification attempts or viral manipulation
5. Echo chamber indicators
6. Other inauthentic activity patterns

Respond with a JSON object like this:
{
  "hasCoordination": true/false,
  "coordinationScore": 0.0-1.0,
  "severity": "none/low/medium/high/critical",
  "categories": ["bot_like_behavior", "coordinated_campaigns", "manipulation_tactics"],
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
                hasCoordination: parsed.hasCoordination || false,
                coordinationScore: parsed.coordinationScore || 0,
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
                hasCoordination: false,
                coordinationScore: 0,
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
                hasCoordination: false,
                coordinationScore: 0,
                severity: 'none',
                categories: {},
                flaggedPhrases: [],
                recommendations: [],
                model: 'fallback-pattern-matcher',
                timestamp: new Date().toISOString()
            };

            // Analyze each category
            for (const [category, keywords] of Object.entries(this.fallbackPatterns)) {
                const matches = [];
                let categoryScore = 0;

                for (const word of words) {
                    if (keywords.some(keyword => word.includes(keyword))) {
                        matches.push(word);
                        categoryScore += 0.15;
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

            // Calculate overall coordination score
            const categoryScores = Object.values(analysis.categories).map(cat => cat.score);
            analysis.coordinationScore = categoryScores.length > 0 
                ? Math.min(categoryScores.reduce((sum, score) => sum + score, 0) / categoryScores.length, 1.0)
                : 0;

            // Determine severity level
            if (analysis.coordinationScore >= 0.8) {
                analysis.severity = 'critical';
                analysis.hasCoordination = true;
            } else if (analysis.coordinationScore >= 0.6) {
                analysis.severity = 'high';
                analysis.hasCoordination = true;
            } else if (analysis.coordinationScore >= 0.4) {
                analysis.severity = 'medium';
                analysis.hasCoordination = true;
            } else if (analysis.coordinationScore >= 0.2) {
                analysis.severity = 'low';
                analysis.hasCoordination = true;
            }

            // Generate recommendations
            if (analysis.hasCoordination) {
                analysis.recommendations.push('Coordinated behavior detected. Monitor for inauthentic activity.');
            }

            return analysis;
        } catch (error) {
            console.error('Fallback analysis error:', error);
            return {
                text: text,
                error: 'Both API and fallback analysis failed',
                hasCoordination: false,
                coordinationScore: 0,
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
            features: ['ai_analysis', 'coordination_detection', 'manipulation_identification'],
            accuracy: 'High (32B parameter model)',
            context: 'Understands coordinated behavior and manipulation patterns'
        };
    }

    /**
     * Update coordination patterns
     * @param {Object} newPatterns - New coordination patterns
     */
    updatePatterns(newPatterns) {
        this.coordinationPatterns = { ...this.coordinationPatterns, ...newPatterns };
    }

    /**
     * Update manipulation indicators
     * @param {Object} newIndicators - New manipulation indicators
     */
    updateIndicators(newIndicators) {
        this.manipulationIndicators = { ...this.manipulationIndicators, ...newIndicators };
    }

    /**
     * Update severity thresholds
     * @param {Object} newThresholds - New severity thresholds
     */
    updateSeverityLevels(newThresholds) {
        this.severityLevels = { ...this.severityLevels, ...newThresholds };
    }
}

module.exports = CoordinationAgent; 