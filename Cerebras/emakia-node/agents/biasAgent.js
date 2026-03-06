class BiasAgent {
    constructor() {
        this.name = 'BiasAgent';
        this.description = 'Real AI bias detection using Cerebras Qwen-3-32B model';
        this.cerebrasApiKey = process.env.CEREBRAS_API_KEY;
        this.cerebrasApiUrl = process.env.CEREBRAS_API_URL || 'https://api.cerebras.com';
        this.model = 'Qwen-3-32B';
        this.isReady = !!this.cerebrasApiKey;
        
        // Fallback patterns for when API is unavailable
        this.fallbackPatterns = {
            gender_bias: ['women can\'t', 'men are better', 'female driver', 'male nurse'],
            racial_bias: ['race', 'ethnicity', 'minority', 'stereotype', 'prejudice'],
            age_bias: ['old people', 'young people', 'millennial', 'boomer'],
            religious_bias: ['religion', 'faith', 'atheist', 'believer', 'heathen'],
            socioeconomic_bias: ['poor', 'rich', 'wealthy', 'poverty', 'privilege']
        };
    }

    /**
     * Analyze text for bias using Cerebras Qwen-3-32B
     * @param {string} text - Text to analyze
     * @returns {Promise<Object>} - Bias analysis results
     */
    async analyze(text) {
        try {
            if (!this.cerebrasApiKey) {
                console.log('No Cerebras API key configured, using fallback analysis...');
                return this.fallbackAnalysis(text);
            }

            // Create prompt for bias analysis
            const prompt = this.createBiasPrompt(text);
            
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
                            content: 'You are a bias detection expert. Analyze the given text and determine if it contains bias, discrimination, or unfair content. Respond with a JSON object containing: hasBias (boolean), biasScore (0-1), severity (none/low/medium/high/critical), categories (array of detected bias types), and recommendations (array of suggestions).'
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
            console.error('Cerebras bias analysis error:', error);
            console.log('Falling back to pattern-based analysis...');
            
            // Fallback to pattern-based analysis
            return this.fallbackAnalysis(text);
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
                hasBias: false,
                biasScore: 0,
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

            // Calculate overall bias score
            const categoryScores = Object.values(analysis.categories).map(cat => cat.score);
            analysis.biasScore = categoryScores.length > 0 
                ? Math.min(categoryScores.reduce((sum, score) => sum + score, 0) / categoryScores.length, 1.0)
                : 0;

            // Determine severity level
            if (analysis.biasScore >= 0.8) {
                analysis.severity = 'critical';
                analysis.hasBias = true;
            } else if (analysis.biasScore >= 0.6) {
                analysis.severity = 'high';
                analysis.hasBias = true;
            } else if (analysis.biasScore >= 0.4) {
                analysis.severity = 'medium';
                analysis.hasBias = true;
            } else if (analysis.biasScore >= 0.2) {
                analysis.severity = 'low';
                analysis.hasBias = true;
            }

            // Generate recommendations
            if (analysis.hasBias) {
                analysis.recommendations.push('Bias detected. Consider using more inclusive language.');
            }

            return analysis;
        } catch (error) {
            console.error('Fallback analysis error:', error);
            return {
                text: text,
                error: 'Both API and fallback analysis failed',
                hasBias: false,
                biasScore: 0,
                severity: 'error',
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Create prompt for bias analysis
     * @param {string} text - Text to analyze
     * @returns {string} - Formatted prompt
     */
    createBiasPrompt(text) {
        return `Analyze the following text for bias, discrimination, and unfair content:

Text: "${text}"

Please evaluate for:
1. Gender bias or stereotypes
2. Racial or ethnic bias
3. Age-based discrimination
4. Religious bias
5. Socioeconomic bias
6. Other forms of discrimination

Respond with a JSON object like this:
{
  "hasBias": true/false,
  "biasScore": 0.0-1.0,
  "severity": "none/low/medium/high/critical",
  "categories": ["gender_bias", "racial_bias", "age_bias"],
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
                hasBias: parsed.hasBias || false,
                biasScore: parsed.biasScore || 0,
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
                hasBias: false,
                biasScore: 0,
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
            features: ['ai_analysis', 'bias_detection', 'discrimination_identification'],
            accuracy: 'High (32B parameter model)',
            context: 'Understands subtle bias and discrimination patterns'
        };
    }
}

module.exports = BiasAgent; 