class ToxicityAgent {
    constructor() {
        this.name = 'ToxicityAgent';
        this.description = 'Real AI toxicity detection using Cerebras Qwen-3-32B model';
        this.cerebrasApiKey = process.env.CEREBRAS_API_KEY;
        this.cerebrasApiUrl = process.env.CEREBRAS_API_URL || 'https://api.cerebras.com';
        this.model = 'Qwen-3-32B';
        this.isReady = !!this.cerebrasApiKey;
        
        // Fallback patterns for when API is unavailable
        this.fallbackPatterns = {
            insults: ['stupid', 'idiot', 'dumb', 'fool', 'moron', 'imbecile', 'retard', 'cretin'],
            hate_speech: ['hate', 'racist', 'bigot', 'nazi', 'supremacist', 'white power', 'black power'],
            violence: ['kill', 'murder', 'attack', 'violence', 'weapon', 'bomb', 'shoot', 'stab'],
            profanity: ['fuck', 'shit', 'damn', 'hell', 'bitch', 'ass', 'cunt', 'dick', 'pussy'],
            threats: ['threaten', 'hurt you', 'kill you', 'attack you', 'destroy you'],
            harassment: ['harass', 'stalk', 'bully', 'intimidate', 'terrorize']
        };
    }

    /**
     * Analyze text for toxicity using Cerebras Qwen-3-32B
     * @param {string} text - Text to analyze
     * @returns {Promise<Object>} - Toxicity analysis results
     */
    async analyze(text) {
        try {
            if (!this.cerebrasApiKey) {
                console.log('No Cerebras API key configured, using fallback analysis...');
                return this.fallbackAnalysis(text);
            }

            // Create prompt for toxicity analysis
            const prompt = this.createToxicityPrompt(text);
            
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
                            content: 'You are a toxicity detection expert. Analyze the given text and determine if it contains toxic, harmful, or inappropriate content. Respond with a JSON object containing: isToxic (boolean), toxicityScore (0-1), severity (none/low/medium/high/critical), categories (array of detected issues), and recommendations (array of suggestions).'
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
            console.error('Cerebras toxicity analysis error:', error);
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
                isToxic: false,
                toxicityScore: 0,
                severity: 'none',
                categories: {},
                flaggedWords: [],
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
                        categoryScore += 0.3;
                    }
                }

                if (matches.length > 0) {
                    analysis.categories[category] = {
                        score: Math.min(categoryScore, 1.0),
                        matches: matches,
                        count: matches.length
                    };
                    analysis.flaggedWords.push(...matches);
                }
            }

            // Calculate overall toxicity score
            const categoryScores = Object.values(analysis.categories).map(cat => cat.score);
            analysis.toxicityScore = categoryScores.length > 0 
                ? Math.min(categoryScores.reduce((sum, score) => sum + score, 0) / categoryScores.length, 1.0)
                : 0;

            // Determine severity level
            if (analysis.toxicityScore >= 0.8) {
                analysis.severity = 'critical';
                analysis.isToxic = true;
            } else if (analysis.toxicityScore >= 0.6) {
                analysis.severity = 'high';
                analysis.isToxic = true;
            } else if (analysis.toxicityScore >= 0.4) {
                analysis.severity = 'medium';
                analysis.isToxic = true;
            } else if (analysis.toxicityScore >= 0.2) {
                analysis.severity = 'low';
                analysis.isToxic = true;
            }

            // Generate recommendations
            if (analysis.isToxic) {
                analysis.recommendations.push('Toxic content detected. Consider moderation.');
                if (analysis.categories.insults) {
                    analysis.recommendations.push('Personal insults detected. Review for community guidelines.');
                }
            }

            return analysis;
        } catch (error) {
            console.error('Fallback analysis error:', error);
            return {
                text: text,
                error: 'Both API and fallback analysis failed',
                isToxic: false,
                toxicityScore: 0,
                severity: 'error',
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Create prompt for toxicity analysis
     * @param {string} text - Text to analyze
     * @returns {string} - Formatted prompt
     */
    createToxicityPrompt(text) {
        return `Analyze the following text for toxicity, harm, and inappropriate content:

Text: "${text}"

Please evaluate for:
1. Hate speech or discrimination
2. Violence or threats
3. Harassment or bullying
4. Insults or personal attacks
5. Profanity or vulgar language
6. Other harmful content

Respond with a JSON object like this:
{
  "isToxic": true/false,
  "toxicityScore": 0.0-1.0,
  "severity": "none/low/medium/high/critical",
  "categories": ["hate_speech", "insults", "profanity"],
  "flaggedWords": ["word1", "word2"],
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
                isToxic: parsed.isToxic || false,
                toxicityScore: parsed.toxicityScore || 0,
                severity: parsed.severity || 'none',
                categories: parsed.categories || [],
                flaggedWords: parsed.flaggedWords || [],
                recommendations: parsed.recommendations || [],
                model: this.model,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Failed to parse AI response:', error);
            return {
                text: originalText,
                error: 'Failed to parse AI response',
                isToxic: false,
                toxicityScore: 0,
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
            features: ['ai_analysis', 'context_understanding', 'semantic_analysis'],
            accuracy: 'High (32B parameter model)',
            context: 'Understands meaning and intent, not just keywords'
        };
    }
}

module.exports = ToxicityAgent; 