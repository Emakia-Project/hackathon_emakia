const express = require('express');
const router = express.Router();

/**
 * GET /api/classify - Classify text using query parameter
 */
router.get('/classify', async (req, res) => {
    const { text, agents } = req.query;

    if (!text) {
        return res.json({
            message: 'Please provide text to analyze. Use ?text=your message or POST with JSON body.',
            availableAgents: ['toxicity', 'bias', 'misinformation', 'coordination'],
            example: '/api/classify?text=your message here&agents=toxicity,bias'
        });
    }

    try {
        const results = await analyzeWithAgents(text, agents, req);
        res.status(200).json(results);
    } catch (error) {
        console.error('Error during GET classification:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST /api/classify - Classify text using JSON body
 */
router.post('/classify', async (req, res) => {
    try {
        const { text, agents } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({ 
                error: 'Text is required and must be a string.',
                example: { text: 'your message here', agents: ['toxicity', 'bias'] }
            });
        }

        const results = await analyzeWithAgents(text, agents, req);
        res.status(200).json(results);
    } catch (error) {
        console.error('Error during POST classification:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST /api/classify/toxicity - Analyze text for toxicity only
 */
router.post('/classify/toxicity', async (req, res) => {
    try {
        const { text } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({ error: 'Text is required and must be a string.' });
        }

        const toxicityAgent = req.app.locals.agents.toxicity;
        const analysis = await toxicityAgent.analyze(text);

        res.status(200).json({
            text,
            analysis,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Toxicity analysis error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST /api/classify/bias - Analyze text for bias only
 */
router.post('/classify/bias', async (req, res) => {
    try {
        const { text } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({ error: 'Text is required and must be a string.' });
        }

        const biasAgent = req.app.locals.agents.bias;
        const analysis = await biasAgent.analyze(text);

        res.status(200).json({
            text,
            analysis,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Bias analysis error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST /api/classify/misinformation - Analyze text for misinformation only
 */
router.post('/classify/misinformation', async (req, res) => {
    try {
        const { text } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({ error: 'Text is required and must be a string.' });
        }

        const misinformationAgent = req.app.locals.agents.misinformation;
        const analysis = await misinformationAgent.analyze(text);

        res.status(200).json({
            text,
            analysis,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Misinformation analysis error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST /api/classify/coordination - Analyze text for coordination only
 */
router.post('/classify/coordination', async (req, res) => {
    try {
        const { text } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({ error: 'Text is required and must be a string.' });
        }

        const coordinationAgent = req.app.locals.agents.coordination;
        const analysis = await coordinationAgent.analyze(text);

        res.status(200).json({
            text,
            analysis,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Coordination analysis error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * Helper function to analyze text with specified agents
 * @param {string} text - Text to analyze
 * @param {string|Array} agents - Agents to use for analysis
 * @returns {Promise<Object>} - Analysis results
 */
async function analyzeWithAgents(text, agents, req) {
    const availableAgents = req.app.locals.agents;
    const results = {
        text,
        timestamp: new Date().toISOString(),
        analyses: {},
        summary: {
            totalIssues: 0,
            criticalIssues: 0,
            highIssues: 0,
            mediumIssues: 0,
            lowIssues: 0
        }
    };

    // Determine which agents to use
    let agentsToUse = Object.keys(availableAgents);
    
    if (agents) {
        if (typeof agents === 'string') {
            agentsToUse = agents.split(',').map(a => a.trim());
        } else if (Array.isArray(agents)) {
            agentsToUse = agents;
        }
    }

    // Filter to only available agents
    agentsToUse = agentsToUse.filter(agent => availableAgents[agent]);

    // Run analysis with each agent
    for (const agentName of agentsToUse) {
        try {
            const agent = availableAgents[agentName];
            const analysis = await agent.analyze(text);
            
            results.analyses[agentName] = analysis;

            // Update summary
            if (analysis.severity === 'critical') {
                results.summary.criticalIssues++;
                results.summary.totalIssues++;
            } else if (analysis.severity === 'high') {
                results.summary.highIssues++;
                results.summary.totalIssues++;
            } else if (analysis.severity === 'medium') {
                results.summary.mediumIssues++;
                results.summary.totalIssues++;
            } else if (analysis.severity === 'low') {
                results.summary.lowIssues++;
                results.summary.totalIssues++;
            }
        } catch (error) {
            console.error(`Error analyzing with ${agentName}:`, error);
            results.analyses[agentName] = {
                error: error.message,
                severity: 'error'
            };
        }
    }

    // Add overall risk assessment
    results.overallRisk = calculateOverallRisk(results.summary);

    return results;
}

/**
 * Calculate overall risk level based on summary
 * @param {Object} summary - Summary of issues
 * @returns {string} - Overall risk level
 */
function calculateOverallRisk(summary) {
    if (summary.criticalIssues > 0) {
        return 'critical';
    } else if (summary.highIssues > 0) {
        return 'high';
    } else if (summary.mediumIssues > 0) {
        return 'medium';
    } else if (summary.lowIssues > 0) {
        return 'low';
    } else {
        return 'none';
    }
}

module.exports = router; 