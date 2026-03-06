const express = require('express');
const path = require('path');
require('dotenv').config();

// Import agents
const ToxicityAgent = require('./agents/toxicityAgent');
const BiasAgent = require('./agents/biasAgent');
const MisinformationAgent = require('./agents/misinformationAgent');
const CoordinationAgent = require('./agents/coordinationAgent');

// Import routes
const classifyRoutes = require('./routes/classify');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Initialize agents
const agents = {
    toxicity: new ToxicityAgent(),
    bias: new BiasAgent(),
    misinformation: new MisinformationAgent(),
    coordination: new CoordinationAgent()
};

// Make agents available to routes
app.locals.agents = agents;

// Routes
app.use('/api', classifyRoutes);

// Root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    const agentStatus = {};
    
    // Check each agent's status
    Object.keys(agents).forEach(agentName => {
        agentStatus[agentName] = agents[agentName].isReady() ? 'ready' : 'error';
    });

    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        agents: agentStatus,
        version: process.env.npm_package_version || '1.0.0'
    });
});

// Agent status endpoint
app.get('/api/agents', (req, res) => {
    const agentInfo = {};
    
    Object.keys(agents).forEach(agentName => {
        const agent = agents[agentName];
        agentInfo[agentName] = {
            name: agent.getName(),
            description: agent.getDescription(),
            isReady: agent.isReady(),
            capabilities: agent.getCapabilities()
        };
    });

    res.json(agentInfo);
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Not found',
        message: 'The requested endpoint does not exist'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Emakia Node server running on port ${PORT}`);
    console.log(`Visit http://localhost:${PORT} to access the demo`);
    console.log(`API available at http://localhost:${PORT}/api`);
    
    // Log agent status
    console.log('\nAgent Status:');

    Object.keys(agents).forEach((agentName) => {
        const agent = agents[agentName];
        if (agent && typeof agent.isReady === 'function') {
          const status = agent.isReady() ? '✓ Ready' : '✗ Error';
          console.log(`${agentName}: ${status}`);
        } else {
          console.warn(`${agentName} is missing isReady() or is not properly initialized`);
        }
      });
      

}); 