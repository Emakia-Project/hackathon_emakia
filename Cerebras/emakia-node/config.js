// Configuration for Cerebras API integration
module.exports = {
    cerebras: {
        apiKey: process.env.CEREBRAS_API_KEY,
        apiUrl: process.env.CEREBRAS_API_URL || 'https://api.cerebras.com',
        model: process.env.CEREBRAS_MODEL || 'Qwen-3-32B',
        timeout: parseInt(process.env.CEREBRAS_TIMEOUT) || 30000
    },
    server: {
        port: process.env.PORT || 3000,
        environment: process.env.NODE_ENV || 'development'
    },
    agents: {
        timeout: parseInt(process.env.AGENT_TIMEOUT) || 10000,
        retryAttempts: parseInt(process.env.AGENT_RETRY_ATTEMPTS) || 3
    }
}; 