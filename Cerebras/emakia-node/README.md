# Emakia Node

Advanced content analysis platform with multiple AI agents for detecting toxicity, bias, misinformation, and coordination patterns in text content.

## Features

- **Real AI Models**: Uses Cerebras Qwen-3-32B for intelligent analysis
- **Multi-Agent Analysis**: Four specialized AI agents working together
- **Toxicity Detection**: AI-powered detection of harmful and inappropriate content
- **Bias Detection**: Intelligent detection of bias and discrimination patterns
- **Misinformation Detection**: AI analysis of false claims and misleading content
- **Coordination Detection**: Detection of coordinated behavior and manipulation
- **Modern Web UI**: Responsive interface with real-time AI analysis
- **RESTful API**: Comprehensive API for integration with other systems
- **Context Understanding**: Real AI models understand meaning, not just keywords

## Project Structure

```
emakia-node/
├── agents/
│   ├── toxicityAgent.js      # Toxicity detection agent
│   ├── biasAgent.js          # Bias detection agent
│   ├── misinformationAgent.js # Misinformation detection agent
│   └── coordinationAgent.js   # Coordination detection agent
├── routes/
│   └── classify.js           # API routes for classification
├── public/
│   └── index.html            # Web interface
├── app.js                    # Main server file
├── package.json              # Dependencies and scripts
└── README.md                # This file
```

## Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd emakia-node
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables (optional for enhanced AI models):**
   ```bash
   # Create .env file
   touch .env
   
   # Add your Cerebras API configuration (optional)
   echo "CEREBRAS_API_KEY=your_cerebras_api_key_here" >> .env
   echo "CEREBRAS_API_URL=https://api.cerebras.com" >> .env
   echo "CEREBRAS_MODEL=Qwen-3-32B" >> .env
   ```
   
   **Note:** The system works without the API key using intelligent fallback detection, but the Cerebras Qwen-3-32B model provides superior accuracy for complex cases.

## Usage

### Development Mode
```bash
npm run dev
```

### Production Mode
```bash
npm start
```

The application will start on `http://localhost:3000`

## API Endpoints

### POST `/api/classify`
Analyze text with all or selected agents.

**Request Body:**
```json
{
  "text": "Your text to analyze",
  "agents": ["toxicity", "bias", "misinformation", "coordination"]
}
```

**Response:**
```json
{
  "text": "Your text to analyze",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "analyses": {
    "toxicity": {
      "isToxic": true,
      "toxicityScore": 0.8,
      "severity": "high",
      "categories": {...},
      "recommendations": [...]
    },
    "bias": {...},
    "misinformation": {...},
    "coordination": {...}
  },
  "summary": {
    "totalIssues": 3,
    "criticalIssues": 1,
    "highIssues": 2,
    "mediumIssues": 0,
    "lowIssues": 0
  },
  "overallRisk": "critical"
}
```

### GET `/api/classify`
Analyze text using query parameters.

**Example:**
```
GET /api/classify?text=your message&agents=toxicity,bias
```

### Individual Agent Endpoints

- `POST /api/classify/toxicity` - Toxicity analysis only
- `POST /api/classify/bias` - Bias analysis only
- `POST /api/classify/misinformation` - Misinformation analysis only
- `POST /api/classify/coordination` - Coordination analysis only

### GET `/api/health`
Check system health and agent status.

### GET `/api/agents`
Get information about available agents.

## Agent Capabilities

### ToxicityAgent
- **Categories**: Hate speech, violence, harassment, profanity
- **Severity Levels**: Low, medium, high, critical
- **Features**: Pattern matching, severity assessment, recommendations

### BiasAgent
- **Categories**: Gender, racial, age, religious, socioeconomic bias
- **Indicators**: Absolute statements, generalizations, stereotypes, loaded language
- **Features**: Bias detection, inclusive language recommendations

### MisinformationAgent
- **Categories**: Conspiracy theories, pseudoscience, clickbait, sensationalism
- **Red Flags**: Emotional language, urgency, authority appeals, exclusivity
- **Features**: Fact-checking guidance, source verification

### CoordinationAgent
- **Categories**: Bot-like behavior, coordinated campaigns, manipulation tactics
- **Indicators**: Urgency pressure, emotional manipulation, social proof
- **Features**: Authenticity verification, manipulation detection

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Server Configuration
PORT=3000
NODE_ENV=development

# API Configuration
API_VERSION=v1
API_TIMEOUT=30000

# Agent Configuration
AGENT_TIMEOUT=10000
AGENT_CONFIDENCE_THRESHOLD=0.6

# Logging Configuration
LOG_LEVEL=info
LOG_FORMAT=combined

# Security Configuration
CORS_ORIGIN=*
RATE_LIMIT_WINDOW=15
RATE_LIMIT_MAX=100
```

### Agent Customization

Each agent can be customized by updating patterns and thresholds:

```javascript
// Example: Update toxicity patterns
const toxicityAgent = new ToxicityAgent();
toxicityAgent.updatePatterns({
  new_category: ['new', 'keywords', 'here']
});

// Example: Update severity thresholds
toxicityAgent.updateSeverityLevels({
  custom: 0.5
});
```

## Web Interface

The web interface provides:

- **Multi-agent selection**: Choose which agents to use
- **Real-time analysis**: Instant results with visual indicators
- **Detailed breakdown**: Individual agent results with scores
- **Recommendations**: Actionable advice for each detected issue
- **Responsive design**: Works on desktop and mobile devices

## Development

### Project Structure Details

- **`app.js`**: Express server with agent initialization and route handling
- **`agents/`**: Individual agent classes with specialized analysis logic
- **`routes/classify.js`**: API endpoints for text classification
- **`public/index.html`**: Modern web interface with JavaScript

### Adding New Agents

1. Create a new agent class in `agents/`
2. Implement the required methods: `analyze()`, `getName()`, `getDescription()`, etc.
3. Add the agent to `app.js`
4. Update the web interface to include the new agent

### Extending Existing Agents

Each agent supports:
- Pattern updates via `updatePatterns()`
- Threshold updates via `updateSeverityLevels()`
- Custom analysis logic in the `analyze()` method

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   PORT=3001 npm start
   ```

2. **Agent errors:**
   - Check agent initialization in `app.js`
   - Verify agent class exports
   - Review console logs for specific errors

3. **API errors:**
   - Verify request format
   - Check required fields (text, agents)
   - Review server logs

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=debug npm start
```

## Security Considerations

- Input validation on all endpoints
- Rate limiting to prevent abuse
- Error handling without exposing sensitive information
- CORS configuration for cross-origin requests

## Performance

- Agent analysis runs asynchronously
- Configurable timeouts for external API calls
- Efficient pattern matching algorithms
- Memory-conscious text processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub

---

**Note**: This is a demonstration platform. For production use, implement proper security measures, error handling, and integrate with real AI models for improved accuracy. 