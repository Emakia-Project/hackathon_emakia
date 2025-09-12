const biasAgent = require('../agents/biasAgent');
const coordinationAgent = require('../agents/coordinationAgent');
const misinformationAgent = require('../agents/misinformationAgent');
const toxicityAgent = require('../agents/toxicityAgent');
const toxicityAgentAI = require('../agents/toxicityAgentAI');

function classifyPost(post) {
  const { title, content, metadata = {} } = post;

  const bias = biasAgent.analyze(content);
  const coordination = coordinationAgent.analyze(content);
  const misinformation = misinformationAgent.analyze(content);
  const toxicity = toxicityAgent.analyze(content);
  const advancedToxicity = toxicityAgentAI.evaluate(content);

  const scoreWeights = {
    bias: 0.2,
    coordination: 0.2,
    misinformation: 0.3,
    toxicity: 0.15,
    advancedToxicity: 0.15,
  };

  const trustScore = 1 - (
    bias.score * scoreWeights.bias +
    coordination.score * scoreWeights.coordination +
    misinformation.score * scoreWeights.misinformation +
    toxicity.score * scoreWeights.toxicity +
    advancedToxicity.score * scoreWeights.advancedToxicity
  );

  const flags = [];
  if (bias.flagged) flags.push('Bias');
  if (coordination.flagged) flags.push('Coordination');
  if (misinformation.flagged) flags.push('Misinformation');
  if (toxicity.flagged || advancedToxicity.flagged) flags.push('Toxicity');

  return {
    id: metadata.id || null,
    title,
    trustScore: Math.max(0, Math.min(1, trustScore.toFixed(2))),
    flaggedCategories: flags,
    analysis: {
      bias,
      coordination,
      misinformation,
      toxicity,
      advancedToxicity
    }
  };
}

module.exports = classifyPost;
