// Simple test script to verify fallback mechanism
const ToxicityAgent = require('./agents/toxicityAgent');
const BiasAgent = require('./agents/biasAgent');

async function testFallback() {
    console.log('Testing fallback mechanism...\n');
    
    const toxicityAgent = new ToxicityAgent();
    const biasAgent = new BiasAgent();
    
    const testText = "you are stupid";
    
    console.log(`Testing text: "${testText}"\n`);
    
    // Test toxicity agent
    console.log('=== Toxicity Analysis ===');
    const toxicityResult = await toxicityAgent.analyze(testText);
    console.log('Result:', JSON.stringify(toxicityResult, null, 2));
    console.log('\n');
    
    // Test bias agent
    console.log('=== Bias Analysis ===');
    const biasResult = await biasAgent.analyze(testText);
    console.log('Result:', JSON.stringify(biasResult, null, 2));
    console.log('\n');
    
    console.log('Test completed!');
}

testFallback().catch(console.error); 