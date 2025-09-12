const { loadInputs } = require('./inputLoader');

function testInputStructure() {
  const data = loadInputs();

  if (!data.user_query) throw new Error("Missing 'user_query'");
  if (typeof data.context !== 'object') throw new Error("'context' should be an object");
  if (!('trust_score' in data.context)) throw new Error("Missing 'trust_score' in context");

  console.log('✅ Input structure is valid.');
}

try {
  testInputStructure();
} catch (err) {
  console.error('❌ Test failed:', err.message);
}
