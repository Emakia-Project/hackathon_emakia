# Extend your Agent class
from neo4j import GraphDatabase

class Agent:
    def __init__(self, mcp: MCP, neo4j_uri, neo4j_user, neo4j_password):
        self.mcp = mcp
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def analyze_graph_patterns(self, keyword):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Tweet)-[:RETWEETED*1..3]->(related)
                WHERE t.text CONTAINS $keyword
                RETURN related.text AS related_tweet, related.created_at AS time
                ORDER BY time DESC
                LIMIT 5
            """, keyword=keyword)
            return [dict(record) for record in result]

    def close(self):
        self.driver.close()
