import { defineEventHandler, toWebRequest } from "@tanstack/react-start/server";

export default defineEventHandler(async (event) => {
  const request = toWebRequest(event);
  if (!request) {
    return new Response("No request", { status: 400 });
  }

  const url = new URL(request.url);
  const path = url.pathname.replace("/api/", "");
  const method = request.method;

  if (path === "test") {
    if (method === "GET") {
      return new Response(
        JSON.stringify({
          ok: true,
          message: "API route is working",
          timestamp: new Date().toISOString(),
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
    }
    
    if (method === "POST") {
      return new Response(
        JSON.stringify({
          ok: true,
          message: "POST method working",
          timestamp: new Date().toISOString(),
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
    }
    
    return new Response("Method not allowed", { status: 405 });
  }
  
  if (path === "check-misinformation") {
    if (method === "GET") {
      return new Response(
        JSON.stringify({
          ok: true,
          message: "Misinformation check API is working",
          timestamp: new Date().toISOString(),
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
    }
    
    return new Response("Method not allowed", { status: 405 });
  }
  
  return new Response("API endpoint not found", { status: 404 });
});