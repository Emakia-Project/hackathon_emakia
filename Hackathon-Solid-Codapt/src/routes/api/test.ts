// File: ./src/routes/api/test.ts
// This creates an API endpoint at /api/test

export async function GET() {
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

export async function POST() {
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

export async function PUT() {
  return new Response("Method not implemented", { status: 501 });
}

export async function DELETE() {
  return new Response("Method not implemented", { status: 501 });
}
