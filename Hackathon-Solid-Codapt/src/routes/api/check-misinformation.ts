export async function GET() {
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

export async function POST() {
  return new Response("Method not implemented", { status: 501 });
}

