import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return (
    <div className="p-2">
      {/* Remove this when adding actual content/functionality */}
      <p>Actual app content will go here</p>
    </div>
  );
}
