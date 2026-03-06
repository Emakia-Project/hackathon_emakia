import { createApp } from "vinxi";
import { config as vinxiConfig } from "vinxi/plugins/config";
import tsConfigPathsPlugin from "vite-tsconfig-paths";
import { tanstackRouter as tanstackRouterPlugin } from "@tanstack/router-plugin/vite";
import reactRefreshPlugin from "@vitejs/plugin-react";
import { nodePolyfills as nodePolyfillsPlugin } from "vite-plugin-node-polyfills";
import { consoleForwardPlugin } from "./vite-console-forward-plugin";


// Simple environment check
const BASE_URL = process.env.BASE_URL;

export default createApp({
  server: {
    preset: "node-server",
    experimental: {
      asyncContext: true,
    },
  },
  watch: {
    ignored: ["**/app.config.timestamp_*.js"],
  },
  routers: [
    {
      type: "static",
      name: "public",
      dir: "./public",
    },
    {
      type: "http",
      name: "api",
      handler: "./src/routes/api",
      target: "server",
    },
    {
      type: "spa",
      name: "client",
      handler: "./index.html",
      target: "browser",
      plugins: () => {
        const plugins = [];

        plugins.push(vinxiConfig("allowedHosts", {
          server: {
            allowedHosts: BASE_URL ? [BASE_URL.split("://")[1]] : undefined,
          },
        }));

        plugins.push(tsConfigPathsPlugin({ projects: ["./tsconfig.json"] }));

        plugins.push(tanstackRouterPlugin({
          target: "react",
          autoCodeSplitting: true,
          routesDirectory: "./src/routes",
          generatedRouteTree: "./src/generated/routeTree.gen.ts",
        }));

        plugins.push(reactRefreshPlugin());

        plugins.push(nodePolyfillsPlugin());

        plugins.push(consoleForwardPlugin());

        return plugins;
      },
    },
  ],
});
