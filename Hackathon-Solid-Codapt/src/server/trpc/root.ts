import {
  createCallerFactory,
  createTRPCRouter,
  baseProcedure,
} from "~/server/trpc/main";

export const appRouter = createTRPCRouter({
  // when we add a real procedure, remove this placeholder
  placeholderProcedure: baseProcedure.query(() => {
    return "placeholder";
  }),
});

export type AppRouter = typeof appRouter;

export const createCaller = createCallerFactory(appRouter);
