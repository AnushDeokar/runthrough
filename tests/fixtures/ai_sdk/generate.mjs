// Records real Vercel AI SDK UI-message streams for runthrough's adapter tests.
// A mock language model drives streamText, so no API key or network is needed.
//
//   AI_SDK_PROJECT=/path/to/a/project/with/ai-v5 OUT=tests/fixtures/ai_sdk node generate.mjs
//
// AI_SDK_PROJECT must contain node_modules with `ai` (v5) and `zod`.
import { createRequire } from 'node:module';
import { writeFileSync, mkdirSync } from 'node:fs';
const require = createRequire(process.env.AI_SDK_PROJECT + '/package.json');
const { streamText, stepCountIs, tool, simulateReadableStream } = require('ai');
const { z } = require('zod');

const usage = { inputTokens: 10, outputTokens: 5, totalTokens: 15 };
const text = (id, s) => [
  { type: 'text-start', id },
  ...s.match(/.{1,12}/gs).map((d) => ({ type: 'text-delta', id, delta: d })),
  { type: 'text-end', id },
];
const finish = (r) => ({ type: 'finish', finishReason: r, usage });
const call = (id, name, input) => ({ type: 'tool-call', toolCallId: id, toolName: name, input: JSON.stringify(input) });

function model(steps) {
  let i = 0;
  return {
    specificationVersion: 'v2', provider: 'mock', modelId: 'mock', supportedUrls: {},
    doGenerate: async () => { throw new Error('not used'); },
    doStream: async () => {
      const s = steps[i++];
      if (s instanceof Error) throw s;
      return { stream: simulateReadableStream({ chunks: [{ type: 'stream-start', warnings: [] }, ...s] }) };
    },
  };
}

const tools = {
  retrieval: tool({
    description: 'Search the knowledge base',
    inputSchema: z.object({ query: z.string() }),
    execute: async ({ query }) => ({ query, chunks: [{ text: 'FS1 annual fees are AED 52,000.', source: 'fees-2026.pdf' }] }),
  }),
  check_meeting_availability: tool({
    description: 'List free slots',
    inputSchema: z.object({ date: z.string() }),
    execute: async ({ date }) => ({ date, slots: ['10:00', '14:30'] }),
  }),
  request_human_handoff: tool({
    description: 'Hand off to a person',
    inputSchema: z.object({ reason: z.string() }),
    execute: async () => ({ notified: true }),
  }),
  book_meeting: tool({
    description: 'Book a slot',
    inputSchema: z.object({ date: z.string(), time: z.string() }),
    execute: async () => { throw new Error('Slot no longer available'); },
  }),
};

async function record(name, steps, opts = {}) {
  const result = streamText({ model: model(steps), prompt: 'hi', tools, stopWhen: stepCountIs(5), ...opts });
  const res = result.toUIMessageStreamResponse({ generateMessageId: () => 'msg-assistant-1' });
  const body = await res.text();
  writeFileSync(`${process.env.OUT}/${name}.sse`, body);
  console.log(name, res.headers.get('content-type'), res.headers.get('x-vercel-ai-ui-message-stream'), body.length);
}

mkdirSync(process.env.OUT, { recursive: true });
await record('text_only', [[...text('t1', 'Hello! How can I help you with admissions today?'), finish('stop')]]);
await record('tool_then_text', [
  [call('call_r1', 'retrieval', { query: 'FS1 fees' }), finish('tool-calls')],
  [...text('t2', 'Annual fees for FS1 are AED 52,000.'), finish('stop')],
]);
await record('two_steps_tool_error', [
  [call('call_a1', 'check_meeting_availability', { date: '2026-11-03' }), finish('tool-calls')],
  [...text('t2', 'Let me book 10:00 for you.'), call('call_b1', 'book_meeting', { date: '2026-11-03', time: '10:00' }), finish('tool-calls')],
  [...text('t3', "Sorry, that slot was just taken. Would 14:30 work?"), finish('stop')],
]);
await record('model_error', [new Error('upstream model overloaded')]);
await record('streamed_tool_input_with_reasoning', [
  [
    { type: 'reasoning-start', id: 'r1' },
    { type: 'reasoning-delta', id: 'r1', delta: 'The parent wants a person.' },
    { type: 'reasoning-end', id: 'r1' },
    { type: 'tool-input-start', id: 'call_h1', toolName: 'request_human_handoff' },
    { type: 'tool-input-delta', id: 'call_h1', delta: '{"reason":' },
    { type: 'tool-input-delta', id: 'call_h1', delta: '"asked for admissions"}' },
    { type: 'tool-input-end', id: 'call_h1' },
    call('call_h1', 'request_human_handoff', { reason: 'asked for admissions' }),
    finish('tool-calls'),
  ],
  [...text('t2', "I've passed your request to our admissions team."), finish('stop')],
]);
