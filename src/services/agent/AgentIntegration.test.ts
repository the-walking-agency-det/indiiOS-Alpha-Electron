
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AgentService } from './AgentService';
import { agentRegistry } from './registry';
import { useStore } from '@/core/store';
import { AI } from '@/services/ai/AIService';

// Provide stable environment configuration to avoid validation noise
vi.mock('@/config/env', () => ({
    env: {
        apiKey: 'test-api-key',
        projectId: 'test-project',
        location: 'us-central1',
        useVertex: false,
        googleMapsApiKey: 'test-maps',
        VITE_FUNCTIONS_URL: 'http://localhost/functions',
        VITE_RAG_PROXY_URL: 'http://localhost/rag',
        VITE_GOOGLE_MAPS_API_KEY: 'test-maps',
        DEV: true,
        skipOnboarding: true
    }
}));

// Mock AI Service to control responses and simulate tool usage
vi.mock('@/services/ai/AIService', () => ({
    AI: {
        generateContent: vi.fn(),
        embedContent: vi.fn().mockResolvedValue({ embedding: { values: [0.1, 0.1, 0.1] } }),
        generateContentStream: vi.fn()
    }
}));

const buildStreamResponse = (finalText: string) => ({
    getReader: () => ({
        read: vi.fn()
            // First chunk carries a simple final response payload
            .mockResolvedValueOnce({ done: false, value: { text: () => `{"final_response":"${finalText}"}` } })
            // Then signal completion
            .mockResolvedValue({ done: true, value: undefined })
    })
});

// Prevent Firebase from initializing during unit tests
vi.mock('@/services/firebase', () => ({
    app: {},
    auth: {},
    storage: {},
    functions: {},
    db: {}
}));

// Stub FirestoreService so memory access does not hit real Firebase
vi.mock('@/services/FirestoreService', () => {
    class MockFirestoreService<T> {
        constructor(public collectionPath: string) { }

        add = vi.fn().mockResolvedValue('mock-id');
        update = vi.fn().mockResolvedValue();
        delete = vi.fn().mockResolvedValue();
        get = vi.fn().mockResolvedValue(null);
        list = vi.fn().mockResolvedValue([]);
        query = vi.fn().mockResolvedValue([]);
    }

    return { FirestoreService: MockFirestoreService };
});

// Memory service helpers (used by ContextPipeline)
vi.mock('@/services/agent/MemoryService', () => ({
    memoryService: {
        saveMemory: vi.fn(),
        retrieveRelevantMemories: vi.fn().mockResolvedValue([]),
        clearProjectMemory: vi.fn(),
        regenerateEmbeddings: vi.fn()
    }
}));

// Mock Store
vi.mock('@/core/store', () => ({
    useStore: {
        getState: vi.fn(),
        setState: vi.fn()
    }
}));

describe('Agent Architecture Integration (Hardened)', () => {
    let service: AgentService;
    let mockStoreState: any;

    beforeEach(() => {
        vi.clearAllMocks();

        // Provide a safe default response for any AI calls we don't explicitly mock
        (AI.generateContent as any).mockResolvedValue({
            text: () => '',
            functionCalls: () => []
        });

        (AI.generateContentStream as any).mockResolvedValue(buildStreamResponse(''));

        // Setup complex store state
        mockStoreState = {
            agentHistory: [],
            projects: [{ id: 'p1', name: 'Integration Project', type: 'creative' }],
            currentProjectId: 'p1',
            userProfile: { brandKit: { tone: 'Professional' } },
            addAgentMessage: vi.fn((msg) => mockStoreState.agentHistory.push(msg)),
            updateAgentMessage: vi.fn((id, update) => {
                const msg = mockStoreState.agentHistory.find((m: any) => m.id === id);
                if (msg) Object.assign(msg, update);
            })
        };
        (useStore.getState as any).mockReturnValue(mockStoreState);

        service = new AgentService();
    });

    describe('End-to-End Execution Pipeline', () => {
        it('should correctly orchestrate, execute, and return response for a specialist', async () => {
            // 1. Mock Orchestrator Decision
            (AI.generateContent as any).mockResolvedValueOnce({
                text: () => 'marketing'
            });

            // 2. Mock Specialist Execution (Thinking + Final Answer)
            (AI.generateContent as any).mockResolvedValueOnce({
                text: () => 'I have analyzed the market data.',
                functionCalls: () => []
            });

            await service.sendMessage('Analyze market trends');

            // Verify Orchestrator was called
            expect(AI.generateContent).toHaveBeenNthCalledWith(1, expect.objectContaining({
                contents: expect.objectContaining({ role: 'user' })
            }));

            // Verify Specialist was called with correct context stuffed into user prompt
            // BaseAgent injects system prompt into the user message content
            const secondCallArgs = (AI.generateContent as any).mock.calls[1][0];
            const promptText = secondCallArgs.contents[0].parts[0].text;
            expect(promptText).toContain('Campaign Manager');

            // Verify message history updated
            const lastMsg = mockStoreState.agentHistory[mockStoreState.agentHistory.length - 1];
            expect(lastMsg.role).toBe('model');
            expect(lastMsg.text).toBe('I have analyzed the market data.');
        });

        it('should handle tool execution cycles (Thinking -> Tool -> Response)', async () => {
            vi.clearAllMocks();

            // 1. Router Call
            (AI.generateContent as any).mockResolvedValueOnce({ text: () => 'finance' });

            // 2. BaseAgent Execution
            // Agent returns a tool call. BaseAgent executes it and returns the result immediately (One-Shot).
            (AI.generateContent as any).mockResolvedValueOnce({
                text: () => 'Thinking...',
                functionCalls: () => [{ name: 'analyze_budget', args: { amount: 1000, breakdown: 'Test' } }]
            });

            await service.sendMessage('Check this budget');

            // Verification:
            // 1. Router 
            // 2. Agent (Tool Call) -> BaseAgent executes tool -> Returns result
            // Total 2 calls to generateContent.
            expect(AI.generateContent).toHaveBeenCalledTimes(2);

            // Verify final response contains tool output
            const lastMsg = mockStoreState.agentHistory[mockStoreState.agentHistory.length - 1];
            expect(lastMsg.text).toContain('Tool: analyze_budget');
            expect(lastMsg.text).toContain('approved');
        });
    });

    describe('State & Concurrency (Stress Test)', () => {
        it('should prevent concurrent agent executions (Lock Mechanism)', async () => {
            // Mock a slow operation
            (AI.generateContent as any).mockImplementation(async () => {
                await new Promise(resolve => setTimeout(resolve, 50));
                return { text: () => 'slow response' };
            });

            // Fire two requests almost simultaneously
            const p1 = service.sendMessage('Request 1');
            const p2 = service.sendMessage('Request 2');

            await Promise.all([p1, p2]);

            // Only one should have triggered the AI
            // Logic: isProcessing check prevents 2nd call from proceeding past line 22
            expect(AI.generateContent).toHaveBeenCalledTimes(1);
        });

        it('should accumulate context correctly across multiple turns', async () => {
            // Seed store with history
            const { addAgentMessage } = useStore.getState();
            addAgentMessage({ id: '1', role: 'user', text: 'Prev User', timestamp: 100 });
            addAgentMessage({ id: '2', role: 'model', text: 'Prev Model', timestamp: 200 });

            (AI.generateContent as any).mockResolvedValueOnce({ text: () => 'marketing' }); // Router
            (AI.generateContent as any).mockResolvedValueOnce({
                text: () => 'Response',
                functionCalls: () => []
            }); // Agent

            await service.sendMessage('New Message');

            // Verify Context passed to Agent includes history
            const agentCall = (AI.generateContent as any).mock.calls[1][0];
            const historyString = agentCall.contents[0].parts[0].text;

            // We need to verify that ContextPipeline built the string correctly
            // The mock store has the messages, ContextPipeline reads them.
            expect(historyString).toContain('User: Prev User');
            expect(historyString).toContain('Assistant: Prev Model');
        });
    });

    describe('Edge Cases & Recovery', () => {
        it('should handle hallucinated tool calls gracefully', async () => {
            (AI.generateContent as any).mockResolvedValueOnce({ text: () => 'finance' });

            // Agent tries to call a non-existent tool
            (AI.generateContent as any).mockResolvedValueOnce({
                text: () => 'Thinking...',
                functionCalls: () => [{ name: 'make_money_fast', args: {} }]
            });

            await service.sendMessage('Get rich');

            const lastMsg = mockStoreState.agentHistory[mockStoreState.agentHistory.length - 1];
            expect(lastMsg.text).toContain("Error: Tool 'make_money_fast' not implemented");
        });
    });

    describe('Robustness & Error Handling', () => {
        it('should route to Generalist if Orchestrator hallucinations an invalid ID', async () => {
            // First call: orchestrator returns an invalid agent id
            (AI.generateContent as any).mockResolvedValueOnce({
                text: () => 'super-mega-agent-9000',
                functionCalls: () => []
            });

            // Second call: generalist agent handles the task gracefully
            (AI.generateContentStream as any).mockResolvedValueOnce(buildStreamResponse('Generalist fallback response'));

            const generalistSpy = vi.spyOn(agentRegistry, 'get');

            await service.sendMessage('Do something crazy');

            expect(generalistSpy).toHaveBeenCalledWith('generalist');

            const lastMsg = mockStoreState.agentHistory[mockStoreState.agentHistory.length - 1];
            expect(lastMsg.text).toBe('Generalist fallback response');
        });

        it('should gracefully handle agent execution failure', async () => {
            (AI.generateContent as any).mockResolvedValueOnce({ text: () => 'brand' });

            // Make brand agent crash inside BaseAgent logic
            (AI.generateContent as any).mockRejectedValueOnce(new Error('Simulated API Outage'));

            const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => { });

            await service.sendMessage('Verify this');

            // AgentService should receive the error text from BaseAgent's catch block
            const lastMsg = mockStoreState.agentHistory[mockStoreState.agentHistory.length - 1];
            expect(lastMsg.role).toBe('model'); // Not system, because BaseAgent handled it
            expect(lastMsg.text).toContain('Error executing task');
            expect(lastMsg.text).toContain('Simulated API Outage');

            consoleSpy.mockRestore();
        });
    });
});
