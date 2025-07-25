/**
 * Tests for useWebSocket React hook
 */

import { renderHook, act } from '@testing-library/react-hooks';
import { useWebSocket } from '../../../../frontend/src/hooks/useWebSocket';
import WS from 'jest-websocket-mock';

describe('useWebSocket Hook', () => {
  let server: WS;
  const WS_URL = 'ws://localhost:8080/ws';

  beforeEach(async () => {
    server = new WS(WS_URL);
  });

  afterEach(() => {
    WS.clean();
  });

  test('should connect to WebSocket server', async () => {
    const { result } = renderHook(() => useWebSocket(WS_URL));

    await server.connected;

    expect(result.current.readyState).toBe(WebSocket.OPEN);
    expect(result.current.lastMessage).toBeNull();
  });

  test('should receive messages', async () => {
    const onMessage = jest.fn();
    
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      onMessage,
    }));

    await server.connected;

    const testMessage = { type: 'job_update', job_id: '123', status: 'processing' };
    
    act(() => {
      server.send(JSON.stringify(testMessage));
    });

    expect(onMessage).toHaveBeenCalledWith(expect.objectContaining({
      data: JSON.stringify(testMessage),
    }));
    expect(result.current.lastMessage?.data).toBe(JSON.stringify(testMessage));
  });

  test('should send messages', async () => {
    const { result } = renderHook(() => useWebSocket(WS_URL));

    await server.connected;

    const message = { type: 'subscribe', channels: ['job_updates'] };

    act(() => {
      result.current.sendMessage(JSON.stringify(message));
    });

    await expect(server).toReceiveMessage(JSON.stringify(message));
  });

  test('should handle connection errors', async () => {
    const onError = jest.fn();
    
    server.close();

    const { result } = renderHook(() => useWebSocket('ws://invalid-url', {
      onError,
    }));

    // Wait for error
    await new Promise(resolve => setTimeout(resolve, 100));

    expect(onError).toHaveBeenCalled();
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
  });

  test('should reconnect automatically', async () => {
    const onReconnect = jest.fn();
    
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      shouldReconnect: () => true,
      reconnectAttempts: 3,
      reconnectInterval: 100,
      onReconnect,
    }));

    await server.connected;

    // Simulate connection loss
    act(() => {
      server.close();
    });

    expect(result.current.readyState).toBe(WebSocket.CLOSED);

    // Create new server for reconnection
    server = new WS(WS_URL);
    
    // Wait for reconnection
    await server.connected;

    expect(onReconnect).toHaveBeenCalled();
    expect(result.current.readyState).toBe(WebSocket.OPEN);
  });

  test('should handle job status updates', async () => {
    const onJobUpdate = jest.fn();
    
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      onMessage: (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'job_update') {
          onJobUpdate(data);
        }
      },
    }));

    await server.connected;

    const updates = [
      { type: 'job_update', job_id: '123', status: 'queued', progress: 0 },
      { type: 'job_update', job_id: '123', status: 'processing', progress: 0.3 },
      { type: 'job_update', job_id: '123', status: 'processing', progress: 0.7 },
      { type: 'job_update', job_id: '123', status: 'completed', progress: 1.0 },
    ];

    for (const update of updates) {
      act(() => {
        server.send(JSON.stringify(update));
      });
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    expect(onJobUpdate).toHaveBeenCalledTimes(4);
    expect(onJobUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({
        status: 'completed',
        progress: 1.0,
      })
    );
  });

  test('should filter messages by type', async () => {
    const onJobUpdate = jest.fn();
    const onSystemMessage = jest.fn();
    
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      filter: (message) => {
        const data = JSON.parse(message.data);
        return data.type === 'job_update' || data.type === 'system';
      },
      onMessage: (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'job_update') {
          onJobUpdate(data);
        } else if (data.type === 'system') {
          onSystemMessage(data);
        }
      },
    }));

    await server.connected;

    // Send filtered message
    act(() => {
      server.send(JSON.stringify({ type: 'job_update', status: 'processing' }));
    });

    // Send ignored message
    act(() => {
      server.send(JSON.stringify({ type: 'heartbeat', timestamp: Date.now() }));
    });

    // Send another filtered message
    act(() => {
      server.send(JSON.stringify({ type: 'system', message: 'Maintenance mode' }));
    });

    expect(onJobUpdate).toHaveBeenCalledTimes(1);
    expect(onSystemMessage).toHaveBeenCalledTimes(1);
  });

  test('should handle binary messages', async () => {
    const onMessage = jest.fn();
    
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      onMessage,
    }));

    await server.connected;

    const binaryData = new ArrayBuffer(8);
    const view = new Uint8Array(binaryData);
    view.set([1, 2, 3, 4, 5, 6, 7, 8]);

    act(() => {
      server.server.clients().forEach(client => {
        client.send(binaryData);
      });
    });

    expect(onMessage).toHaveBeenCalledWith(expect.objectContaining({
      data: expect.any(ArrayBuffer),
    }));
  });

  test('should manage connection state properly', async () => {
    const onOpen = jest.fn();
    const onClose = jest.fn();
    
    const { result, unmount } = renderHook(() => useWebSocket(WS_URL, {
      onOpen,
      onClose,
    }));

    expect(result.current.readyState).toBe(WebSocket.CONNECTING);

    await server.connected;

    expect(onOpen).toHaveBeenCalled();
    expect(result.current.readyState).toBe(WebSocket.OPEN);

    unmount();

    expect(onClose).toHaveBeenCalled();
  });

  test('should support conditional connection', async () => {
    const { result, rerender } = renderHook(
      ({ shouldConnect }) => useWebSocket(shouldConnect ? WS_URL : null),
      { initialProps: { shouldConnect: false } }
    );

    expect(result.current.readyState).toBe(WebSocket.CLOSED);

    rerender({ shouldConnect: true });

    await server.connected;

    expect(result.current.readyState).toBe(WebSocket.OPEN);

    rerender({ shouldConnect: false });

    expect(result.current.readyState).toBe(WebSocket.CLOSED);
  });

  test('should handle authentication', async () => {
    const { result } = renderHook(() => useWebSocket(`${WS_URL}?token=auth-token-123`));

    await server.connected;

    // Verify auth token was sent in connection
    const client = server.server.clients()[0];
    expect(client.url).toContain('token=auth-token-123');
  });

  test('should throttle message sending', async () => {
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      sendJsonMessage: {
        throttle: 100, // 100ms throttle
      },
    }));

    await server.connected;

    const messages = [
      { id: 1, data: 'message 1' },
      { id: 2, data: 'message 2' },
      { id: 3, data: 'message 3' },
    ];

    // Send messages rapidly
    act(() => {
      messages.forEach(msg => {
        result.current.sendJsonMessage(msg);
      });
    });

    // Only last message should be sent due to throttling
    await expect(server).toReceiveMessage(JSON.stringify(messages[2]));
    
    // Should not receive other messages immediately
    expect(server).not.toHaveReceived(JSON.stringify(messages[0]));
    expect(server).not.toHaveReceived(JSON.stringify(messages[1]));
  });

  test('should maintain message history', async () => {
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      messageHistory: 5, // Keep last 5 messages
    }));

    await server.connected;

    const messages = [
      { id: 1, text: 'First' },
      { id: 2, text: 'Second' },
      { id: 3, text: 'Third' },
    ];

    for (const message of messages) {
      act(() => {
        server.send(JSON.stringify(message));
      });
    }

    expect(result.current.lastJsonMessage).toEqual(messages[2]);
    expect(result.current.messageHistory).toHaveLength(3);
    expect(result.current.messageHistory[0].data).toBe(JSON.stringify(messages[0]));
  });

  test('should handle connection timeout', async () => {
    const onError = jest.fn();
    
    // Don't start server to simulate timeout
    const { result } = renderHook(() => useWebSocket(WS_URL, {
      onError,
      connectionTimeout: 100, // 100ms timeout
    }));

    await new Promise(resolve => setTimeout(resolve, 200));

    expect(onError).toHaveBeenCalled();
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
  });
});