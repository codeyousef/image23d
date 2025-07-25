/**
 * Tests for API service module
 */

import { ApiService } from '../../../../frontend/src/services/api';
import { mockServer } from '../../../mocks/server';
import { rest } from 'msw';

describe('ApiService', () => {
  let apiService: ApiService;

  beforeEach(() => {
    apiService = new ApiService({
      baseURL: 'http://localhost:8000',
      timeout: 5000,
    });
  });

  afterEach(() => {
    apiService.clearCache();
  });

  describe('Basic HTTP Methods', () => {
    test('should make GET request', async () => {
      const response = await apiService.get('/api/v1/models');
      
      expect(response.data).toEqual({
        models: [
          { id: 'flux-1-dev', name: 'FLUX.1-dev', type: 'image' },
          { id: 'sdxl-turbo', name: 'SDXL-Turbo', type: 'image' },
        ],
      });
    });

    test('should make POST request', async () => {
      const requestData = {
        type: 'text_to_image',
        params: {
          prompt: 'A beautiful sunset',
          model: 'flux-1-dev',
        },
      };

      const response = await apiService.post('/api/v1/jobs/create', requestData);
      
      expect(response.data).toEqual({
        job_id: 'test-123',
        status: 'queued',
      });
    });

    test('should make PUT request', async () => {
      const updateData = { status: 'cancelled' };
      
      mockServer.use(
        rest.put('/api/v1/jobs/test-123', (req, res, ctx) => {
          return res(ctx.json({ 
            job_id: 'test-123', 
            status: 'cancelled',
            updated_at: '2024-01-01T12:00:00Z'
          }));
        })
      );

      const response = await apiService.put('/api/v1/jobs/test-123', updateData);
      
      expect(response.data.status).toBe('cancelled');
    });

    test('should make DELETE request', async () => {
      mockServer.use(
        rest.delete('/api/v1/jobs/test-123', (req, res, ctx) => {
          return res(ctx.json({ message: 'Job deleted successfully' }));
        })
      );

      const response = await apiService.delete('/api/v1/jobs/test-123');
      
      expect(response.data.message).toBe('Job deleted successfully');
    });
  });

  describe('Authentication', () => {
    test('should include auth token in requests', async () => {
      apiService.setAuthToken('bearer-token-123');
      
      let capturedHeaders: any;
      mockServer.use(
        rest.get('/api/v1/user/profile', (req, res, ctx) => {
          capturedHeaders = req.headers;
          return res(ctx.json({ user_id: 'test-user' }));
        })
      );

      await apiService.get('/api/v1/user/profile');
      
      expect(capturedHeaders.get('authorization')).toBe('Bearer bearer-token-123');
    });

    test('should handle token refresh', async () => {
      let refreshCalled = false;
      
      apiService.setTokenRefreshHandler(async () => {
        refreshCalled = true;
        return 'new-token-456';
      });

      mockServer.use(
        rest.get('/api/v1/protected', (req, res, ctx) => {
          const auth = req.headers.get('authorization');
          if (auth === 'Bearer expired-token') {
            return res(ctx.status(401), ctx.json({ detail: 'Token expired' }));
          }
          if (auth === 'Bearer new-token-456') {
            return res(ctx.json({ data: 'protected data' }));
          }
          return res(ctx.status(401));
        })
      );

      apiService.setAuthToken('expired-token');
      
      const response = await apiService.get('/api/v1/protected');
      
      expect(refreshCalled).toBe(true);
      expect(response.data.data).toBe('protected data');
    });
  });

  describe('Error Handling', () => {
    test('should handle HTTP errors', async () => {
      mockServer.use(
        rest.get('/api/v1/error', (req, res, ctx) => {
          return res(ctx.status(400), ctx.json({ detail: 'Bad request' }));
        })
      );

      await expect(apiService.get('/api/v1/error')).rejects.toThrow('Bad request');
    });

    test('should handle network errors', async () => {
      mockServer.use(
        rest.get('/api/v1/network-error', (req, res, ctx) => {
          return res.networkError('Network error');
        })
      );

      await expect(apiService.get('/api/v1/network-error')).rejects.toThrow();
    });

    test('should handle timeout', async () => {
      mockServer.use(
        rest.get('/api/v1/slow', (req, res, ctx) => {
          return res(ctx.delay(6000), ctx.json({ data: 'slow response' }));
        })
      );

      await expect(apiService.get('/api/v1/slow')).rejects.toThrow();
    });

    test('should retry failed requests', async () => {
      let attempts = 0;
      
      mockServer.use(
        rest.get('/api/v1/retry', (req, res, ctx) => {
          attempts++;
          if (attempts < 3) {
            return res(ctx.status(500));
          }
          return res(ctx.json({ success: true, attempts }));
        })
      );

      const response = await apiService.get('/api/v1/retry', {
        retry: 3,
        retryDelay: 100,
      });
      
      expect(response.data.success).toBe(true);
      expect(response.data.attempts).toBe(3);
    });
  });

  describe('Request/Response Interceptors', () => {
    test('should apply request interceptors', async () => {
      let interceptorCalled = false;
      
      apiService.addRequestInterceptor((config) => {
        interceptorCalled = true;
        config.headers['X-Custom-Header'] = 'test-value';
        return config;
      });

      let capturedHeaders: any;
      mockServer.use(
        rest.get('/api/v1/intercepted', (req, res, ctx) => {
          capturedHeaders = req.headers;
          return res(ctx.json({ success: true }));
        })
      );

      await apiService.get('/api/v1/intercepted');
      
      expect(interceptorCalled).toBe(true);
      expect(capturedHeaders.get('x-custom-header')).toBe('test-value');
    });

    test('should apply response interceptors', async () => {
      let interceptorCalled = false;
      
      apiService.addResponseInterceptor((response) => {
        interceptorCalled = true;
        response.data.intercepted = true;
        return response;
      });

      const response = await apiService.get('/api/v1/models');
      
      expect(interceptorCalled).toBe(true);
      expect(response.data.intercepted).toBe(true);
    });
  });

  describe('File Upload', () => {
    test('should upload single file', async () => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      
      const response = await apiService.uploadFile('/api/v1/upload/file', file);
      
      expect(response.data).toEqual({
        file_id: 'uploaded-123',
        url: 'https://api.example.com/files/uploaded-123',
      });
    });

    test('should upload multiple files', async () => {
      const files = [
        new File(['content 1'], 'file1.txt', { type: 'text/plain' }),
        new File(['content 2'], 'file2.txt', { type: 'text/plain' }),
      ];
      
      mockServer.use(
        rest.post('/api/v1/upload/multiple', (req, res, ctx) => {
          return res(ctx.json({
            files: [
              { file_id: 'file1-123', url: 'https://api.example.com/files/file1-123' },
              { file_id: 'file2-456', url: 'https://api.example.com/files/file2-456' },
            ],
          }));
        })
      );

      const response = await apiService.uploadFiles('/api/v1/upload/multiple', files);
      
      expect(response.data.files).toHaveLength(2);
    });

    test('should track upload progress', async () => {
      const file = new File(['large content'], 'large.txt', { type: 'text/plain' });
      const progressCallback = jest.fn();
      
      await apiService.uploadFile('/api/v1/upload/file', file, {
        onUploadProgress: progressCallback,
      });
      
      expect(progressCallback).toHaveBeenCalled();
    });
  });

  describe('Caching', () => {
    test('should cache GET requests', async () => {
      let requestCount = 0;
      
      mockServer.use(
        rest.get('/api/v1/cacheable', (req, res, ctx) => {
          requestCount++;
          return res(ctx.json({ data: 'cached data', count: requestCount }));
        })
      );

      // First request
      const response1 = await apiService.get('/api/v1/cacheable', { cache: true });
      
      // Second request should use cache
      const response2 = await apiService.get('/api/v1/cacheable', { cache: true });
      
      expect(requestCount).toBe(1);
      expect(response1.data).toEqual(response2.data);
    });

    test('should respect cache TTL', async () => {
      let requestCount = 0;
      
      mockServer.use(
        rest.get('/api/v1/ttl-cache', (req, res, ctx) => {
          requestCount++;
          return res(ctx.json({ count: requestCount }));
        })
      );

      // First request
      await apiService.get('/api/v1/ttl-cache', { 
        cache: true, 
        cacheTTL: 100, // 100ms TTL
      });
      
      // Second request within TTL
      await apiService.get('/api/v1/ttl-cache', { cache: true });
      
      expect(requestCount).toBe(1);
      
      // Wait for TTL to expire
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // Third request should make new request
      await apiService.get('/api/v1/ttl-cache', { cache: true });
      
      expect(requestCount).toBe(2);
    });

    test('should invalidate cache', async () => {
      let requestCount = 0;
      
      mockServer.use(
        rest.get('/api/v1/invalidate', (req, res, ctx) => {
          requestCount++;
          return res(ctx.json({ count: requestCount }));
        })
      );

      // First request
      await apiService.get('/api/v1/invalidate', { cache: true });
      
      // Invalidate cache
      apiService.invalidateCache('/api/v1/invalidate');
      
      // Second request should make new request
      await apiService.get('/api/v1/invalidate', { cache: true });
      
      expect(requestCount).toBe(2);
    });
  });

  describe('Batch Requests', () => {
    test('should batch multiple requests', async () => {
      const requests = [
        { method: 'GET', url: '/api/v1/models' },
        { method: 'GET', url: '/api/v1/user/profile' },
        { method: 'POST', url: '/api/v1/jobs/create', data: { prompt: 'test' } },
      ];
      
      mockServer.use(
        rest.get('/api/v1/user/profile', (req, res, ctx) => {
          return res(ctx.json({ user_id: 'test-user' }));
        })
      );

      const responses = await apiService.batch(requests);
      
      expect(responses).toHaveLength(3);
      expect(responses[0].data.models).toBeTruthy();
      expect(responses[1].data.user_id).toBe('test-user');
      expect(responses[2].data.job_id).toBeTruthy();
    });

    test('should handle partial batch failures', async () => {
      const requests = [
        { method: 'GET', url: '/api/v1/models' },
        { method: 'GET', url: '/api/v1/nonexistent' },
      ];
      
      mockServer.use(
        rest.get('/api/v1/nonexistent', (req, res, ctx) => {
          return res(ctx.status(404), ctx.json({ error: 'Not found' }));
        })
      );

      const responses = await apiService.batch(requests, { 
        continueOnError: true 
      });
      
      expect(responses).toHaveLength(2);
      expect(responses[0].success).toBe(true);
      expect(responses[1].success).toBe(false);
      expect(responses[1].error).toBeTruthy();
    });
  });

  describe('WebSocket Integration', () => {
    test('should provide WebSocket URL', () => {
      const wsUrl = apiService.getWebSocketUrl('/ws/jobs');
      
      expect(wsUrl).toBe('ws://localhost:8000/ws/jobs');
    });

    test('should include auth in WebSocket URL', () => {
      apiService.setAuthToken('ws-token-123');
      
      const wsUrl = apiService.getWebSocketUrl('/ws/jobs');
      
      expect(wsUrl).toContain('token=ws-token-123');
    });
  });

  describe('Request Cancellation', () => {
    test('should cancel ongoing requests', async () => {
      const abortController = new AbortController();
      
      mockServer.use(
        rest.get('/api/v1/slow-request', (req, res, ctx) => {
          return res(ctx.delay(1000), ctx.json({ data: 'slow data' }));
        })
      );

      const requestPromise = apiService.get('/api/v1/slow-request', {
        signal: abortController.signal,
      });
      
      // Cancel after 100ms
      setTimeout(() => abortController.abort(), 100);
      
      await expect(requestPromise).rejects.toThrow();
    });

    test('should cancel all pending requests', async () => {
      const promises = [
        apiService.get('/api/v1/request1'),
        apiService.get('/api/v1/request2'),
        apiService.get('/api/v1/request3'),
      ];
      
      apiService.cancelAllRequests();
      
      // All requests should be cancelled
      const results = await Promise.allSettled(promises);
      results.forEach(result => {
        expect(result.status).toBe('rejected');
      });
    });
  });
});