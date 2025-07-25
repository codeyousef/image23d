/**
 * Tests for useAPI React hook
 */

import { renderHook, act } from '@testing-library/react-hooks';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAPI } from '../../../../frontend/src/hooks/useAPI';
import { mockServer } from '../../../mocks/server';
import { rest } from 'msw';

describe('useAPI Hook', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  test('should fetch data successfully', async () => {
    const { result, waitFor } = renderHook(() => useAPI('/api/v1/models'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual({
      models: [
        { id: 'flux-1-dev', name: 'FLUX.1-dev', type: 'image' },
        { id: 'sdxl-turbo', name: 'SDXL-Turbo', type: 'image' },
      ],
    });
  });

  test('should handle API errors', async () => {
    mockServer.use(
      rest.get('/api/v1/models', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ detail: 'Server error' }));
      })
    );

    const { result, waitFor } = renderHook(() => useAPI('/api/v1/models'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBeTruthy();
    expect(result.current.error.message).toContain('Server error');
  });

  test('should support custom query options', async () => {
    const onSuccess = jest.fn();
    const onError = jest.fn();

    const { result, waitFor } = renderHook(
      () => useAPI('/api/v1/models', {
        onSuccess,
        onError,
        enabled: true,
      }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalledWith(expect.objectContaining({
      models: expect.any(Array),
    }));
    expect(onError).not.toHaveBeenCalled();
  });

  test('should refetch data on demand', async () => {
    const { result, waitFor } = renderHook(() => useAPI('/api/v1/models'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const initialData = result.current.data;

    // Mock different response for refetch
    mockServer.use(
      rest.get('/api/v1/models', (req, res, ctx) => {
        return res(ctx.json({
          models: [
            { id: 'new-model', name: 'New Model', type: 'image' },
          ],
        }));
      })
    );

    act(() => {
      result.current.refetch();
    });

    await waitFor(() => expect(result.current.isFetching).toBe(false));

    expect(result.current.data).not.toEqual(initialData);
    expect(result.current.data.models).toHaveLength(1);
    expect(result.current.data.models[0].id).toBe('new-model');
  });

  test('should handle authentication headers', async () => {
    let capturedHeaders: any;

    mockServer.use(
      rest.get('/api/v1/user/profile', (req, res, ctx) => {
        capturedHeaders = req.headers;
        return res(ctx.json({ user_id: 'test-123' }));
      })
    );

    const { waitFor } = renderHook(
      () => useAPI('/api/v1/user/profile', {
        headers: {
          'Authorization': 'Bearer test-token',
        },
      }),
      { wrapper }
    );

    await waitFor(() => expect(capturedHeaders).toBeTruthy());

    expect(capturedHeaders.get('authorization')).toBe('Bearer test-token');
  });

  test('should handle query parameters', async () => {
    let capturedUrl: string;

    mockServer.use(
      rest.get('/api/v1/jobs', (req, res, ctx) => {
        capturedUrl = req.url.toString();
        return res(ctx.json({ jobs: [] }));
      })
    );

    const { waitFor } = renderHook(
      () => useAPI('/api/v1/jobs', {
        params: {
          status: 'completed',
          limit: 10,
        },
      }),
      { wrapper }
    );

    await waitFor(() => expect(capturedUrl).toBeTruthy());

    expect(capturedUrl).toContain('status=completed');
    expect(capturedUrl).toContain('limit=10');
  });

  test('should support mutations', async () => {
    const { result, waitFor } = renderHook(
      () => useAPI.mutation('/api/v1/jobs/create'),
      { wrapper }
    );

    act(() => {
      result.current.mutate({
        type: 'text_to_image',
        params: {
          prompt: 'Test prompt',
          model: 'flux-1-dev',
        },
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual({
      job_id: 'test-123',
      status: 'queued',
    });
  });

  test('should handle optimistic updates', async () => {
    const { result: queryResult } = renderHook(
      () => useAPI('/api/v1/jobs'),
      { wrapper }
    );

    await act(async () => {
      await queryClient.prefetchQuery(['api', '/api/v1/jobs'], async () => ({
        jobs: [{ id: '1', status: 'completed' }],
      }));
    });

    const { result: mutationResult } = renderHook(
      () => useAPI.mutation('/api/v1/jobs/create', {
        onMutate: async (newJob) => {
          await queryClient.cancelQueries(['api', '/api/v1/jobs']);
          
          const previousJobs = queryClient.getQueryData(['api', '/api/v1/jobs']);
          
          queryClient.setQueryData(['api', '/api/v1/jobs'], (old: any) => ({
            jobs: [...(old?.jobs || []), { id: 'optimistic', ...newJob }],
          }));
          
          return { previousJobs };
        },
        onError: (err, newJob, context) => {
          queryClient.setQueryData(['api', '/api/v1/jobs'], context?.previousJobs);
        },
        onSettled: () => {
          queryClient.invalidateQueries(['api', '/api/v1/jobs']);
        },
      }),
      { wrapper }
    );

    act(() => {
      mutationResult.current.mutate({
        type: 'text_to_image',
        params: { prompt: 'New job' },
      });
    });

    // Check optimistic update
    const jobs = queryClient.getQueryData(['api', '/api/v1/jobs']) as any;
    expect(jobs.jobs).toHaveLength(2);
    expect(jobs.jobs[1].id).toBe('optimistic');
  });

  test('should handle request cancellation', async () => {
    const abortController = new AbortController();

    const { result, waitFor } = renderHook(
      () => useAPI('/api/v1/models', {
        signal: abortController.signal,
      }),
      { wrapper }
    );

    // Cancel immediately
    act(() => {
      abortController.abort();
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error.message).toContain('aborted');
  });

  test('should support pagination', async () => {
    const { result, waitFor, rerender } = renderHook(
      ({ page }) => useAPI('/api/v1/jobs', {
        params: { page, per_page: 10 },
      }),
      {
        wrapper,
        initialProps: { page: 1 },
      }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toBeTruthy();

    // Change page
    rerender({ page: 2 });

    await waitFor(() => expect(result.current.isFetching).toBe(true));
    await waitFor(() => expect(result.current.isFetching).toBe(false));

    // Should have fetched new page
    expect(result.current.isSuccess).toBe(true);
  });

  test('should handle file uploads', async () => {
    const file = new File(['test'], 'test.png', { type: 'image/png' });

    const { result, waitFor } = renderHook(
      () => useAPI.upload('/api/v1/upload/image'),
      { wrapper }
    );

    act(() => {
      result.current.mutate({
        file,
        onUploadProgress: (progress) => {
          console.log(`Upload progress: ${progress}%`);
        },
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual({
      file_id: 'uploaded-123',
      url: 'https://api.example.com/files/uploaded-123',
    });
  });

  test('should retry failed requests', async () => {
    let attempts = 0;

    mockServer.use(
      rest.get('/api/v1/flaky', (req, res, ctx) => {
        attempts++;
        if (attempts < 3) {
          return res(ctx.status(500));
        }
        return res(ctx.json({ success: true }));
      })
    );

    const { result, waitFor } = renderHook(
      () => useAPI('/api/v1/flaky', {
        retry: 3,
        retryDelay: 10,
      }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true), {
      timeout: 5000,
    });

    expect(attempts).toBe(3);
    expect(result.current.data).toEqual({ success: true });
  });

  test('should cache responses', async () => {
    let fetchCount = 0;

    mockServer.use(
      rest.get('/api/v1/cached', (req, res, ctx) => {
        fetchCount++;
        return res(ctx.json({ count: fetchCount }));
      })
    );

    // First hook
    const { result: result1, waitFor } = renderHook(
      () => useAPI('/api/v1/cached'),
      { wrapper }
    );

    await waitFor(() => expect(result1.current.isSuccess).toBe(true));

    // Second hook with same key
    const { result: result2 } = renderHook(
      () => useAPI('/api/v1/cached'),
      { wrapper }
    );

    // Should use cached data
    expect(result2.current.data).toEqual(result1.current.data);
    expect(fetchCount).toBe(1);
  });
});