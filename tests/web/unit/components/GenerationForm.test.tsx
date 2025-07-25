/**
 * Tests for GenerationForm React component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GenerationForm } from '../../../../frontend/src/components/GenerationForm';
import { mockServer } from '../../../mocks/server';
import { rest } from 'msw';

describe('GenerationForm Component', () => {
  let queryClient: QueryClient;
  const mockOnSubmit = jest.fn();
  const mockOnProgress = jest.fn();

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    mockOnSubmit.mockClear();
    mockOnProgress.mockClear();
  });

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <GenerationForm 
          onSubmit={mockOnSubmit}
          onProgress={mockOnProgress}
          {...props}
        />
      </QueryClientProvider>
    );
  };

  test('should render all form fields', () => {
    renderComponent();

    expect(screen.getByLabelText(/prompt/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/model/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/negative prompt/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate/i })).toBeInTheDocument();
  });

  test('should load available models on mount', async () => {
    renderComponent();

    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/model/i);
      expect(modelSelect).toHaveTextContent('FLUX.1-dev');
      expect(modelSelect).toHaveTextContent('SDXL-Turbo');
    });
  });

  test('should validate required fields', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Try to submit without prompt
    const submitButton = screen.getByRole('button', { name: /generate/i });
    await user.click(submitButton);

    expect(await screen.findByText(/prompt is required/i)).toBeInTheDocument();
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('should submit form with valid data', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Fill form
    await user.type(screen.getByLabelText(/prompt/i), 'A beautiful sunset');
    await user.selectOptions(screen.getByLabelText(/model/i), 'flux-1-dev');
    await user.type(screen.getByLabelText(/negative prompt/i), 'ugly, blurry');
    
    // Submit
    await user.click(screen.getByRole('button', { name: /generate/i }));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        prompt: 'A beautiful sunset',
        model: 'flux-1-dev',
        negativePrompt: 'ugly, blurry',
        width: 1024,
        height: 1024,
        numInferenceSteps: 50,
        guidanceScale: 7.5,
        seed: null,
      });
    });
  });

  test('should show advanced options when toggled', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Advanced options should be hidden initially
    expect(screen.queryByLabelText(/seed/i)).not.toBeInTheDocument();

    // Click advanced toggle
    await user.click(screen.getByText(/show advanced options/i));

    // Should show advanced fields
    expect(screen.getByLabelText(/seed/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/steps/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/guidance scale/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/width/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/height/i)).toBeInTheDocument();
  });

  test('should handle preset dimensions', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Show advanced options
    await user.click(screen.getByText(/show advanced options/i));

    // Click preset button
    await user.click(screen.getByRole('button', { name: /1:1/i }));

    // Check dimensions updated
    expect(screen.getByLabelText(/width/i)).toHaveValue(1024);
    expect(screen.getByLabelText(/height/i)).toHaveValue(1024);

    // Try 16:9 preset
    await user.click(screen.getByRole('button', { name: /16:9/i }));
    expect(screen.getByLabelText(/width/i)).toHaveValue(1344);
    expect(screen.getByLabelText(/height/i)).toHaveValue(768);
  });

  test('should handle file upload for image-to-image', async () => {
    renderComponent({ mode: 'image-to-image' });
    const user = userEvent.setup();

    const file = new File(['test'], 'test.png', { type: 'image/png' });
    const input = screen.getByLabelText(/upload image/i);

    await user.upload(input, file);

    expect(input.files?.[0]).toBe(file);
    expect(input.files).toHaveLength(1);

    // Should show preview
    await waitFor(() => {
      expect(screen.getByAltText(/preview/i)).toBeInTheDocument();
    });
  });

  test('should validate file type', async () => {
    renderComponent({ mode: 'image-to-image' });
    const user = userEvent.setup();

    const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByLabelText(/upload image/i);

    await user.upload(input, invalidFile);

    expect(await screen.findByText(/invalid file type/i)).toBeInTheDocument();
  });

  test('should disable form during submission', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Fill and submit
    await user.type(screen.getByLabelText(/prompt/i), 'Test prompt');
    
    // Mock slow API response
    mockServer.use(
      rest.post('/api/v1/jobs/create', async (req, res, ctx) => {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return res(ctx.json({ job_id: 'test-123', status: 'queued' }));
      })
    );

    await user.click(screen.getByRole('button', { name: /generate/i }));

    // Form should be disabled
    expect(screen.getByLabelText(/prompt/i)).toBeDisabled();
    expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled();
  });

  test('should show error on API failure', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Mock error response
    mockServer.use(
      rest.post('/api/v1/jobs/create', (req, res, ctx) => {
        return res(ctx.status(400), ctx.json({ detail: 'Invalid prompt' }));
      })
    );

    await user.type(screen.getByLabelText(/prompt/i), 'Test');
    await user.click(screen.getByRole('button', { name: /generate/i }));

    expect(await screen.findByText(/invalid prompt/i)).toBeInTheDocument();
  });

  test('should handle model-specific options', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Select video model
    await user.selectOptions(screen.getByLabelText(/model/i), 'ltx-video');

    // Should show video-specific options
    await waitFor(() => {
      expect(screen.getByLabelText(/duration/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/fps/i)).toBeInTheDocument();
    });

    // Select 3D model
    await user.selectOptions(screen.getByLabelText(/model/i), 'hunyuan3d-mini');

    // Should show 3D-specific options
    await waitFor(() => {
      expect(screen.getByLabelText(/quality preset/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/export formats/i)).toBeInTheDocument();
    });
  });

  test('should calculate and display credit cost', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Select model and options
    await user.selectOptions(screen.getByLabelText(/model/i), 'flux-1-dev');
    await user.click(screen.getByText(/show advanced options/i));
    await user.clear(screen.getByLabelText(/steps/i));
    await user.type(screen.getByLabelText(/steps/i), '100');

    // Should show updated cost
    await waitFor(() => {
      const costText = screen.getByText(/credits/i);
      expect(costText).toHaveTextContent('10'); // Example cost
    });
  });

  test('should save and load form presets', async () => {
    renderComponent();
    const user = userEvent.setup();

    // Fill form
    await user.type(screen.getByLabelText(/prompt/i), 'Preset test');
    await user.click(screen.getByText(/show advanced options/i));
    await user.clear(screen.getByLabelText(/steps/i));
    await user.type(screen.getByLabelText(/steps/i), '30');

    // Save preset
    await user.click(screen.getByRole('button', { name: /save preset/i }));
    await user.type(screen.getByLabelText(/preset name/i), 'My Preset');
    await user.click(screen.getByRole('button', { name: /save/i }));

    // Clear form
    await user.clear(screen.getByLabelText(/prompt/i));

    // Load preset
    await user.selectOptions(screen.getByLabelText(/load preset/i), 'My Preset');

    // Check values restored
    expect(screen.getByLabelText(/prompt/i)).toHaveValue('Preset test');
    expect(screen.getByLabelText(/steps/i)).toHaveValue(30);
  });
});