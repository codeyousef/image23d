/**
 * Cypress E2E tests for generation workflows
 */

describe('Generation Workflow E2E Tests', () => {
  beforeEach(() => {
    // Set up API mocks
    cy.intercept('GET', '/api/v1/models', {
      fixture: 'models.json',
    }).as('getModels');

    cy.intercept('POST', '/api/v1/jobs/create', {
      fixture: 'job-created.json',
    }).as('createJob');

    cy.intercept('GET', '/api/v1/jobs/*/status', {
      fixture: 'job-status.json',
    }).as('getJobStatus');

    cy.intercept('GET', '/api/v1/user/profile', {
      fixture: 'user-profile.json',
    }).as('getUserProfile');

    // Visit the application
    cy.visit('/');
    
    // Wait for initial load
    cy.wait('@getModels');
    cy.wait('@getUserProfile');
  });

  describe('Text-to-Image Generation', () => {
    it('should complete full text-to-image workflow', () => {
      // Navigate to generation page
      cy.get('[data-cy=nav-generate]').click();
      
      // Fill in the form
      cy.get('[data-cy=prompt-input]').type('A beautiful sunset over mountains');
      cy.get('[data-cy=model-select]').select('FLUX.1-dev');
      cy.get('[data-cy=negative-prompt]').type('ugly, blurry, low quality');
      
      // Open advanced options
      cy.get('[data-cy=advanced-toggle]').click();
      cy.get('[data-cy=width-input]').clear().type('1024');
      cy.get('[data-cy=height-input]').clear().type('1024');
      cy.get('[data-cy=steps-input]').clear().type('50');
      
      // Start generation
      cy.get('[data-cy=generate-button]').click();
      
      // Verify job creation
      cy.wait('@createJob').then((interception) => {
        expect(interception.request.body).to.deep.include({
          type: 'text_to_image',
          params: {
            prompt: 'A beautiful sunset over mountains',
            negative_prompt: 'ugly, blurry, low quality',
            model: 'flux-1-dev',
            width: 1024,
            height: 1024,
            num_inference_steps: 50,
          },
        });
      });
      
      // Check progress display
      cy.get('[data-cy=progress-bar]').should('be.visible');
      cy.get('[data-cy=progress-message]').should('contain', 'Generating');
      
      // Mock job completion
      cy.intercept('GET', '/api/v1/jobs/*/status', {
        body: {
          job_id: 'test-123',
          status: 'completed',
          progress: 1.0,
          result: {
            output_url: 'https://example.com/result.png',
            metadata: {
              seed: 42,
              model: 'flux-1-dev',
              generation_time: 15.3,
            },
          },
        },
      }).as('getCompletedStatus');
      
      cy.wait('@getCompletedStatus');
      
      // Check result display
      cy.get('[data-cy=result-image]').should('be.visible');
      cy.get('[data-cy=download-button]').should('be.visible');
      cy.get('[data-cy=result-metadata]').should('contain', 'Seed: 42');
    });

    it('should handle generation errors gracefully', () => {
      cy.get('[data-cy=nav-generate]').click();
      
      // Mock API error
      cy.intercept('POST', '/api/v1/jobs/create', {
        statusCode: 400,
        body: {
          detail: 'Insufficient credits to complete this generation',
        },
      }).as('createJobError');
      
      cy.get('[data-cy=prompt-input]').type('Test prompt');
      cy.get('[data-cy=generate-button]').click();
      
      cy.wait('@createJobError');
      
      // Check error display
      cy.get('[data-cy=error-message]').should('be.visible');
      cy.get('[data-cy=error-message]').should('contain', 'Insufficient credits');
      
      // Check retry option
      cy.get('[data-cy=retry-button]').should('be.visible');
    });

    it('should validate form inputs', () => {
      cy.get('[data-cy=nav-generate]').click();
      
      // Try to generate without prompt
      cy.get('[data-cy=generate-button]').click();
      
      cy.get('[data-cy=prompt-error]').should('be.visible');
      cy.get('[data-cy=prompt-error]').should('contain', 'Prompt is required');
      
      // Test invalid dimensions
      cy.get('[data-cy=advanced-toggle]').click();
      cy.get('[data-cy=width-input]').clear().type('2048');
      cy.get('[data-cy=height-input]').clear().type('2048');
      
      cy.get('[data-cy=prompt-input]').type('Valid prompt');
      cy.get('[data-cy=generate-button]').click();
      
      cy.get('[data-cy=dimension-warning]').should('be.visible');
      cy.get('[data-cy=dimension-warning]').should('contain', 'Large dimensions');
    });
  });

  describe('Image-to-Image Generation', () => {
    it('should complete image-to-image workflow', () => {
      cy.get('[data-cy=nav-generate]').click();
      cy.get('[data-cy=workflow-tabs]').contains('Image to Image').click();
      
      // Upload image
      const fileName = 'test-image.png';
      cy.get('[data-cy=image-upload]').selectFile(`cypress/fixtures/${fileName}`);
      
      // Verify image preview
      cy.get('[data-cy=image-preview]').should('be.visible');
      
      // Fill form
      cy.get('[data-cy=prompt-input]').type('Make it more colorful');
      cy.get('[data-cy=strength-slider]').invoke('val', 0.7).trigger('input');
      
      // Generate
      cy.get('[data-cy=generate-button]').click();
      
      cy.wait('@createJob').then((interception) => {
        expect(interception.request.body.type).to.equal('image_to_image');
        expect(interception.request.body.params).to.include({
          prompt: 'Make it more colorful',
          strength: 0.7,
        });
      });
    });

    it('should validate uploaded image', () => {
      cy.get('[data-cy=nav-generate]').click();
      cy.get('[data-cy=workflow-tabs]').contains('Image to Image').click();
      
      // Upload invalid file
      const fileName = 'invalid-file.txt';
      cy.get('[data-cy=image-upload]').selectFile(`cypress/fixtures/${fileName}`, {
        force: true,
      });
      
      cy.get('[data-cy=upload-error]').should('be.visible');
      cy.get('[data-cy=upload-error]').should('contain', 'Invalid file type');
    });
  });

  describe('Batch Generation', () => {
    it('should handle batch generation workflow', () => {
      cy.get('[data-cy=nav-generate]').click();
      cy.get('[data-cy=batch-toggle]').click();
      
      const prompts = [
        'A red sports car',
        'A blue ocean wave',
        'A green forest path',
      ];
      
      cy.get('[data-cy=batch-prompts]').type(prompts.join('\n'));
      
      // Check batch count
      cy.get('[data-cy=batch-count]').should('contain', '3 prompts');
      
      // Mock batch job creation
      cy.intercept('POST', '/api/v1/jobs/batch', {
        body: {
          batch_id: 'batch-123',
          job_ids: ['job-1', 'job-2', 'job-3'],
          total_credits: 30,
        },
      }).as('createBatchJob');
      
      cy.get('[data-cy=generate-button]').click();
      
      cy.wait('@createBatchJob');
      
      // Check batch progress
      cy.get('[data-cy=batch-progress]').should('be.visible');
      cy.get('[data-cy=batch-status]').should('contain', '0 of 3 completed');
    });
  });

  describe('Real-time Updates', () => {
    it('should receive real-time job updates via WebSocket', () => {
      // Mock WebSocket connection
      cy.window().then((win) => {
        const mockWebSocket = {
          send: cy.stub(),
          close: cy.stub(),
          addEventListener: cy.stub(),
          removeEventListener: cy.stub(),
        };
        
        cy.stub(win, 'WebSocket').returns(mockWebSocket);
      });
      
      cy.get('[data-cy=nav-generate]').click();
      cy.get('[data-cy=prompt-input]').type('Test prompt');
      cy.get('[data-cy=generate-button]').click();
      
      cy.wait('@createJob');
      
      // Simulate WebSocket messages
      cy.window().then((win) => {
        const updates = [
          { type: 'job_update', job_id: 'test-123', status: 'processing', progress: 0.3 },
          { type: 'job_update', job_id: 'test-123', status: 'processing', progress: 0.7 },
          { type: 'job_update', job_id: 'test-123', status: 'completed', progress: 1.0 },
        ];
        
        updates.forEach((update, index) => {
          setTimeout(() => {
            win.dispatchEvent(
              new CustomEvent('websocket-message', { detail: update })
            );
          }, index * 1000);
        });
      });
      
      // Check progress updates
      cy.get('[data-cy=progress-bar]')
        .should('have.attr', 'value', '30')
        .wait(1000)
        .should('have.attr', 'value', '70')
        .wait(1000)
        .should('have.attr', 'value', '100');
      
      cy.get('[data-cy=result-image]').should('be.visible');
    });
  });

  describe('Generation History', () => {
    it('should display and manage generation history', () => {
      cy.get('[data-cy=nav-history]').click();
      
      cy.intercept('GET', '/api/v1/user/history*', {
        fixture: 'generation-history.json',
      }).as('getHistory');
      
      cy.wait('@getHistory');
      
      // Check history items
      cy.get('[data-cy=history-item]').should('have.length', 5);
      cy.get('[data-cy=history-item]').first().should('contain', 'A beautiful sunset');
      
      // Test filtering
      cy.get('[data-cy=filter-status]').select('completed');
      cy.get('[data-cy=history-item]').should('have.length', 3);
      
      // Test search
      cy.get('[data-cy=search-input]').type('sunset');
      cy.get('[data-cy=history-item]').should('have.length', 1);
      
      // Test item actions
      cy.get('[data-cy=history-item]').first().click();
      cy.get('[data-cy=history-detail]').should('be.visible');
      
      cy.get('[data-cy=regenerate-button]').click();
      cy.url().should('include', '/generate');
      cy.get('[data-cy=prompt-input]').should('have.value', 'A beautiful sunset');
    });

    it('should delete history items', () => {
      cy.get('[data-cy=nav-history]').click();
      cy.wait('@getHistory');
      
      cy.intercept('DELETE', '/api/v1/jobs/job-1', {
        body: { message: 'Job deleted successfully' },
      }).as('deleteJob');
      
      cy.get('[data-cy=history-item]').first().find('[data-cy=delete-button]').click();
      cy.get('[data-cy=confirm-delete]').click();
      
      cy.wait('@deleteJob');
      
      cy.get('[data-cy=success-message]').should('contain', 'Job deleted');
      cy.get('[data-cy=history-item]').should('have.length', 4);
    });
  });

  describe('Settings and Preferences', () => {
    it('should save and apply user preferences', () => {
      cy.get('[data-cy=nav-settings]').click();
      
      // Change default model
      cy.get('[data-cy=default-model-select]').select('SDXL-Turbo');
      
      // Change quality settings
      cy.get('[data-cy=default-steps]').clear().type('30');
      cy.get('[data-cy=default-guidance]').clear().type('8.0');
      
      // Save settings
      cy.get('[data-cy=save-settings]').click();
      
      cy.get('[data-cy=success-message]').should('contain', 'Settings saved');
      
      // Verify settings applied
      cy.get('[data-cy=nav-generate]').click();
      cy.get('[data-cy=model-select]').should('have.value', 'sdxl-turbo');
      
      cy.get('[data-cy=advanced-toggle]').click();
      cy.get('[data-cy=steps-input]').should('have.value', '30');
      cy.get('[data-cy=guidance-input]').should('have.value', '8.0');
    });

    it('should manage API keys', () => {
      cy.get('[data-cy=nav-settings]').click();
      cy.get('[data-cy=api-keys-tab]').click();
      
      // Add new API key
      cy.get('[data-cy=add-api-key]').click();
      cy.get('[data-cy=api-key-name]').type('Test Key');
      cy.get('[data-cy=api-key-value]').type('sk-test-key-123');
      cy.get('[data-cy=save-api-key]').click();
      
      // Verify key added
      cy.get('[data-cy=api-key-item]').should('contain', 'Test Key');
      
      // Test key deletion
      cy.get('[data-cy=api-key-item]').first().find('[data-cy=delete-key]').click();
      cy.get('[data-cy=confirm-delete]').click();
      
      cy.get('[data-cy=api-key-item]').should('not.exist');
    });
  });

  describe('Responsive Design', () => {
    it('should work on mobile devices', () => {
      cy.viewport('iphone-x');
      
      cy.get('[data-cy=nav-generate]').click();
      
      // Check mobile layout
      cy.get('[data-cy=mobile-menu]').should('be.visible');
      cy.get('[data-cy=desktop-sidebar]').should('not.be.visible');
      
      // Test form interaction on mobile
      cy.get('[data-cy=prompt-input]').type('Mobile test prompt');
      cy.get('[data-cy=generate-button]').should('be.visible').click();
      
      cy.wait('@createJob');
      
      // Check mobile progress display
      cy.get('[data-cy=mobile-progress]').should('be.visible');
    });

    it('should adapt to tablet view', () => {
      cy.viewport('ipad-2');
      
      cy.get('[data-cy=nav-generate]').click();
      
      // Tablet should show condensed layout
      cy.get('[data-cy=tablet-layout]').should('be.visible');
      cy.get('[data-cy=sidebar]').should('be.visible');
      
      // Test touch interactions
      cy.get('[data-cy=advanced-toggle]').click();
      cy.get('[data-cy=quality-slider]').invoke('val', 0.8).trigger('input');
      
      cy.get('[data-cy=quality-value]').should('contain', '0.8');
    });
  });

  describe('Accessibility', () => {
    it('should be keyboard navigable', () => {
      // Test keyboard navigation
      cy.get('body').tab();
      cy.focused().should('have.attr', 'data-cy', 'nav-generate');
      
      cy.focused().tab();
      cy.focused().should('have.attr', 'data-cy', 'nav-history');
      
      // Navigate to form
      cy.get('[data-cy=nav-generate]').click();
      cy.get('[data-cy=prompt-input]').focus().type('Keyboard navigation test');
      
      // Tab through form elements
      cy.focused().tab();
      cy.focused().should('have.attr', 'data-cy', 'model-select');
      
      cy.focused().tab();
      cy.focused().should('have.attr', 'data-cy', 'generate-button');
      
      // Activate with Enter key
      cy.focused().type('{enter}');
      cy.wait('@createJob');
    });

    it('should have proper ARIA labels', () => {
      cy.get('[data-cy=nav-generate]').click();
      
      cy.get('[data-cy=prompt-input]').should('have.attr', 'aria-label', 'Prompt input');
      cy.get('[data-cy=model-select]').should('have.attr', 'aria-label', 'Model selection');
      cy.get('[data-cy=generate-button]').should('have.attr', 'aria-label', 'Generate image');
      
      cy.get('[data-cy=advanced-toggle]').click();
      cy.get('[data-cy=progress-bar]').should('have.attr', 'role', 'progressbar');
    });

    it('should work with screen readers', () => {
      cy.get('[data-cy=nav-generate]').click();
      
      // Check for screen reader text
      cy.get('[data-cy=sr-only]').should('contain', 'Image generation form');
      
      cy.get('[data-cy=prompt-input]').type('Screen reader test');
      cy.get('[data-cy=generate-button]').click();
      
      cy.wait('@createJob');
      
      // Check progress announcements
      cy.get('[data-cy=sr-progress]').should('contain', 'Generation in progress');
    });
  });
});