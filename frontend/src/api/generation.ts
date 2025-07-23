import { apiClient } from './client'

export type ExecutionMode = 'auto' | 'local' | 'serverless'
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface ImageGenerationRequest {
  prompt: string
  negative_prompt?: string
  model: string
  width: number
  height: number
  steps: number
  guidance_scale: number
  seed?: number
  enhancement_fields?: Record<string, any>
  use_enhancement: boolean
  execution_mode: ExecutionMode
}

export interface ThreeDGenerationRequest {
  prompt?: string
  image_url?: string
  model: string
  quality_preset: string
  export_formats: string[]
  remove_background: boolean
  enhancement_fields?: Record<string, any>
  use_enhancement: boolean
  execution_mode: ExecutionMode
}

export interface GenerationResponse {
  job_id: string
  status: JobStatus
  message: string
  estimated_cost?: number
  estimated_time?: number
}

export interface EnhancePromptRequest {
  prompt: string
  model_type: string
  fields?: Record<string, any>
}

export interface EnhancePromptResponse {
  original: string
  enhanced: string
  credits_used: number
}

export interface Model {
  id: string
  name: string
  description: string
  vram_required: string
}

export interface EnhancementFields {
  [key: string]: {
    label: string
    type?: string
    options: Record<string, string>
  }
}

export const generationApi = {
  generateImage: async (data: ImageGenerationRequest): Promise<GenerationResponse> => {
    const response = await apiClient.post('/generate/image', data)
    return response.data
  },

  generate3D: async (data: ThreeDGenerationRequest): Promise<GenerationResponse> => {
    const response = await apiClient.post('/generate/3d', data)
    return response.data
  },

  enhancePrompt: async (data: EnhancePromptRequest): Promise<EnhancePromptResponse> => {
    const response = await apiClient.post('/generate/enhance-prompt', {
      prompt: data.prompt,
      model_type: data.model_type,
      fields: data.fields,
    })
    return response.data
  },

  getModels: async (): Promise<{ image: Model[], '3d': Model[] }> => {
    const response = await apiClient.get('/generate/models')
    return response.data
  },

  getEnhancementFields: async (modelType: string): Promise<EnhancementFields> => {
    const response = await apiClient.get(`/generate/enhancement-fields/${modelType}`)
    return response.data
  },
}