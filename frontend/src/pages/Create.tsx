import { useState, useEffect } from 'react'
import { Sparkles, Image as ImageIcon, Box, Video } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/components/ui/use-toast'
import { api } from '@/lib/api'
import { useAuthStore } from '@/store/authStore'
import { useWebSocket } from '@/hooks/useWebSocket'

export default function CreatePage() {
  const { toast } = useToast()
  const user = useAuthStore((state) => state.user)
  const { lastMessage, isConnected } = useWebSocket()
  const [activeTab, setActiveTab] = useState('image')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [generationProgress, setGenerationProgress] = useState(0)
  
  // Image generation state
  const [imagePrompt, setImagePrompt] = useState('')
  const [imageModel, setImageModel] = useState('flux-schnell')
  const [imageSettings, setImageSettings] = useState({
    width: 1024,
    height: 1024,
    num_inference_steps: 4,
    guidance_scale: 0,
    seed: -1,
  })
  
  // 3D generation state
  const [threeDPrompt, setThreeDPrompt] = useState('')
  const [threeDModel, setThreeDModel] = useState('hunyuan3d-2.1')
  const [threeDSettings, setThreeDSettings] = useState({
    quality_preset: 'standard',
    export_format: 'glb',
    texture_resolution: 2048,
  })

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage && currentJobId) {
      if (lastMessage.type === 'job_update' && lastMessage.job_id === currentJobId) {
        if (lastMessage.status === 'completed') {
          setIsGenerating(false)
          setCurrentJobId(null)
          setGenerationProgress(100)
          if (lastMessage.data?.image_url) {
            setGeneratedImage(lastMessage.data.image_url)
          }
          toast({
            title: 'Generation completed!',
          })
        } else if (lastMessage.status === 'failed') {
          setIsGenerating(false)
          setCurrentJobId(null)
          setGenerationProgress(0)
          toast({
            title: 'Generation failed',
            description: lastMessage.message || 'Unknown error',
            variant: 'destructive',
          })
        }
      } else if (lastMessage.type === 'generation_progress' && lastMessage.job_id === currentJobId) {
        setGenerationProgress(lastMessage.progress || 0)
      }
    }
  }, [lastMessage, currentJobId, toast])

  const handleImageGenerate = async () => {
    if (!imagePrompt.trim()) {
      toast({
        title: 'Please enter a prompt',
        variant: 'destructive',
      })
      return
    }

    setIsGenerating(true)
    setGeneratedImage(null)
    setGenerationProgress(0)

    try {
      const response = await api.post('/api/generation/image', {
        prompt: imagePrompt,
        model: imageModel,
        settings: imageSettings,
      })

      const jobId = response.data.job_id
      setCurrentJobId(jobId)

      // If WebSocket is not connected, fall back to polling
      if (!isConnected) {
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await api.get(`/api/jobs/${jobId}`)
            const job = statusResponse.data

            if (job.status === 'completed') {
              clearInterval(pollInterval)
              setIsGenerating(false)
              setCurrentJobId(null)
              if (job.result?.image_url) {
                setGeneratedImage(job.result.image_url)
              }
              toast({
                title: 'Image generated successfully!',
              })
            } else if (job.status === 'failed') {
              clearInterval(pollInterval)
              setIsGenerating(false)
              setCurrentJobId(null)
              toast({
                title: 'Generation failed',
                description: job.error || 'Unknown error',
                variant: 'destructive',
              })
            } else if (job.progress) {
              setGenerationProgress(job.progress)
            }
          } catch (error) {
            clearInterval(pollInterval)
            setIsGenerating(false)
            setCurrentJobId(null)
            console.error('Error polling job status:', error)
          }
        }, 2000)
      }
    } catch (error: any) {
      setIsGenerating(false)
      toast({
        title: 'Generation failed',
        description: error.response?.data?.detail || 'Failed to generate image',
        variant: 'destructive',
      })
    }
  }

  const handle3DGenerate = async () => {
    if (!threeDPrompt.trim()) {
      toast({
        title: 'Please enter a prompt',
        variant: 'destructive',
      })
      return
    }

    setIsGenerating(true)
    setGenerationProgress(0)

    try {
      const response = await api.post('/api/generation/3d', {
        prompt: threeDPrompt,
        model: threeDModel,
        settings: threeDSettings,
      })

      const jobId = response.data.job_id
      setCurrentJobId(jobId)

      // If WebSocket is not connected, fall back to polling
      if (!isConnected) {
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await api.get(`/api/jobs/${jobId}`)
            const job = statusResponse.data

            if (job.status === 'completed') {
              clearInterval(pollInterval)
              setIsGenerating(false)
              setCurrentJobId(null)
              toast({
                title: '3D model generated successfully!',
                description: 'Check your library to view and download the model.',
              })
            } else if (job.status === 'failed') {
              clearInterval(pollInterval)
              setIsGenerating(false)
              setCurrentJobId(null)
              toast({
                title: 'Generation failed',
                description: job.error || 'Unknown error',
                variant: 'destructive',
              })
            } else if (job.progress) {
              setGenerationProgress(job.progress)
            }
          } catch (error) {
            clearInterval(pollInterval)
            setIsGenerating(false)
            setCurrentJobId(null)
            console.error('Error polling job status:', error)
          }
        }, 2000)
      }
    } catch (error: any) {
      setIsGenerating(false)
      toast({
        title: 'Generation failed',
        description: error.response?.data?.detail || 'Failed to generate 3D model',
        variant: 'destructive',
      })
    }
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Create</h1>
        <p className="text-muted-foreground">
          Generate amazing AI content with our powerful models. You have {user?.credits || 0} credits remaining.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3 max-w-md">
          <TabsTrigger value="image" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Image
          </TabsTrigger>
          <TabsTrigger value="3d" className="flex items-center gap-2">
            <Box className="h-4 w-4" />
            3D Model
          </TabsTrigger>
          <TabsTrigger value="video" className="flex items-center gap-2" disabled>
            <Video className="h-4 w-4" />
            Video
          </TabsTrigger>
        </TabsList>

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column - Input */}
          <div>
            <TabsContent value="image" className="mt-0">
              <Card>
                <CardHeader>
                  <CardTitle>Image Generation</CardTitle>
                  <CardDescription>
                    Create stunning images from text descriptions
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="image-prompt">Prompt</Label>
                    <Textarea
                      id="image-prompt"
                      placeholder="A futuristic city at sunset with flying cars..."
                      value={imagePrompt}
                      onChange={(e) => setImagePrompt(e.target.value)}
                      className="mt-2 min-h-[100px]"
                    />
                  </div>

                  <div>
                    <Label htmlFor="image-model">Model</Label>
                    <Select value={imageModel} onValueChange={setImageModel}>
                      <SelectTrigger id="image-model" className="mt-2">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="flux-schnell">FLUX Schnell (Fast)</SelectItem>
                        <SelectItem value="flux-dev">FLUX Dev (Quality)</SelectItem>
                        <SelectItem value="stable-diffusion-xl">Stable Diffusion XL</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="width">Width</Label>
                      <Input
                        id="width"
                        type="number"
                        value={imageSettings.width}
                        onChange={(e) => setImageSettings({ ...imageSettings, width: parseInt(e.target.value) })}
                        className="mt-2"
                      />
                    </div>
                    <div>
                      <Label htmlFor="height">Height</Label>
                      <Input
                        id="height"
                        type="number"
                        value={imageSettings.height}
                        onChange={(e) => setImageSettings({ ...imageSettings, height: parseInt(e.target.value) })}
                        className="mt-2"
                      />
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="steps">Steps: {imageSettings.num_inference_steps}</Label>
                    <Slider
                      id="steps"
                      value={[imageSettings.num_inference_steps]}
                      onValueChange={(value) => setImageSettings({ ...imageSettings, num_inference_steps: value[0] })}
                      max={50}
                      min={1}
                      className="mt-2"
                    />
                  </div>

                  <Button 
                    onClick={handleImageGenerate} 
                    disabled={isGenerating}
                    className="w-full"
                  >
                    {isGenerating ? (
                      <>Generating...</>
                    ) : (
                      <>
                        <Sparkles className="mr-2 h-4 w-4" />
                        Generate Image
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="3d" className="mt-0">
              <Card>
                <CardHeader>
                  <CardTitle>3D Model Generation</CardTitle>
                  <CardDescription>
                    Transform text or images into 3D models
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="3d-prompt">Prompt</Label>
                    <Textarea
                      id="3d-prompt"
                      placeholder="A cute cartoon robot with big eyes..."
                      value={threeDPrompt}
                      onChange={(e) => setThreeDPrompt(e.target.value)}
                      className="mt-2 min-h-[100px]"
                    />
                  </div>

                  <div>
                    <Label htmlFor="3d-model">Model</Label>
                    <Select value={threeDModel} onValueChange={setThreeDModel}>
                      <SelectTrigger id="3d-model" className="mt-2">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hunyuan3d-2.1">HunyuanVideo 3D v2.1</SelectItem>
                        <SelectItem value="hunyuan3d-2.0">HunyuanVideo 3D v2.0</SelectItem>
                        <SelectItem value="hunyuan3d-mini">HunyuanVideo 3D Mini</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="quality">Quality Preset</Label>
                    <Select 
                      value={threeDSettings.quality_preset} 
                      onValueChange={(value) => setThreeDSettings({ ...threeDSettings, quality_preset: value })}
                    >
                      <SelectTrigger id="quality" className="mt-2">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="draft">Draft (Fast)</SelectItem>
                        <SelectItem value="standard">Standard</SelectItem>
                        <SelectItem value="high">High Quality</SelectItem>
                        <SelectItem value="ultra">Ultra (Slow)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="format">Export Format</Label>
                    <Select 
                      value={threeDSettings.export_format} 
                      onValueChange={(value) => setThreeDSettings({ ...threeDSettings, export_format: value })}
                    >
                      <SelectTrigger id="format" className="mt-2">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="glb">GLB (Recommended)</SelectItem>
                        <SelectItem value="obj">OBJ</SelectItem>
                        <SelectItem value="fbx">FBX</SelectItem>
                        <SelectItem value="usdz">USDZ (iOS)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button 
                    onClick={handle3DGenerate} 
                    disabled={isGenerating}
                    className="w-full"
                  >
                    {isGenerating ? (
                      <>Generating...</>
                    ) : (
                      <>
                        <Sparkles className="mr-2 h-4 w-4" />
                        Generate 3D Model
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="video" className="mt-0">
              <Card>
                <CardHeader>
                  <CardTitle>Video Generation</CardTitle>
                  <CardDescription>
                    Coming soon - Create videos from text
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground text-center py-8">
                    Video generation will be available in a future update
                  </p>
                </CardContent>
              </Card>
            </TabsContent>
          </div>

          {/* Right column - Output */}
          <div>
            <Card className="h-full min-h-[500px]">
              <CardHeader>
                <CardTitle>Output</CardTitle>
                <CardDescription>
                  Your generated content will appear here
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isGenerating ? (
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center space-y-4">
                      <Sparkles className="h-12 w-12 mx-auto animate-pulse text-primary" />
                      <div className="space-y-2">
                        <p className="text-muted-foreground">Generating your content...</p>
                        <Progress value={generationProgress} className="w-48 mx-auto" />
                        <p className="text-sm text-muted-foreground">{generationProgress}%</p>
                      </div>
                      {!isConnected && (
                        <p className="text-xs text-yellow-600">WebSocket disconnected, using polling</p>
                      )}
                    </div>
                  </div>
                ) : generatedImage ? (
                  <div className="space-y-4">
                    <img 
                      src={generatedImage} 
                      alt="Generated" 
                      className="w-full rounded-lg"
                    />
                    <Button 
                      variant="outline" 
                      className="w-full"
                      onClick={() => window.open(generatedImage, '_blank')}
                    >
                      View Full Size
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-96 text-muted-foreground">
                    Generate something amazing!
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </Tabs>
    </div>
  )
}