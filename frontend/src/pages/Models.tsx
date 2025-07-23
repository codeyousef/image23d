import { useState, useEffect } from 'react'
import { Download, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/components/ui/use-toast'
import { api } from '@/lib/api'

interface Model {
  id: string
  name: string
  description: string
  type: 'image' | '3d' | 'video'
  size: string
  vram_required: string
  status: 'available' | 'downloading' | 'downloaded' | 'failed'
  download_progress?: number
}

export default function ModelsPage() {
  const { toast } = useToast()
  const [models, setModels] = useState<Model[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    setIsLoading(true)
    try {
      const response = await api.get('/api/models')
      setModels(response.data.models)
    } catch (error) {
      toast({
        title: 'Failed to load models',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = async (modelId: string) => {
    try {
      // Update local state
      setModels(models.map(m => 
        m.id === modelId ? { ...m, status: 'downloading', download_progress: 0 } : m
      ))

      const response = await api.post(`/api/models/${modelId}/download`)
      
      // Poll for download progress
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await api.get(`/api/models/${modelId}/status`)
          const model = statusResponse.data
          
          setModels(prev => prev.map(m => 
            m.id === modelId ? { ...m, ...model } : m
          ))

          if (model.status === 'downloaded' || model.status === 'failed') {
            clearInterval(pollInterval)
            if (model.status === 'downloaded') {
              toast({
                title: 'Model downloaded successfully',
              })
            } else {
              toast({
                title: 'Download failed',
                variant: 'destructive',
              })
            }
          }
        } catch (error) {
          clearInterval(pollInterval)
        }
      }, 2000)
    } catch (error) {
      toast({
        title: 'Failed to start download',
        variant: 'destructive',
      })
    }
  }

  const getStatusIcon = (status: Model['status']) => {
    switch (status) {
      case 'downloaded':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'downloading':
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return null
    }
  }

  const getTypeColor = (type: Model['type']) => {
    switch (type) {
      case 'image':
        return 'default'
      case '3d':
        return 'secondary'
      case 'video':
        return 'outline'
      default:
        return 'default'
    }
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Models</h1>
        <p className="text-muted-foreground">
          Download and manage AI models
        </p>
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading models...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <Card key={model.id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-lg">{model.name}</CardTitle>
                    <div className="flex items-center gap-2">
                      <Badge variant={getTypeColor(model.type)}>
                        {model.type.toUpperCase()}
                      </Badge>
                      {getStatusIcon(model.status)}
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <CardDescription>{model.description}</CardDescription>
                
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Size:</span>
                    <span>{model.size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">VRAM Required:</span>
                    <span>{model.vram_required}</span>
                  </div>
                </div>

                {model.status === 'downloading' && model.download_progress !== undefined && (
                  <div className="space-y-2">
                    <Progress value={model.download_progress} />
                    <p className="text-xs text-center text-muted-foreground">
                      {model.download_progress}% complete
                    </p>
                  </div>
                )}

                <Button
                  className="w-full"
                  disabled={model.status === 'downloading' || model.status === 'downloaded'}
                  onClick={() => handleDownload(model.id)}
                >
                  {model.status === 'downloading' ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Downloading...
                    </>
                  ) : model.status === 'downloaded' ? (
                    <>
                      <CheckCircle className="mr-2 h-4 w-4" />
                      Downloaded
                    </>
                  ) : (
                    <>
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}