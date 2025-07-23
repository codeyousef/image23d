import { useState, useEffect } from 'react'
import { Image as ImageIcon, Box, Download, Trash2, Eye } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/components/ui/use-toast'
import { api } from '@/lib/api'

interface Asset {
  id: string
  type: 'image' | '3d' | 'video'
  prompt: string
  model: string
  created_at: string
  file_url: string
  thumbnail_url?: string
  metadata: any
}

export default function LibraryPage() {
  const { toast } = useToast()
  const [assets, setAssets] = useState<Asset[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('all')

  useEffect(() => {
    fetchAssets()
  }, [])

  const fetchAssets = async () => {
    setIsLoading(true)
    try {
      const response = await api.get('/api/library')
      setAssets(response.data.assets)
    } catch (error) {
      toast({
        title: 'Failed to load assets',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = async (asset: Asset) => {
    try {
      const response = await fetch(asset.file_url)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${asset.id}.${asset.type === 'image' ? 'png' : 'glb'}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      toast({
        title: 'Download failed',
        variant: 'destructive',
      })
    }
  }

  const handleDelete = async (assetId: string) => {
    if (!confirm('Are you sure you want to delete this asset?')) return

    try {
      await api.delete(`/api/library/${assetId}`)
      setAssets(assets.filter(a => a.id !== assetId))
      toast({
        title: 'Asset deleted',
      })
    } catch (error) {
      toast({
        title: 'Failed to delete asset',
        variant: 'destructive',
      })
    }
  }

  const filteredAssets = assets.filter(asset => {
    if (activeTab === 'all') return true
    return asset.type === activeTab
  })

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Library</h1>
        <p className="text-muted-foreground">
          Your generated assets
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="image" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Images
          </TabsTrigger>
          <TabsTrigger value="3d" className="flex items-center gap-2">
            <Box className="h-4 w-4" />
            3D Models
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab}>
          {isLoading ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">Loading assets...</p>
            </div>
          ) : filteredAssets.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No assets found</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredAssets.map((asset) => (
                <Card key={asset.id} className="overflow-hidden">
                  <CardContent className="p-0">
                    {asset.type === 'image' ? (
                      <img
                        src={asset.thumbnail_url || asset.file_url}
                        alt={asset.prompt}
                        className="w-full h-48 object-cover"
                      />
                    ) : (
                      <div className="w-full h-48 bg-muted flex items-center justify-center">
                        <Box className="h-12 w-12 text-muted-foreground" />
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="p-4 flex flex-col space-y-2">
                    <p className="text-sm text-muted-foreground truncate w-full">
                      {asset.prompt}
                    </p>
                    <div className="flex items-center justify-between w-full text-xs text-muted-foreground">
                      <span>{asset.model}</span>
                      <span>{new Date(asset.created_at).toLocaleDateString()}</span>
                    </div>
                    <div className="flex gap-2 w-full">
                      {asset.type === 'image' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => window.open(asset.file_url, '_blank')}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownload(asset)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDelete(asset.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}