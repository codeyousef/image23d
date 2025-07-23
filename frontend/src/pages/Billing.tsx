import { useState, useEffect } from 'react'
import { CreditCard, Plus, History, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/components/ui/use-toast'
import { useAuthStore } from '@/store/authStore'
import { api } from '@/lib/api'

interface CreditPackage {
  id: string
  name: string
  credits: number
  price: number
  popular?: boolean
}

interface Transaction {
  id: string
  type: 'purchase' | 'usage'
  amount: number
  description: string
  created_at: string
}

export default function BillingPage() {
  const { toast } = useToast()
  const user = useAuthStore((state) => state.user)
  const [creditPackages, setCreditPackages] = useState<CreditPackage[]>([])
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [purchasingPackageId, setPurchasingPackageId] = useState<string | null>(null)

  useEffect(() => {
    fetchBillingData()
  }, [])

  const fetchBillingData = async () => {
    setIsLoading(true)
    try {
      const [packagesRes, transactionsRes] = await Promise.all([
        api.get('/api/billing/packages'),
        api.get('/api/billing/transactions')
      ])
      setCreditPackages(packagesRes.data.packages)
      setTransactions(transactionsRes.data.transactions)
    } catch (error) {
      toast({
        title: 'Failed to load billing data',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handlePurchase = async (packageId: string) => {
    setPurchasingPackageId(packageId)
    try {
      const response = await api.post('/api/billing/purchase', { package_id: packageId })
      
      // In a real app, this would redirect to a payment gateway
      // For demo, we'll simulate a successful purchase
      if (response.data.payment_url) {
        window.location.href = response.data.payment_url
      } else {
        // Demo mode - instant credit
        toast({
          title: 'Credits added successfully!',
          description: 'Your account has been credited.',
        })
        // Refresh user data
        const userRes = await api.get('/api/auth/me')
        useAuthStore.getState().setUser(userRes.data)
        fetchBillingData()
      }
    } catch (error) {
      toast({
        title: 'Purchase failed',
        variant: 'destructive',
      })
    } finally {
      setPurchasingPackageId(null)
    }
  }

  const creditUsagePercentage = user ? Math.min((user.credits_used || 0) / (user.credits + (user.credits_used || 0)) * 100, 100) : 0

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Billing & Credits</h1>
        <p className="text-muted-foreground">
          Manage your credits and view transaction history
        </p>
      </div>

      {/* Current Balance */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Credit Balance</CardTitle>
            <CardDescription>Your current credit balance and usage</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold">{user?.credits || 0}</p>
                <p className="text-sm text-muted-foreground">Credits remaining</p>
              </div>
              <TrendingUp className="h-8 w-8 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Usage this month</span>
                <span>{user?.credits_used || 0} credits</span>
              </div>
              <Progress value={creditUsagePercentage} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Stats</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">Images generated</span>
              <span className="font-medium">42</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">3D models created</span>
              <span className="font-medium">8</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">Avg. cost per asset</span>
              <span className="font-medium">2.5 credits</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Credit Packages */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Purchase Credits</h2>
        {isLoading ? (
          <div className="text-center py-8">
            <p className="text-muted-foreground">Loading packages...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {creditPackages.map((pkg) => (
              <Card key={pkg.id} className={pkg.popular ? 'border-primary' : ''}>
                {pkg.popular && (
                  <div className="bg-primary text-primary-foreground text-center py-1 text-sm font-medium">
                    Most Popular
                  </div>
                )}
                <CardHeader>
                  <CardTitle>{pkg.name}</CardTitle>
                  <div className="text-3xl font-bold">${pkg.price}</div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold">{pkg.credits}</p>
                    <p className="text-sm text-muted-foreground">Credits</p>
                  </div>
                  <Button
                    className="w-full"
                    variant={pkg.popular ? 'default' : 'outline'}
                    onClick={() => handlePurchase(pkg.id)}
                    disabled={purchasingPackageId === pkg.id}
                  >
                    {purchasingPackageId === pkg.id ? (
                      'Processing...'
                    ) : (
                      <>
                        <Plus className="mr-2 h-4 w-4" />
                        Purchase
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Transaction History */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Transaction History</h2>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Recent Transactions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {transactions.length === 0 ? (
              <p className="text-center py-8 text-muted-foreground">No transactions yet</p>
            ) : (
              <div className="space-y-2">
                {transactions.map((transaction) => (
                  <div key={transaction.id} className="flex items-center justify-between py-2 border-b last:border-0">
                    <div>
                      <p className="font-medium">{transaction.description}</p>
                      <p className="text-sm text-muted-foreground">
                        {new Date(transaction.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div className={`font-medium ${transaction.type === 'purchase' ? 'text-green-600' : 'text-red-600'}`}>
                      {transaction.type === 'purchase' ? '+' : '-'}{transaction.amount} credits
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}