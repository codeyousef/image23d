# Tips for Faster Hunyuan3D Downloads

## 1. Use hunyuan3d-2mini instead of standard
- Only ~8GB vs ~25GB (actually seems to be 100GB+)
- Still produces good quality results
- Downloads 3-10x faster

## 2. Ensure hf_transfer is installed
```bash
pip install hf_transfer
```
This can speed up downloads by 2-5x.

## 3. Download during off-peak hours
- Try downloading late at night or early morning
- HuggingFace servers are less congested

## 4. Use a wired connection
- WiFi can be unstable for large downloads
- Ethernet provides more consistent speeds

## 5. Check your download location
- Ensure you have enough disk space (150GB+ for standard model)
- Use an SSD if possible for faster writes

## 6. Alternative: Download manually
If the app downloads are too slow, you can:
1. Go to https://huggingface.co/tencent/Hunyuan3D-2
2. Download the files manually using your browser or a download manager
3. Place them in the correct directory structure

## 7. Consider using cloud services
- Use Google Colab or other cloud services with fast internet
- Download there and transfer to your local machine

## Current download speed improvements in the app:
- ✅ Added hf_transfer support (2-5x faster)
- ✅ Increased max_workers to 8 for parallel downloads
- ✅ Added automatic retry on network failures
- ✅ Resume capability for interrupted downloads