# 🚀 PickleTrack Deployment Guide

## 📱 Streamlit Cloud Deployment

### Quick Deploy
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy PickleTrack"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Visit: [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub account
   - Select repository: `pickletrack`
   - Main file: `app.py`
   - Requirements: `requirements.txt`
   - Click "Deploy"

3. **Expected URL**
   - `https://pickletrack-[username].streamlit.app`

### Configuration Notes
- Demo data included in repository
- `.streamlit/config.toml` auto-detected
- No additional setup required

## 🌐 Portfolio Deployment

### Next.js Portfolio (Vercel)
```bash
cd web/portfolio
npm install
npm run build
```

Deploy via Vercel:
- Import GitHub repository
- Select `web/portfolio` folder
- Auto-deploy

### Expected URLs
- **Portfolio**: `https://[username].vercel.app`
- **PickleTrack Page**: `https://[username].vercel.app/projects/pickletrack`

## 🔗 Integration

1. Update portfolio with live dashboard URL
2. Test all links and functionality
3. Verify demo mode works correctly

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Local dashboard test: `streamlit run app.py`
- [ ] Portfolio build test: `npm run build`
- [ ] Demo data loads correctly
- [ ] All features functional

### Post-Deployment
- [ ] Live dashboard accessible
- [ ] Demo mode functional
- [ ] Portfolio links to dashboard
- [ ] Mobile responsive
- [ ] Professional presentation ready

## 🎯 Success Criteria

✅ Live, accessible dashboard  
✅ Professional portfolio integration  
✅ Seamless user experience  
✅ Employer-ready demonstration  
✅ Technical showcase complete