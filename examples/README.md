# Using Collagen

- [x] Classification
- [ ] Segmentation
- [x] GAN
- [x] SSGAN
- [x] PI model (SSL)
- [x] Mean Teacher (EMA)
- [x] Stochastic Weight Averaging (SWA)
- [x] MixMatch (with and without EMA)
- [ ] Metric Learning
- [x] AutoEncoder
- [ ] VAE
- [ ] VAE-SSL
- [ ] VAE-GAN


# Dependencies

- SOLT (> 0.1.9)
- Hydra

# Create default configurations

```shell script
python -c "from collagen.core.utils import create_default_config; create_default_config(<root_dir>)"
``` 