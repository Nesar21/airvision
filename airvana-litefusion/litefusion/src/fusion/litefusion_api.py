class FusedAQI:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

class LiteFusionPredictor:
    def __init__(self):
        self.sigma_img = 8.0
        self.sigma_num = 10.0
        self.sigma_news = 15.0

    def fuse_aqi(self, image=None, numeric=None, news=None):
        means = []
        vars_ = []

        if image is not None:
            m, s = image
            means.append(m)
            vars_.append(s*s)

        if numeric is not None:
            m, s = numeric
            means.append(m)
            vars_.append(s*s)

        if news is not None:
            m, s = news
            means.append(m)
            vars_.append(s*s)

        if not means:
            return None

        w = [1/v for v in vars_]
        fused_mean = sum(w[i]*means[i] for i in range(len(w))) / sum(w)
        fused_sigma = 1.0 / sum(w)

        return FusedAQI(fused_mean, fused_sigma)
