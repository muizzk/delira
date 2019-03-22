from delira import get_backends

if "TORCH" in get_backends():
    from .generative_adversarial_network import \
        GenerativeAdversarialNetworkBasePyTorch

if "TF" in get_backends():
    from.generative_adversarial_network_tf import \
        GenerativeAdversarialNetworkBaseTf