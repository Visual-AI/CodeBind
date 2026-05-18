# from models.VQ.looc import VectorQuantizer
from models.VQ.multimodal_quantise import VectorQuantizer, MultimodalVectorQuantizer


def build_vq(args):
    if args.get('use_mvq'):
        return MultimodalVectorQuantizer(args)
    else:
        return VectorQuantizer(args.get('single_vq_embed_num'), args.get('single_vq_embed_dim'), args=args)
