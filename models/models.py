import torch

def create_model(opt):
    if opt.model in ['pix2pixHD','colorization']:
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel, ColorizationModel
        if opt.isTrain:
            if opt.model == 'colorization':
                model = ColorizationModel()
            else:
                model = Pix2PixHDModel()
        else:
            if opt.model == 'colorization':
                model = ColorizationModel()
            else:
                model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
