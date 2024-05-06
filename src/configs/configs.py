cropper = {
    "path_to_model": "./saved_models/corner/detect.tflite",
    "path_to_labels": "./saved_models/corner/labelmap.txt",
    "nms_ths": 0.2,
    "min_conf": 0.3,
    "debug": True,
}

dectector = {
    "path_to_model": "./saved_models/detector/detect.tflite",
    "path_to_labels": "./saved_models/detector/labelmap.txt",
    "nms_ths": 0.2,
    "min_conf": 0.3,
    "debug": True,
}

reader = {
    "base_config": "./saved_models/vietocr/base.yml",
    "vgg_config": "./saved_models/vietocr/vgg-transformer.yml",
    "model_weight": "./saved_models/vietocr/vgg_transformer.pth",
    "device": "cpu",
}
