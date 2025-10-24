from face_recognize import FaceExtractor

class SingletonFaceExtractor:
    _instance = None
    _model_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._model_loaded:
            # 在这里加载模型
            self.extractor = FaceExtractor()
            self._model_loaded = True

    def extract_features(self, image):
        return self.extractor.extract_features(image)

# 创建全局实例
face_extractor = SingletonFaceExtractor()